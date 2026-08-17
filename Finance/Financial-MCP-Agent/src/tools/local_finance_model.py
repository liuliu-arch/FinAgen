"""LangChain tool backed by the locally fine-tuned Qwen3 QLoRA adapters."""

import json
import os
import threading
from pathlib import Path

import torch
from langchain_core.tools import tool
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


FINANCE_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = Path(os.getenv("FINANCE_MODEL_PATH", FINANCE_ROOT / "Qwen3-8B"))
SENTIMENT_ADAPTER = Path(
    os.getenv(
        "FINANCE_SENTIMENT_ADAPTER",
        FINANCE_ROOT / "qwen3_8b_sentiment_final",
    )
)
RISK_ADAPTER = Path(
    os.getenv("FINANCE_RISK_ADAPTER", FINANCE_ROOT / "qwen3_8b_risk_final")
)

PROMPTS = {
    "sentiment": (
        "You are a financial news sentiment analyst. Given one summarized news item "
        "about a stock, return exactly one integer score from 1 to 5: "
        "1=negative, 2=somewhat negative, 3=neutral, "
        "4=somewhat positive, 5=positive. Return only the score."
    ),
    "risk": (
        "You are a financial risk analyst. Given one summarized news item about a stock, "
        "return exactly one integer risk score from 1 to 5: "
        "1=very low risk, 2=low risk, 3=moderate risk, "
        "4=high risk, 5=very high risk. Return only the score."
    ),
}


class LocalFinanceScorer:
    def __init__(self):
        for path in (MODEL_PATH, SENTIMENT_ADAPTER, RISK_ADAPTER):
            if not path.exists():
                raise FileNotFoundError(f"Local model path does not exist: {path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=quantization_config,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        self.model = PeftModel.from_pretrained(
            base_model,
            SENTIMENT_ADAPTER,
            adapter_name="sentiment",
            is_trainable=False,
        )
        self.model.load_adapter(
            RISK_ADAPTER,
            adapter_name="risk",
            is_trainable=False,
        )
        self.model.eval()
        self.digit_tokens = self._digit_token_ids()
        self.lock = threading.Lock()

    def _digit_token_ids(self):
        result = {}
        for score in range(1, 6):
            token_ids = self.tokenizer.encode(str(score), add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(f"Score {score} is not represented by one token")
            result[score] = token_ids[0]
        return result

    def _prompt(self, task, symbol, news):
        messages = [
            {"role": "system", "content": PROMPTS[task]},
            {
                "role": "user",
                "content": f"Stock symbol: {symbol}\nSummarized news: {news}",
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _score(self, adapter_name, task, symbol, news):
        self.model.set_adapter(adapter_name)
        prompt = self._prompt(task, symbol, news)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1]

        scores = list(self.digit_tokens)
        candidate_logits = torch.stack(
            [logits[self.digit_tokens[score]] for score in scores]
        )
        probabilities = torch.softmax(candidate_logits.float(), dim=0).cpu().tolist()
        best_index = int(torch.argmax(candidate_logits).item())
        return {
            "score": scores[best_index],
            "probabilities": {
                str(score): round(probability, 4)
                for score, probability in zip(scores, probabilities)
            },
        }

    def analyze(self, stock_symbol, summarized_news):
        with self.lock:
            return {
                "stock_symbol": stock_symbol,
                "sentiment": self._score(
                    "sentiment",
                    "sentiment",
                    stock_symbol,
                    summarized_news,
                ),
                "risk": self._score(
                    "risk",
                    "risk",
                    stock_symbol,
                    summarized_news,
                ),
            }


_scorer = None
_scorer_init_lock = threading.Lock()


def get_local_finance_scorer():
    global _scorer
    if _scorer is None:
        with _scorer_init_lock:
            if _scorer is None:
                _scorer = LocalFinanceScorer()
    return _scorer


@tool
def score_financial_news(stock_symbol: str, summarized_news: str) -> str:
    """Score one real financial news summary with the local fine-tuned models.

    Use this tool once for every news item after obtaining the article and creating
    a concise factual summary. It returns sentiment and risk scores from 1 to 5.
    The returned model scores must be used verbatim in the final news report.
    """
    result = get_local_finance_scorer().analyze(stock_symbol, summarized_news)
    return json.dumps(result, ensure_ascii=False)
