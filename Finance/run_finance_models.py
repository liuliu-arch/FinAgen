import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run sentiment and risk QLoRA adapters on one Qwen3-8B base model"
    )
    parser.add_argument("--model-path", default="./Qwen3-8B")
    parser.add_argument(
        "--sentiment-adapter",
        default="./qwen3_8b_sentiment_final",
    )
    parser.add_argument(
        "--risk-adapter",
        default="./qwen3_8b_risk_final",
    )
    parser.add_argument("--symbol", default="STOCK")
    parser.add_argument("--news", default=None)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def render_prompt(tokenizer, task, symbol, news):
    messages = [
        {"role": "system", "content": PROMPTS[task]},
        {
            "role": "user",
            "content": f"Stock symbol: {symbol}\nSummarized news: {news}",
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def digit_token_ids(tokenizer):
    result = {}
    for score in range(1, 6):
        token_ids = tokenizer.encode(str(score), add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Score {score} is not represented by exactly one token")
        result[score] = token_ids[0]
    return result


def predict_score(model, tokenizer, adapter_name, task, symbol, news, max_length):
    model.set_adapter(adapter_name)
    prompt = render_prompt(tokenizer, task, symbol, news)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1]

    token_map = digit_token_ids(tokenizer)
    scores = list(token_map.keys())
    candidate_logits = torch.stack([logits[token_map[score]] for score in scores])
    probabilities = torch.softmax(candidate_logits.float(), dim=0).cpu().tolist()
    best_index = int(torch.argmax(candidate_logits).item())
    return {
        "score": scores[best_index],
        "probabilities": {
            str(score): round(probability, 4)
            for score, probability in zip(scores, probabilities)
        },
    }


def main():
    args = parse_args()
    news = args.news
    if not news:
        news = input("请输入新闻摘要: ").strip()
    if not news:
        raise ValueError("News text cannot be empty")

    print("正在加载 Qwen3-8B 4-bit 基座模型和两个 LoRA 适配器...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quantization_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        args.sentiment_adapter,
        adapter_name="sentiment",
        is_trainable=False,
    )
    model.load_adapter(
        args.risk_adapter,
        adapter_name="risk",
        is_trainable=False,
    )
    model.eval()

    sentiment = predict_score(
        model,
        tokenizer,
        "sentiment",
        "sentiment",
        args.symbol,
        news,
        args.max_length,
    )
    risk = predict_score(
        model,
        tokenizer,
        "risk",
        "risk",
        args.symbol,
        news,
        args.max_length,
    )
    result = {
        "stock_symbol": args.symbol,
        "news": news,
        "sentiment": sentiment,
        "risk": risk,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
