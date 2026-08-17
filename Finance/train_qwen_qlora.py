import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)


TASKS = {
    "sentiment": {
        "label_column": "sentiment_deepseek",
        "default_csv": "nasdaq_news_sentiment/sentiment_deepseek_new_cleaned_nasdaq_news_full.csv",
        "system_prompt": (
            "You are a financial news sentiment analyst. Given one summarized news item "
            "about a stock, return exactly one integer score from 1 to 5: "
            "1=negative, 2=somewhat negative, 3=neutral, "
            "4=somewhat positive, 5=positive. Return only the score."
        ),
    },
    "risk": {
        "label_column": "risk_deepseek",
        "default_csv": "risk_nasdaq/risk_deepseek_cleaned_nasdaq_news_full.csv",
        "system_prompt": (
            "You are a financial risk analyst. Given one summarized news item about a stock, "
            "return exactly one integer risk score from 1 to 5: "
            "1=very low risk, 2=low risk, 3=moderate risk, "
            "4=high risk, 5=very high risk. Return only the score."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-8B 4-bit QLoRA trainer")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--model-path", default="./Qwen3-8B")
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument(
        "--sampling",
        choices=["head", "random", "balanced"],
        default="head",
        help="How to select max-samples rows. Use balanced for formal training.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_rows(
    df: pd.DataFrame,
    label_column: str,
    max_samples: int,
    sampling: str,
    seed: int,
) -> pd.DataFrame:
    if max_samples <= 0 or max_samples >= len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if sampling == "head":
        return df.head(max_samples).reset_index(drop=True)
    if sampling == "random":
        return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    labels = sorted(df[label_column].unique().tolist())
    quota, remainder = divmod(max_samples, len(labels))
    selected_indices = []
    for position, label in enumerate(labels):
        group = df[df[label_column] == label]
        requested = quota + (1 if position < remainder else 0)
        take = min(requested, len(group))
        selected_indices.extend(
            group.sample(n=take, random_state=seed + int(label)).index.tolist()
        )

    shortfall = max_samples - len(selected_indices)
    if shortfall > 0:
        remaining = df.drop(index=selected_indices)
        selected_indices.extend(
            remaining.sample(
                n=min(shortfall, len(remaining)),
                random_state=seed,
            ).index.tolist()
        )
    return (
        df.loc[selected_indices]
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def load_dataframe(
    csv_path: str,
    label_column: str,
    max_samples: int,
    sampling: str,
    seed: int,
) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    # Head sampling is fast for smoke tests; random/balanced sampling must inspect
    # the complete label column to avoid temporal and class-order bias.
    nrows = max_samples if sampling == "head" and max_samples > 0 else None
    df = pd.read_csv(
        csv_path,
        usecols=["Lsa_summary", "Stock_symbol", label_column],
        nrows=nrows,
    )
    df = df.dropna(subset=["Lsa_summary", label_column]).copy()
    df[label_column] = pd.to_numeric(df[label_column], errors="coerce")
    df = df.dropna(subset=[label_column])
    df[label_column] = df[label_column].astype(int)
    df = df[df[label_column].between(1, 5)]
    df["Stock_symbol"] = df["Stock_symbol"].fillna("STOCK").astype(str)
    df["Lsa_summary"] = df["Lsa_summary"].astype(str)
    df = select_rows(df, label_column, max_samples, sampling, seed)

    if len(df) < 20:
        raise ValueError(f"Only {len(df)} valid rows remain; at least 20 are required.")

    print(f"Valid samples: {len(df)}")
    print("Label distribution:")
    print(df[label_column].value_counts().sort_index().to_string())
    return df


def render_prompt(tokenizer: Any, system_prompt: str, symbol: str, summary: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Stock symbol: {symbol}\nSummarized news: {summary}",
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode_example(
    tokenizer: Any,
    system_prompt: str,
    symbol: str,
    summary: str,
    score: int,
    max_length: int,
) -> Dict[str, List[int]]:
    prompt = render_prompt(tokenizer, system_prompt, symbol, summary)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(str(score) + tokenizer.eos_token, add_special_tokens=False)

    if len(answer_ids) >= max_length:
        raise ValueError("max_length is too small to hold the target answer")

    # Always preserve the target tokens so every sample contributes a valid loss.
    prompt_ids = prompt_ids[: max_length - len(answer_ids)]
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def build_datasets(
    df: pd.DataFrame,
    tokenizer: Any,
    label_column: str,
    system_prompt: str,
    max_length: int,
    eval_ratio: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    records = [
        encode_example(
            tokenizer,
            system_prompt,
            row.Stock_symbol,
            row.Lsa_summary,
            int(getattr(row, label_column)),
            max_length,
        )
        for row in df.itertuples(index=False)
    ]

    labels = df[label_column].tolist()
    label_counts = pd.Series(labels).value_counts()
    eval_count = max(1, int(round(len(labels) * eval_ratio)))
    can_stratify = (
        label_counts.min() >= 2
        and eval_count >= len(label_counts)
        and len(labels) - eval_count >= len(label_counts)
    )
    stratify = labels if can_stratify else None
    train_records, eval_records = train_test_split(
        records,
        test_size=eval_ratio,
        random_state=seed,
        stratify=stratify,
    )
    print(f"Train samples: {len(train_records)}; eval samples: {len(eval_records)}")
    return Dataset.from_list(train_records), Dataset.from_list(eval_records)


@dataclass
class CompletionOnlyCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        model_features = [
            {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]}
            for item in features
        ]
        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        sequence_length = batch["input_ids"].shape[1]
        padded_labels = []
        for item in features:
            labels = list(item["labels"])
            padded_labels.append(labels + [-100] * (sequence_length - len(labels)))
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    task = TASKS[args.task]
    csv_path = args.csv_path or task["default_csv"]
    output_dir = args.output_dir or f"qwen3_8b_{args.task}_qlora"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU was not detected")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Task: {args.task}")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {csv_path}")
    print(f"Output: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quantization_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    df = load_dataframe(
        csv_path,
        task["label_column"],
        args.max_samples,
        args.sampling,
        args.seed,
    )
    train_dataset, eval_dataset = build_datasets(
        df,
        tokenizer,
        task["label_column"],
        task["system_prompt"],
        args.max_length,
        args.eval_ratio,
        args.seed,
    )

    use_bf16 = torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CompletionOnlyCollator(tokenizer),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"QLoRA adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
