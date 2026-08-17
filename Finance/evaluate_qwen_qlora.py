import argparse
import re

import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Qwen3 QLoRA adapter")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--model-path", default="./Qwen3-8B")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument(
        "--sampling",
        choices=["head", "random", "balanced"],
        default="head",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_rows(df, label_column, max_samples, sampling, seed):
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


def load_eval_rows(args, task):
    csv_path = args.csv_path or task["default_csv"]
    label_column = task["label_column"]
    nrows = (
        args.max_samples
        if args.sampling == "head" and args.max_samples > 0
        else None
    )
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
    df = select_rows(
        df,
        label_column,
        args.max_samples,
        args.sampling,
        args.seed,
    )

    counts = df[label_column].value_counts()
    eval_count = max(1, int(round(len(df) * args.eval_ratio)))
    can_stratify = (
        counts.min() >= 2
        and eval_count >= len(counts)
        and len(df) - eval_count >= len(counts)
    )
    _, eval_df = train_test_split(
        df,
        test_size=args.eval_ratio,
        random_state=args.seed,
        stratify=df[label_column] if can_stratify else None,
    )
    return eval_df.reset_index(drop=True)


def render_prompt(tokenizer, system_prompt, symbol, summary):
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


def extract_score(text):
    match = re.search(r"(?<!\d)([1-5])(?!\d)", text)
    return int(match.group(1)) if match else None


def main():
    args = parse_args()
    task = TASKS[args.task]
    eval_df = load_eval_rows(args, task)
    print(f"Evaluation samples: {len(eval_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    truths = []
    predictions = []
    invalid = 0
    for index, row in eval_df.iterrows():
        prompt = render_prompt(
            tokenizer,
            task["system_prompt"],
            row["Stock_symbol"],
            row["Lsa_summary"],
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=508)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()
        prediction = extract_score(generated)
        truth = int(row[task["label_column"]])
        truths.append(truth)
        predictions.append(prediction if prediction is not None else -1)
        invalid += prediction is None
        print(
            f"[{index + 1:02d}/{len(eval_df):02d}] "
            f"true={truth} pred={prediction} raw={generated!r}"
        )

    print(f"\nInvalid outputs: {invalid}/{len(eval_df)}")
    print(f"Accuracy: {accuracy_score(truths, predictions):.4f}")
    print("Confusion matrix (labels 1..5):")
    print(confusion_matrix(truths, predictions, labels=[1, 2, 3, 4, 5]))
    print("Classification report:")
    print(
        classification_report(
            truths,
            predictions,
            labels=[1, 2, 3, 4, 5],
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
