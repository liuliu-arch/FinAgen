import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import warnings
warnings.filterwarnings("ignore")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理函数
def load_and_preprocess_data(csv_path):
    """加载和预处理数据"""
    print("正在加载数据...")
    df = pd.read_csv(csv_path)[:1000]
    
    # 过滤有效数据
    df = df[df['Lsa_summary'].notna() & df['sentiment_deepseek'].notna()]
    df = df[df['sentiment_deepseek'] != 0]  # 移除无效的情感标签
    
    print(f"有效数据数量: {len(df)}")
    print(f"情感分布: {df['sentiment_deepseek'].value_counts().sort_index()}")
    
    return df

def create_prompt_template(text, sentiment, stock_symbol="STOCK"):
    """创建训练提示模板"""
    # 使用与sentiment_deepseek_deepinfra.py相同的对话格式
    system_prompt = "Forget all your previous instructions. You are a financial expert with stock recommendation experience. Based on a specific stock, score for range from 1 to 5, where 1 is negative, 2 is somewhat negative, 3 is neutral, 4 is somewhat positive, 5 is positive. 1 summarized news will be passed in each time, you will give score in format as shown below in the response from assistant."
    
    # 构建用户输入
    user_content = f"News to Stock Symbol -- {stock_symbol}: {text}"
    
    # 构建完整的对话
    conversation = f"""System: {system_prompt}

User: News to Stock Symbol -- AAPL: Apple (AAPL) increase 22%
Assistant: 5

User: News to Stock Symbol -- AAPL: Apple (AAPL) price decreased 30%
Assistant: 1

User: News to Stock Symbol -- AAPL: Apple (AAPL) announced iPhone 15
Assistant: 4

User: {user_content}
Assistant: {sentiment}"""
    
    return conversation

def prepare_dataset(df, tokenizer, max_length=512):
    """准备训练数据集"""
    print("正在准备数据集...")
    
    texts = []
    labels = []
    
    for _, row in df.iterrows():
        text = row['Lsa_summary']
        sentiment = int(row['sentiment_deepseek'])
        stock_symbol = row.get('Stock_symbol', 'STOCK')  # 获取股票符号，如果没有则使用默认值
        
        if pd.isna(text) or text == '':
            continue
            
        prompt = create_prompt_template(text, sentiment, stock_symbol)
        texts.append(prompt)
        labels.append(sentiment)
    
    # 分割训练集和验证集 (80% 训练, 20% 验证)
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"训练集大小: {len(train_texts)}")
    print(f"验证集大小: {len(eval_texts)}")
    
    # 创建训练数据集
    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'label': train_labels
    })
    
    # 创建验证数据集
    eval_dataset = Dataset.from_dict({
        'text': eval_texts,
        'label': eval_labels
    })
    
    def tokenize_function(examples):
        # 先tokenize获取input_ids
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # 创建labels，初始化为input_ids的副本
        labels = tokenized['input_ids'].clone()
        
        # 只对Assistant的回答部分计算损失，其他部分设置为-100（忽略）
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        # 对每个样本处理
        for i, text in enumerate(examples['text']):
            # 找到最后一个"Assistant: "在文本中的位置
            assistant_marker = "Assistant: "
            last_assistant_pos = text.rfind(assistant_marker)
            
            if last_assistant_pos != -1:
                # 找到Assistant回答开始的位置（跳过"Assistant: "）
                answer_start_pos = last_assistant_pos + len(assistant_marker)
                
                # 分别tokenize输入部分和完整文本，找到对应的token位置
                input_part = text[:answer_start_pos]
                input_part_tokens = tokenizer.encode(input_part, add_special_tokens=False)
                
                # 计算需要mask的token数量（输入部分的token数）
                mask_length = len(input_part_tokens)
                
                input_ids = labels[i]
                actual_length = (input_ids != pad_token_id).sum().item()
                
                # 将输入部分的labels设置为-100（不计算损失）
                if mask_length <= actual_length:
                    labels[i, :mask_length] = -100
                else:
                    # 如果计算出的mask长度过大，使用更保守的方法
                    # 找到最后一个"Assistant" token的位置
                    assistant_text = "Assistant"
                    assistant_tokens = tokenizer.encode(assistant_text, add_special_tokens=False)
                    if len(assistant_tokens) > 0:
                        # 在input_ids中查找最后一个assistant token
                        for j in range(actual_length - 1, -1, -1):
                            if input_ids[j].item() == assistant_tokens[0]:
                                # 找到后，跳过"Assistant: "对应的token（大约+3个token）
                                mask_end = min(j + len(assistant_tokens) + 3, actual_length)
                                labels[i, :mask_end] = -100
                                break
                
                # 确保padding部分的labels也是-100
                if actual_length < len(input_ids):
                    labels[i, actual_length:] = -100
            else:
                # 如果没有找到Assistant标记，mask掉整个序列（不应该发生）
                labels[i, :] = -100
        
        tokenized['labels'] = labels
        return tokenized
    
    # 对训练集和验证集进行tokenization
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    eval_tokenized = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    return train_tokenized, eval_tokenized

def create_model_and_tokenizer():
    """创建模型和分词器"""
    print("正在加载模型和分词器...")
    
    model_name = "/root/code/Finance/Qwen"
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 准备模型进行训练
    model = prepare_model_for_kbit_training(model)
    
    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_model(model, tokenizer, train_dataset, eval_dataset, output_dir="./qwen_sentiment_model"):
    """训练模型"""
    print("开始训练模型...")
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # report_to=None,  # 禁用wandb等报告工具
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"模型已保存到: {output_dir}")

def main():
    """主函数"""
    # 数据路径
    csv_path = "nasdaq_news_sentiment/sentiment_deepseek_new_cleaned_nasdaq_news_full.csv"
    
    # 加载和预处理数据
    df = load_and_preprocess_data(csv_path)
    
    # 创建模型和分词器
    model, tokenizer = create_model_and_tokenizer()
    
    # 准备数据集
    train_dataset, eval_dataset = prepare_dataset(df, tokenizer)
    
    # 训练模型
    train_model(model, tokenizer, train_dataset, eval_dataset)
    
    print("训练完成！")

if __name__ == "__main__":
    main() 