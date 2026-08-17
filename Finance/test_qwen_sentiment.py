import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd

def load_trained_sentiment_model(model_path="/root/code/Finance/qwen_sentiment_model"):
    """加载训练好的情感分析模型"""
    print("正在加载训练好的情感分析模型...")
    
    # 加载基础模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "/root/code/Finance/Qwen",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    
    model.eval()
    return model, tokenizer

def create_sentiment_test_prompt(text, stock_symbol="STOCK"):
    """创建情感分析测试提示"""
    system_prompt = "Forget all your previous instructions. You are a financial expert with stock recommendation experience. Based on a specific stock, score for range from 1 to 5, where 1 is negative, 2 is somewhat negative, 3 is neutral, 4 is somewhat positive, 5 is positive. 1 summarized news will be passed in each time, you will give score in format as shown below in the response from assistant."
    
    user_content = f"News to Stock Symbol -- {stock_symbol}: {text}"
    
    conversation = f"""System: {system_prompt}

User: News to Stock Symbol -- AAPL: Apple (AAPL) increase 22%
Assistant: 5

User: News to Stock Symbol -- AAPL: Apple (AAPL) price decreased 30%
Assistant: 1

User: News to Stock Symbol -- AAPL: Apple (AAPL) announced iPhone 15
Assistant: 4

User: {user_content}
Assistant:"""
    
    return conversation

def predict_sentiment(model, tokenizer, text, stock_symbol="STOCK"):
    """预测情感分数"""
    prompt = create_sentiment_test_prompt(text, stock_symbol)
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成预测
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取预测的情感分数
    assistant_response = generated_text.split("Assistant:")[-1].strip()
    
    # 尝试提取数字
    try:
        sentiment_score = int(assistant_response.split()[0])
        if 1 <= sentiment_score <= 5:
            return sentiment_score
    except:
        pass
    
    return None

def test_sentiment_model():
    """测试情感分析模型"""
    # 加载模型
    model, tokenizer = load_trained_sentiment_model()
    
    # 测试数据
    test_cases = [
        ("Apple reported strong quarterly earnings with revenue growth of 15%", "AAPL"),
        ("Apple faces supply chain disruptions and production delays", "AAPL"),
        ("Apple announces new iPhone with innovative features", "AAPL"),
        ("Apple stock price remains stable amid market volatility", "AAPL"),
        ("Apple CEO resigns amid scandal and controversy", "AAPL"),
        ("Tesla delivers record number of vehicles in Q4", "TSLA"),
        ("Microsoft announces major layoffs affecting 10,000 employees", "MSFT"),
        ("Google reports disappointing ad revenue decline", "GOOGL"),
        ("Amazon Prime membership reaches new milestone", "AMZN"),
        ("Netflix loses subscribers for the first time", "NFLX")
    ]
    
    print("\n=== 情感分析模型测试结果 ===")
    for i, (text, symbol) in enumerate(test_cases, 1):
        print(f"\n测试 {i}:")
        print(f"新闻: {text}")
        print(f"股票: {symbol}")
        
        predicted_sentiment = predict_sentiment(model, tokenizer, text, symbol)
        
        if predicted_sentiment:
            sentiment_map = {1: "负面", 2: "轻微负面", 3: "中性", 4: "正面", 5: "极正面"}
            print(f"预测情感: {predicted_sentiment} ({sentiment_map[predicted_sentiment]})")
        else:
            print("预测情感: 解析失败")
    
    # 使用真实数据测试
    print("\n=== 真实数据测试 ===")
    try:
        df = pd.read_csv("nasdaq_news_sentiment/1.csv", nrows=10)
        df = df[df['Lsa_summary'].notna() & df['sentiment_deepseek'].notna()]
        
        correct_predictions = 0
        total_predictions = 0
        
        for i, (_, row) in enumerate(df.head(5).iterrows(), 1):
            text = row['Lsa_summary']
            true_sentiment = int(row['sentiment_deepseek'])
            stock_symbol = row.get('Stock_symbol', 'STOCK')
            
            predicted_sentiment = predict_sentiment(model, tokenizer, text, stock_symbol)
            
            print(f"\n真实测试 {i}:")
            print(f"股票: {stock_symbol}")
            print(f"新闻摘要: {text[:100]}...")
            print(f"真实情感: {true_sentiment}")
            print(f"预测情感: {predicted_sentiment}")
            
            if predicted_sentiment is not None:
                total_predictions += 1
                if predicted_sentiment == true_sentiment:
                    correct_predictions += 1
                    print(f"准确性: ✓")
                else:
                    print(f"准确性: ✗")
            else:
                print(f"准确性: 解析失败")
        
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions * 100
            print(f"\n整体准确率: {correct_predictions}/{total_predictions} = {accuracy:.1f}%")
            
    except Exception as e:
        print(f"真实数据测试失败: {e}")

def test_sentiment_distribution():
    """测试模型在不同情感类别上的表现"""
    print("\n=== 情感分布测试 ===")
    
    model, tokenizer = load_trained_sentiment_model()
    
    # 针对不同情感类别的测试用例
    sentiment_test_cases = {
        1: [  # 负面
            "Company files for bankruptcy protection",
            "CEO arrested for fraud charges",
            "Stock crashes 50% in single day"
        ],
        2: [  # 轻微负面
            "Quarterly earnings miss analyst expectations",
            "Company faces regulatory investigation",
            "Product recall affects sales"
        ],
        3: [  # 中性
            "Company maintains steady performance",
            "Stock price remains unchanged",
            "Quarterly report meets expectations"
        ],
        4: [  # 正面
            "Company beats earnings expectations",
            "New product launch receives positive reviews",
            "Stock price increases 10%"
        ],
        5: [  # 极正面
            "Company reports record-breaking profits",
            "Stock soars 30% on breakthrough announcement",
            "Revolutionary product disrupts entire industry"
        ]
    }
    
    for expected_sentiment, test_texts in sentiment_test_cases.items():
        print(f"\n--- 测试情感类别 {expected_sentiment} ---")
        correct = 0
        total = len(test_texts)
        
        for text in test_texts:
            predicted = predict_sentiment(model, tokenizer, text, "TEST")
            match = "✓" if predicted == expected_sentiment else "✗"
            print(f"预期: {expected_sentiment}, 预测: {predicted} {match}")
            if predicted == expected_sentiment:
                correct += 1
        
        accuracy = correct / total * 100
        print(f"类别准确率: {correct}/{total} = {accuracy:.1f}%")

if __name__ == "__main__":
    test_sentiment_model()
    test_sentiment_distribution() 