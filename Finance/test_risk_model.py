import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd

def load_trained_risk_model(model_path="/root/code/Finance/qwen_risk_model"):
    """加载训练好的风险评估模型"""
    print("正在加载训练好的风险评估模型...")
    
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

def create_risk_test_prompt(text, stock_symbol="STOCK"):
    """创建风险评估测试提示"""
    system_prompt = "Forget all your previous instructions. You are a financial expert specializing in risk assessment for stock recommendations. Based on a specific stock, provide a risk score from 1 to 5, where: 1 indicates very low risk, 2 indicates low risk, 3 indicates moderate risk (default if the news lacks any clear indication of risk), 4 indicates high risk, and 5 indicates very high risk. 1 summarized news will be passed in each time. Provide the score in the format shown below in the response from the assistant."
    
    user_content = f"News to Stock Symbol -- {stock_symbol}: {text}"
    
    conversation = f"""System: {system_prompt}

User: News to Stock Symbol -- AAPL: Apple (AAPL) increases 22%
Assistant: 3

User: News to Stock Symbol -- AAPL: Apple (AAPL) price decreased 30%
Assistant: 4

User: News to Stock Symbol -- AAPL: Apple (AAPL) announced iPhone 15
Assistant: 3

User: {user_content}
Assistant:"""
    
    return conversation

def predict_risk(model, tokenizer, text, stock_symbol="STOCK"):
    """预测风险分数"""
    prompt = create_risk_test_prompt(text, stock_symbol)
    
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
    
    # 提取预测的风险分数
    assistant_response = generated_text.split("Assistant:")[-1].strip()
    
    # 尝试提取数字
    try:
        risk_score = int(assistant_response.split()[0])
        if 1 <= risk_score <= 5:
            return risk_score
    except:
        pass
    
    return None

def test_risk_model():
    """测试风险评估模型"""
    # 加载模型
    model, tokenizer = load_trained_risk_model()
    
    # 测试数据
    test_cases = [
        ("Apple reported strong quarterly earnings with revenue growth of 15%", "AAPL"),
        ("Apple faces major supply chain disruptions and production delays", "AAPL"),
        ("Apple announces bankruptcy filing and CEO resignation", "AAPL"),
        ("Apple stock price remains stable amid market volatility", "AAPL"),
        ("Apple receives regulatory approval for new product launch", "AAPL"),
        ("Tesla recalls 500,000 vehicles due to safety concerns", "TSLA"),
        ("Microsoft announces layoffs affecting 10,000 employees", "MSFT")
    ]
    
    print("\n=== 风险评估模型测试结果 ===")
    for i, (text, symbol) in enumerate(test_cases, 1):
        print(f"\n测试 {i}:")
        print(f"新闻: {text}")
        print(f"股票: {symbol}")
        
        predicted_risk = predict_risk(model, tokenizer, text, symbol)
        
        if predicted_risk:
            risk_map = {1: "极低风险", 2: "低风险", 3: "中等风险", 4: "高风险", 5: "极高风险"}
            print(f"预测风险: {predicted_risk} ({risk_map[predicted_risk]})")
        else:
            print("预测风险: 解析失败")
    
    # 使用真实数据测试
    print("\n=== 真实数据测试 ===")
    try:
        df = pd.read_csv("risk_nasdaq/2.csv", nrows=5)
        df = df[df['Lsa_summary'].notna() & df['risk_deepseek'].notna()]
        
        for i, (_, row) in enumerate(df.head(3).iterrows(), 1):
            text = row['Lsa_summary']
            true_risk = int(row['risk_deepseek'])
            stock_symbol = row.get('Stock_symbol', 'STOCK')
            
            predicted_risk = predict_risk(model, tokenizer, text, stock_symbol)
            
            print(f"\n真实测试 {i}:")
            print(f"股票: {stock_symbol}")
            print(f"新闻摘要: {text[:100]}...")
            print(f"真实风险: {true_risk}")
            print(f"预测风险: {predicted_risk}")
            print(f"准确性: {'✓' if predicted_risk == true_risk else '✗'}")
            
    except Exception as e:
        print(f"真实数据测试失败: {e}")

if __name__ == "__main__":
    test_risk_model() 