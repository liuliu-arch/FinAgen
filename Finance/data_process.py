import json
import hashlib
import unicodedata
import re
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import jieba
from tqdm import tqdm
import binascii

class NewsDeduplicator:
    def __init__(self):
        self.title_threshold = 0.8
        self.content_threshold = 0.75  # 正文重合度阈值调整为0.75
        self.simhash_threshold = 3
        self.minhash_permutations = 128
        self.processed_data = []
        
    def unicode_normalize(self, text: str) -> str:
        """Unicode归一化处理"""
        if not text:
            return ""
        # Unicode标准化
        text = unicodedata.normalize('NFKC', text)
        # 去除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        # 去除特殊字符
        text = re.sub(r'[^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a\s\.\!\?\,\;\:]', '', text)
        return text
    
    def edit_distance(self, s1: str, s2: str) -> float:
        """计算编辑距离并归一化"""
        if not s1 or not s2:
            return 0.0
        
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        
        max_len = max(m, n)
        return 1 - (dp[m][n] / max_len) if max_len > 0 else 0.0
    
    def text_to_tfidf_vector(self, texts: List[str]) -> np.ndarray:
        """将文本转换为TF-IDF向量"""
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        try:
            vectors = vectorizer.fit_transform(texts)
            return vectors.toarray()
        except:
            return np.zeros((len(texts), 1000))
    
    def title_similarity(self, title1: str, title2: str) -> float:
        """计算标题相似度：编辑距离 + 余弦相似度"""
        title1 = self.unicode_normalize(title1)
        title2 = self.unicode_normalize(title2)
        
        # 编辑距离相似度
        edit_sim = self.edit_distance(title1, title2)
        
        # 余弦相似度
        if title1 and title2:
            vectors = self.text_to_tfidf_vector([title1, title2])
            if vectors.shape[0] == 2:
                cos_sim = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
            else:
                cos_sim = 0.0
        else:
            cos_sim = 0.0
        
        # 组合相似度：取平均值
        return (edit_sim + cos_sim) / 2
    
    def get_shingles(self, text: str, k: int = 3) -> Set[str]:
        """生成k-shingles"""
        text = self.unicode_normalize(text)
        words = list(jieba.cut(text))
        if len(words) < k:
            return {text}
        return {' '.join(words[i:i+k]) for i in range(len(words) - k + 1)}
    
    def minhash_signature(self, shingles: Set[str]) -> List[int]:
        """计算MinHash签名"""
        if not shingles:
            return [0] * self.minhash_permutations
        
        signature = [float('inf')] * self.minhash_permutations
        
        for shingle in shingles:
            shingle_bytes = shingle.encode('utf-8')
            for i in range(self.minhash_permutations):
                # 使用不同的盐值来模拟不同的哈希函数
                hash_input = shingle_bytes + str(i).encode('utf-8')
                hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
                signature[i] = min(signature[i], hash_value)
        
        return signature
    
    def jaccard_similarity_minhash(self, sig1: List[int], sig2: List[int]) -> float:
        """使用MinHash估计Jaccard相似度"""
        if len(sig1) != len(sig2):
            return 0.0
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)
    
    def content_overlap(self, content1: str, content2: str) -> float:
        """计算正文重合度使用MinHash"""
        shingles1 = self.get_shingles(content1)
        shingles2 = self.get_shingles(content2)
        
        sig1 = self.minhash_signature(shingles1)
        sig2 = self.minhash_signature(shingles2)
        
        return self.jaccard_similarity_minhash(sig1, sig2)
    
    def hash_string(self, text: str) -> int:
        """将字符串转换为整数哈希值"""
        return int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
    
    def simhash(self, text: str) -> int:
        """计算SimHash值"""
        text = self.unicode_normalize(text)
        words = list(jieba.cut(text))
        
        if not words:
            return 0
        
        # 创建64位特征向量
        features = [0] * 64
        
        for word in words:
            word_hash = self.hash_string(word)
            for i in range(64):
                bit = (word_hash >> i) & 1
                if bit:
                    features[i] += 1
                else:
                    features[i] -= 1
        
        # 生成最终的SimHash值
        simhash_value = 0
        for i in range(64):
            if features[i] > 0:
                simhash_value |= (1 << i)
        
        return simhash_value
    
    def hamming_distance(self, hash1: int, hash2: int) -> int:
        """计算汉明距离"""
        return bin(hash1 ^ hash2).count('1')
    
    def semantic_similarity(self, content1: str, content2: str) -> int:
        """计算语义相似度（返回汉明距离）"""
        hash1 = self.simhash(content1)
        hash2 = self.simhash(content2)
        return self.hamming_distance(hash1, hash2)
    
    def is_duplicate(self, item1: Dict, item2: Dict) -> bool:
        """判断两个新闻是否重复"""
        # 提取标题和内容
        title1 = item1.get('title', '') or item1.get('doc', '')[:100]
        title2 = item2.get('title', '') or item2.get('doc', '')[:100]
        content1 = item1.get('doc', '')
        content2 = item2.get('doc', '')
        
        # 计算三种相似度
        title_sim = self.title_similarity(title1, title2)
        content_sim = self.content_overlap(content1, content2)
        semantic_dist = self.semantic_similarity(content1, content2)
        
        # 应用三重阈值
        return (title_sim > self.title_threshold and 
                content_sim > self.content_threshold and 
                semantic_dist <= self.simhash_threshold)
    
    def load_and_preprocess_data(self, csv_file_path: str = "/mnt/data/Finance/risk_nasdaq/2.csv"):
        """从本地CSV文件加载并预处理数据"""
        print("正在加载CSV文件...")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file_path)
            print(f"CSV文件加载成功，共 {len(df)} 行数据")
            
            # 转换为我们需要的格式
            processed_items = []
            for index, row in df.iterrows():
                # 提取文章内容，优先使用Article列，如果没有则使用Textrank_summary列
                article_content = str(row.get('Article', '')) or str(row.get('Textrank_summary', ''))
                
                # 提取标题
                title = str(row.get('Article_title', '')) or str(row.get('Stock_symbol', ''))
                
                # 提取标签（风险评分）
                risk_score = row.get('risk_deepseek', '')
                labels = f"risk_score:{risk_score}" if pd.notna(risk_score) else ""
                
                # 提取股票代码
                stock_symbol = str(row.get('Stock_symbol', '')) if pd.notna(row.get('Stock_symbol')) else ""
                
                # 提取日期
                date = str(row.get('Date', '')) if pd.notna(row.get('Date')) else ""
                
                processed_item = {
                    'source': 'local_csv',
                    'doc': self.unicode_normalize(article_content),
                    'labels': labels,
                    'title': self.unicode_normalize(title),
                    'stock_symbol': stock_symbol,
                    'date': date,
                    'original_index': index
                }
                
                if processed_item['doc']:  # 只保留有内容的条目
                    processed_items.append(processed_item)
            
            print(f"预处理完成，有效数据量: {len(processed_items)}")
            return processed_items
            
        except Exception as e:
            print(f"加载CSV文件时出现错误: {e}")
            print("请确保CSV文件路径正确且文件可读")
            return []
    
    def deduplicate(self, data: List[Dict]) -> List[Dict]:
        """执行去重操作"""
        print("开始去重处理...")
        unique_items = []
        duplicate_count = 0
        
        for i, current_item in enumerate(tqdm(data, desc="处理进度")):
            is_dup = False
            
            # 与已保留的唯一项目进行比较
            for unique_item in unique_items:
                if self.is_duplicate(current_item, unique_item):
                    is_dup = True
                    duplicate_count += 1
                    break
            
            if not is_dup:
                unique_items.append(current_item)
        
        print(f"去重完成:")
        print(f"  原始数据: {len(data)} 条")
        print(f"  重复数据: {duplicate_count} 条")
        print(f"  保留数据: {len(unique_items)} 条")
        
        return unique_items
    
    def save_to_jsonl(self, data: List[Dict], output_file: str):
        """保存数据到JSONL文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"数据已保存到: {output_file}")
    
    def process_dataset(self, csv_file_path: str = "/mnt/data/Finance/risk_nasdaq/2.csv", 
                       output_file: str = "deduplicated_news.jsonl"):
        """完整的数据处理流程"""
        # 1. 加载和预处理数据
        data = self.load_and_preprocess_data(csv_file_path)
        
        if not data:
            print("没有加载到有效数据，程序退出")
            return []
        
        # 2. 执行去重
        unique_data = self.deduplicate(data)
        
        # 3. 保存结果
        self.save_to_jsonl(unique_data, output_file)
        
        return unique_data

# 使用示例
if __name__ == "__main__":
    # 创建去重器实例
    deduplicator = NewsDeduplicator()
    
    # 处理数据集
    try:
        unique_news = deduplicator.process_dataset(
            csv_file_path="/mnt/data/Finance/risk_nasdaq/risk_deepseek_cleaned_nasdaq_news_full.csv", #  替换为你的路径
            output_file="deduplicated_risk_nasdaq.jsonl"
        )
        
        print(f"\n处理完成！最终保留 {len(unique_news)} 条新闻")
        
        # 显示示例数据
        if unique_news:
            print("\n示例数据:")
            sample = unique_news[0]
            print(f"来源: {sample['source']}")
            print(f"股票代码: {sample.get('stock_symbol', '无')}")
            print(f"日期: {sample.get('date', '无')}")
            print(f"标题: {sample.get('title', '无标题')[:100]}...")
            print(f"内容: {sample['doc'][:200]}...")
            print(f"标签: {sample['labels']}")
            
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        print("请确保已安装所需依赖:")
        print("pip install pandas scikit-learn jieba tqdm numpy")