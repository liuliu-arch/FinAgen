# 安装依赖：transformers, torch 和 huggingface_hub
# pip install transformers torch huggingface_hub tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import torch
import os
from tqdm import tqdm
import time

def download_qwen3():
    model_id = "Qwen/Qwen3-8B"
    local_dir = "./Qwen"  # 本地保存目录

    print(f"开始使用 snap_download 下载模型：{model_id}")
    print(f"下载目标路径：{os.path.abspath(local_dir)}")
    print("=" * 50)
    
    start_time = time.time()
    
    # 使用 snapshot_download 下载模型文件，显示进度
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        resume_download=True,  # 支持断点续传
        # allow_patterns=["*.bin", "*.safetensors", "*.json", "*.txt", "*.model"],  # 下载模型文件
        # ignore_patterns=["*.md", "*.h5", "*.msgpack"],  # 忽略不需要的文件
        tqdm_class=tqdm,  # 使用tqdm显示进度条
        local_files_only=False,  # 允许从网络下载
    )
    
    end_time = time.time()
    download_time = end_time - start_time
    
    print("=" * 50)
    print(f"模型文件下载完成！总耗时：{download_time:.2f} 秒")
    
    # 验证下载的文件
    if os.path.exists(local_dir):
        print(f"模型已保存到：{os.path.abspath(local_dir)}")
        
        # 计算总文件大小
        total_size = 0
        files = []
        for root, dirs, filenames in os.walk(local_dir):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                file_size = os.path.getsize(file_path)
                total_size += file_size
                files.append((filename, file_size))
        
        print(f"下载的文件数量：{len(files)}")
        print(f"总文件大小：{total_size / (1024 * 1024):.2f} MB")
        print("\n主要文件：")
        
        # 按文件大小排序，显示最大的文件
        files.sort(key=lambda x: x[1], reverse=True)
        for filename, file_size in files[:10]:
            size_mb = file_size / (1024 * 1024)
            print(f"  - {filename} ({size_mb:.2f} MB)")
            
    else:
        print("下载失败，目录不存在")

if __name__ == "__main__":
    download_qwen3()
