"""
IT运维训练集生成脚本
生成适用于 Qwen3.5 微调的 JSONL 格式训练数据
"""

import csv
import json
import os

CSV_PATH = r"<path>\training_dataset.csv"
OUTPUT_PATH = r"<path>\training_dataset.jsonl"

def csv_to_training_data(csv_path, output_path):
    """将 CSV 训练集转换为 Qwen3.5 ChatML 格式的 JSONL"""
    
    system_prompt = (
        "你是一位资深的IT运维专家，拥有超过10年的企业级系统运维经验。"
        "你擅长快速诊断和解决各类IT基础设施问题，包括网络故障、服务器性能优化、"
        "数据库问题、安全事件响应和系统备份恢复。"
        "请根据问题描述，先进行分析思考，然后给出详细的解决方案。"
    )
    
    samples = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row['question'].strip()
            thinking = row['thinking'].strip()
            solution = row['solution'].strip()
            
            # 构造 Qwen3.5 ChatML 格式
            # Qwen3.5 使用 <think>...</think> 标记思考内容
            assistant_response = "<think>\n{}\n</think>\n\n{}".format(thinking, solution)
            
            sample = {
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": "请帮我分析并解决以下IT运维问题：\n\n{}".format(question)
                    },
                    {
                        "role": "assistant",
                        "content": assistant_response
                    }
                ]
            }
            samples.append(sample)
    
    # 写入 JSONL 文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print("[OK] 成功生成 {} 条训练数据".format(len(samples)))
    print("[INFO] 输出文件: {}".format(output_path))
    
    # 显示样本示例
    print("\n[SAMPLE] 示例数据:")
    print(json.dumps(samples[0], ensure_ascii=False, indent=2)[:500] + "...")
    
    return samples

if __name__ == "__main__":
    csv_to_training_data(CSV_PATH, OUTPUT_PATH)