"""
Qwen3.5 IT运维场景微调脚本
适配 AMD Radeon RX 7900 XT (20GB VRAM) + Windows + ROCm
使用 LoRA 进行参数高效微调
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"          # 解决 OpenMP DLL 冲突
os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['NCCL_P2P_DISABLE'] = '1'

import sys
import json
import torch
import logging
from datetime import datetime

# 路径配置
MODEL_PATH = r"<path>\Qwen\Qwen3.5-9B"
DATA_PATH = r"<path>\training_dataset.jsonl"
OUTPUT_DIR = r"<path>\lora"
MERGED_DIR = r"<path>\merged"

# 训练超参数（针对 7900XT 20GB VRAM 优化）
TRAINING_CONFIG = {
    "num_train_epochs": 5,
    "per_device_train_batch_size": 1,        # 7900XT 显存有限，batch=1
    "gradient_accumulation_steps": 8,         # 有效 batch size = 8
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_seq_length": 1024,                   # 控制序列长度节省显存
    "logging_steps": 5,
    "save_steps": 50,
    "save_total_limit": 3,
    "bf16": True,                             # 7900XT 支持 BF16
    "fp16": False,
    "optim": "adamw_torch",
    "lr_scheduler_type": "cosine",
    "gradient_checkpointing": True,           # 显存优化: 梯度检查点
}

# LoRA 配置
LORA_CONFIG = {
    "r": 16,                    # LoRA 秩
    "lora_alpha": 32,           # LoRA alpha
    "lora_dropout": 0.05,
    "target_modules": [         # 目标模块
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# ========== 日志配置 ==========
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(OUTPUT_DIR, 'training.log'),
            encoding='utf-8'
        )
    ]
)
logger = logging.getLogger(__name__)


def check_gpu():
    """检查 AMD GPU 可用性"""
    if not torch.cuda.is_available():
        logger.error("[FAIL] GPU 不可用! 请检查 ROCm 驱动安装。")
        sys.exit(1)
    
    device_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info("[OK] GPU: {}, VRAM: {:.1f} GB".format(device_name, total_mem_gb))
    
    # 检查可用显存
    free_mem = torch.cuda.mem_get_info(0)[0] / (1024**3)
    logger.info("  可用显存: {:.1f} GB".format(free_mem))
    
    return True


def load_dataset(data_path):
    """加载并预处理训练数据"""
    from datasets import Dataset
    
    logger.info("[LOAD] 加载训练数据: {}".format(data_path))
    
    if not os.path.exists(data_path):
        logger.error("[FAIL] 训练数据文件不存在: {}".format(data_path))
        logger.error("  请先运行 generate_dataset.py 生成训练数据")
        sys.exit(1)
    
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    
    logger.info("  加载了 {} 条训练样本".format(len(samples)))
    
    dataset = Dataset.from_list(samples)
    return dataset


def format_chat_template(example, tokenizer):
    """使用 Qwen3.5 的 chat template 格式化数据"""
    messages = example['messages']
    
    # 使用 tokenizer 的 apply_chat_template 方法
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}


def load_model_and_tokenizer(model_path):
    """加载 Qwen3.5 模型和分词器"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    logger.info("[LOAD] 加载模型: {}".format(model_path))
    
    if not os.path.exists(model_path):
        logger.error("[FAIL] 模型目录不存在: {}".format(model_path))
        logger.error("  请先通过 hf-mirror 下载 Qwen3.5 模型")
        sys.exit(1)
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info("  词表大小: {}".format(len(tokenizer)))
    
    # 加载模型（BF16 精度，适合 AMD 7900XT）
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",           # 自动映射到 GPU
        trust_remote_code=True,
        attn_implementation="eager",  # AMD GPU 建议使用 eager attention
    )
    
    model.config.use_cache = False   # 训练时禁用 KV cache
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("  模型参数量: {:.2f}B".format(total_params / 1e9))
    logger.info("  模型精度: {}".format(model.dtype))
    logger.info("  [OK] 模型加载完成")
    
    return model, tokenizer


def apply_lora(model):
    """应用 LoRA 适配器"""
    from peft import LoraConfig, get_peft_model
    
    logger.info("[CONFIG] 配置 LoRA 适配器...")
    
    # 准备模型用于训练
    model.enable_input_require_grads()
    
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
        bias=LORA_CONFIG["bias"],
        task_type=LORA_CONFIG["task_type"],
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / all_params
    
    logger.info("  可训练参数: {:.2f}M / {:.2f}B ({:.2f}%)".format(
        trainable_params / 1e6, all_params / 1e9, trainable_pct
    ))
    logger.info("  [OK] LoRA 配置完成")
    
    return model


def train(model, tokenizer, dataset):
    """执行微调训练"""
    from trl import SFTConfig, SFTTrainer

    logger.info("[TRAIN] 开始微调训练...")
    logger.info("  训练轮次: {}".format(TRAINING_CONFIG['num_train_epochs']))
    logger.info("  Batch Size: {}".format(TRAINING_CONFIG['per_device_train_batch_size']))
    logger.info("  梯度累积步数: {}".format(TRAINING_CONFIG['gradient_accumulation_steps']))
    logger.info("  学习率: {}".format(TRAINING_CONFIG['learning_rate']))
    logger.info("  最大序列长度: {}".format(TRAINING_CONFIG['max_seq_length']))

    # 格式化数据集
    formatted_dataset = dataset.map(
        lambda example: format_chat_template(example, tokenizer),
        remove_columns=dataset.column_names
    )

    # 计算 warmup 步数（替代已弃用的 warmup_ratio）
    total_samples = len(formatted_dataset)
    effective_batch = (
        TRAINING_CONFIG["per_device_train_batch_size"]
        * TRAINING_CONFIG["gradient_accumulation_steps"]
    )
    steps_per_epoch = max(1, total_samples // effective_batch)
    total_steps = steps_per_epoch * TRAINING_CONFIG["num_train_epochs"]
    warmup_steps = int(total_steps * TRAINING_CONFIG["warmup_ratio"])

    logger.info("  总训练步数: {}".format(total_steps))
    logger.info("  Warmup 步数: {}".format(warmup_steps))

    # 使用 SFTConfig 替代 TrainingArguments
    # max_seq_length, packing 等参数全部放在 SFTConfig 中
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        warmup_steps=warmup_steps,
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        save_total_limit=TRAINING_CONFIG["save_total_limit"],
        bf16=TRAINING_CONFIG["bf16"],
        fp16=TRAINING_CONFIG["fp16"],
        optim=TRAINING_CONFIG["optim"],
        lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
        gradient_checkpointing=TRAINING_CONFIG["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        # 以下参数从 SFTTrainer 构造函数移到了 SFTConfig 中
        max_length=TRAINING_CONFIG["max_seq_length"],
        packing=False,
        dataset_text_field="text",
    )

    # SFTTrainer 构造函数中不再传 max_seq_length 和 packing
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
    )

    # 开始训练
    start_time = datetime.now()
    logger.info("  训练开始时间: {}".format(start_time))

    try:
        train_result = trainer.train()

        end_time = datetime.now()
        duration = end_time - start_time
        logger.info("  训练结束时间: {}".format(end_time))
        logger.info("  训练耗时: {}".format(duration))
        logger.info("  训练损失: {:.4f}".format(train_result.training_loss))

        # 保存 LoRA 适配器
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        logger.info("  [OK] LoRA 适配器已保存到: {}".format(OUTPUT_DIR))

        # 保存训练指标
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    except Exception as e:
        logger.error("[FAIL] 训练出错: {}".format(e))
        raise

    return trainer


def merge_and_save(model_path, lora_path, merged_path):
    """合并 LoRA 权重到基础模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    logger.info("[MERGE] 合并 LoRA 权重到基础模型...")
    
    # 重新加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cpu",  # 在 CPU 上合并以节省显存
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    
    # 加载 LoRA 适配器
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    # 合并权重
    logger.info("  正在合并权重（这可能需要几分钟）...")
    model = model.merge_and_unload()
    
    # 保存合并后的模型
    os.makedirs(merged_path, exist_ok=True)
    model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)
    
    logger.info("  [OK] 合并模型已保存到: {}".format(merged_path))


def main():
    """主流程"""
    logger.info("=" * 60)
    logger.info("[START] Qwen3.5 IT运维微调 - AMD 7900XT")
    logger.info("=" * 60)
    
    # Step 1: 环境检查
    check_gpu()
    
    # Step 2: 加载数据集
    dataset = load_dataset(DATA_PATH)
    
    # Step 3: 加载模型
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    
    # Step 4: 应用 LoRA
    model = apply_lora(model)
    
    # Step 5: 训练
    trainer = train(model, tokenizer, dataset)
    
    # Step 6: 释放 GPU 显存
    del model
    del trainer
    torch.cuda.empty_cache()
    
    # Step 7: 合并 LoRA 到基础模型
    merge_and_save(MODEL_PATH, OUTPUT_DIR, MERGED_DIR)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("[DONE] 微调流程全部完成!")
    logger.info("  LoRA 适配器: {}".format(OUTPUT_DIR))
    logger.info("  合并模型: {}".format(MERGED_DIR))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()