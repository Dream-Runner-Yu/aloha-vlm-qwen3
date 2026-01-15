from typing import Dict, Any, Optional
from transformers import Qwen3VLForConditionalGeneration

from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn
import os


class Qwen3VLRewardClassifier(nn.Module):
    """基于 Qwen3-VL 的 reward 分类器"""
    
    def __init__(
        self, 
        model_name: str, 
        num_classes: int, 
        lora_r: int = 16, 
        lora_alpha: int = 32
    ):
        """
        初始化分类器
        
        Args:
            model_name: Qwen3-VL 模型名称或路径
            num_classes: 分类类别数
            lora_r: LoRA 的秩（仅在 LoRA 模式下使用）
            lora_alpha: LoRA 的 alpha 参数（仅在 LoRA 模式下使用）
        """
        super().__init__()
        
        print(f"Loading base model: {model_name}")
        self.base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f"Model loaded successfully. Config: {self.base_model.config}")

        # 推断 hidden_size
        hidden_size = self._get_hidden_size()
        print(f"Using hidden_size: {hidden_size}")

        # 配置 LoRA 或全参数微调
        mode = os.environ.get("FINETUNE_MODE", "lora").lower()
        use_lora = mode != "full"
        
        if use_lora:
            print(f"Using LoRA fine-tuning (r={lora_r}, alpha={lora_alpha})")
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.base_model = get_peft_model(self.base_model, peft_config)
            print("LoRA adapter added successfully")
        else:
            print("Using full fine-tuning mode")
        
        # 创建分类头
        device = next(self.base_model.parameters()).device
        self.classifier = nn.Linear(hidden_size, num_classes, dtype=torch.bfloat16).to(device)
        print(f"Classifier head: Linear({hidden_size}, {num_classes})")

    def _get_hidden_size(self) -> int:
        """从模型配置中推断 hidden_size"""
        if hasattr(self.base_model.config, "hidden_size"):
            return self.base_model.config.hidden_size
        elif hasattr(self.base_model.config, "text_config") and hasattr(
            self.base_model.config.text_config, "hidden_size"
        ):
            return self.base_model.config.text_config.hidden_size
        else:
            default_size = 2048
            print(f"Warning: Could not determine hidden_size from config. Using default {default_size}.")
            return default_size

    def forward(self, encodings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            encodings: 包含 input_ids, attention_mask, pixel_values 等的字典
            
        Returns:
            logits: [batch_size, num_classes] 形状的分类 logits
        """
        # 获取模型输出
        outputs = self.base_model(**encodings, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # 从最后一个非 padding token 提取特征
        attn = encodings["attention_mask"]
        idx = attn.long().sum(dim=1) - 1  # 每个样本最后一个 token 的位置
        idx = idx.clamp(min=0)
        
        batch_size = hidden.size(0)
        device = hidden.device
        pooled = hidden[torch.arange(batch_size, device=device), idx]  # [batch_size, hidden_size]
        
        # 分类
        logits = self.classifier(pooled)  # [batch_size, num_classes]
        return logits


if __name__ == "__main__":
    import os

    model_name = os.environ.get("QWEN3VL_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
    num_classes = int(os.environ.get("NUM_CLASSES", "5"))
    print(f"Loading Qwen3VLRewardClassifier(model_name={model_name}, num_classes={num_classes})")
    model = Qwen3VLRewardClassifier(model_name=model_name, num_classes=num_classes)
    print(model)

