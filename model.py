from transformers import Qwen3VLForConditionalGeneration

from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn
import os


class Qwen3VLRewardClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, lora_r: int = 16, lora_alpha: int = 32):
        super().__init__()
        self.base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(self.base_model)

        if hasattr(self.base_model.config, "hidden_size"):
            hidden_size = self.base_model.config.hidden_size
        elif hasattr(self.base_model.config, "text_config") and hasattr(self.base_model.config.text_config, "hidden_size"):
            hidden_size = self.base_model.config.text_config.hidden_size
        else:
            hidden_size = 2048 
            print(f"Warning: Could not determine hidden_size from config. Using default {hidden_size}.")

        mode = os.environ.get("FINETUNE_MODE", "lora").lower()
        use_lora = mode != "full"

        if use_lora:
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
        device = next(self.base_model.parameters()).device
        self.classifier = nn.Linear(hidden_size, num_classes, dtype=torch.bfloat16).to(device)

    def forward(self, encodings):
        outputs = self.base_model(**encodings, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        attn = encodings["attention_mask"]
        idx = attn.long().sum(dim=1) - 1
        idx = idx.clamp(min=0)
        b = hidden.size(0)
        device = hidden.device
        pooled = hidden[torch.arange(b, device=device), idx]
        logits = self.classifier(pooled)
        return logits


if __name__ == "__main__":
    import os

    model_name = os.environ.get("QWEN3VL_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
    num_classes = int(os.environ.get("NUM_CLASSES", "5"))
    print(f"Loading Qwen3VLRewardClassifier(model_name={model_name}, num_classes={num_classes})")
    model = Qwen3VLRewardClassifier(model_name=model_name, num_classes=num_classes)
    print(model)

