import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
import numpy as np
import os
# 以上导入包含：
# - PyTorch 基础与 DataLoader
# - Qwen2-VL 模型与处理器（文本+图像预处理）
# - PEFT/LoRA 微调工具
# - PIL 用于读图（或生成 mock 帧）


def _module_summary(module, depth=0, max_depth=1):
    indent = "  " * depth
    for name, child in module.named_children():
        total = sum(p.numel() for p in child.parameters())
        trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
        print(f"{indent}- {name}: {child.__class__.__name__}, params={total}, trainable={trainable}")
        if depth < max_depth:
            _module_summary(child, depth + 1, max_depth)


class Qwen2VLCritic(nn.Module):
    # 封装：Qwen2-VL 作为特征抽取器，插入 LoRA，并在顶部加线性分类头
    def __init__(self, model_path, num_classes=201):
        super().__init__()
        self.base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        # 加载基座模型：
        # - 使用 bfloat16 降低显存占用
        # - device_map="auto" 自动分配到可用设备（多卡/CPU）
        if os.environ.get("DEBUG_MODEL", "0") == "1":
            print("=== Base Model: Qwen2VLForConditionalGeneration ===")
            print(self.base_model)
            cfg = getattr(self.base_model, "config", None)
            if cfg is not None:
                print("=== Key Config Fields ===")
                key_fields = [
                    "hidden_size",
                    "num_hidden_layers",
                    "num_attention_heads",
                    "vocab_size",
                    "vision_hidden_size",
                    "image_size",
                    "mm_patch_size",
                ]
                for k in key_fields:
                    v = getattr(cfg, k, None)
                    if v is not None:
                        print(f"{k}: {v}")
            print("=== Children of base_model ===")
            _module_summary(self.base_model, max_depth=0)
            if hasattr(self.base_model, "model"):
                print("=== Children of base_model.model ===")
                _module_summary(self.base_model.model, max_depth=1)
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj"],
            task_type=None,
            inference_mode=False
        )
        # LoRA 配置：
        # - 在注意力与 FFN 的若干投影层插入低秩适配参数
        # - 自定义任务（不使用内置任务类型）
        self.base_model = get_peft_model(self.base_model, peft_config)
        self.hidden_size = self.base_model.config.hidden_size
        self.critic_head = nn.Linear(self.hidden_size, num_classes, dtype=torch.bfloat16).to(self.base_model.device)
        # 分类头：
        # - 将 pooled 特征映射到 num_classes 维度的 logits

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        # 前向：
        # - 使用 base_model.model 获取隐藏状态而非生成头
        # - 支持图像+文本联合输入
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True
        )
        # 输出包含隐藏状态序列，取最后一层作为序列表示
        hidden_states = getattr(outputs, "hidden_states", None)
        if isinstance(hidden_states, (tuple, list)) and len(hidden_states) > 0:
            last_hidden_state = hidden_states[-1]
        else:
            last_hidden_state = hidden_states
        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        pooled_features = last_hidden_state[batch_indices, sequence_lengths]
        # 使用每个样本“最后一个非 padding token”的向量作为全局表示
        logits = self.critic_head(pooled_features)
        return logits


class SimpleVLMDataset(Dataset):
    # 简单数据集封装：每个样本为字典（text/label/可选 frames 或 image_path）
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_collate_fn(processor, frames_per_sample=4, image_size=(224, 224), use_mock=True, placeholder_str="<image>"):
    # 批处理函数：
    # - 每个样本可包含多帧图像
    # - use_mock=True 时生成随机帧作为占位数据
    def collate_fn(batch):
        frames_batch = []
        texts = []
        labels_list = []
        for item in batch:
            labels_list.append(item["label"])
            if use_mock:
                frames = []
                for _ in range(frames_per_sample):
                    arr = np.random.randint(0, 256, (image_size[1], image_size[0], 3), dtype=np.uint8)
                    frames.append(Image.fromarray(arr))
            else:
                paths = item.get("frames")
                if paths is None:
                    paths = [item["image_path"]]
                paths = paths[:frames_per_sample]
                frames = [Image.open(p).convert("RGB") for p in paths]
            frames_batch.append(frames)
            placeholder = " ".join([placeholder_str] * len(frames))
            texts.append(f"{item['text']} {placeholder}".strip())
        labels = torch.tensor(labels_list, dtype=torch.long)
        enc = processor(images=frames_batch, text=texts, return_tensors="pt", padding=True)
        # processor 负责将多帧图像与文本打包为模型可接受的张量，并自动生成 image_grid_thw 等结构
        return enc, labels
    return collate_fn


def _detect_image_placeholder(processor):
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        return "<image>"
    specials = []
    m = getattr(tok, "special_tokens_map", None)
    if isinstance(m, dict):
        for k, v in m.items():
            if isinstance(v, list):
                specials.extend(v)
            elif isinstance(v, str):
                specials.append(v)
    extras = getattr(tok, "additional_special_tokens", None)
    if isinstance(extras, list):
        specials.extend(extras)
    candidates = [s for s in specials if isinstance(s, str) and "image" in s.lower()]
    if candidates:
        return candidates[0]
    return "<image>"


def train():
    # 训练入口：
    # - 使用 Qwen2-VL 作为编码器（带 LoRA）
    # - 简单 CrossEntropyLoss + AdamW
    debug = os.environ.get("DEBUG_RUN", "0") == "1"
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    num_classes = 201
    batch_size = 1 if debug else 2
    num_epochs = 1 if debug else 3
    lr = 1e-4

    samples = [
        {"text": "描述1", "label": 0},
        {"text": "描述2", "label": 1},
        {"text": "描述3", "label": 2},
        {"text": "描述4", "label": 3},
    ]
    # 上述样本仅为演示；真实使用时可替换为包含 frames 或 image_path 的结构
    if debug:
        samples = [{"text": "调试样本", "label": 0}]

    processor = Qwen2VLProcessor.from_pretrained(model_name)
    dataset = SimpleVLMDataset(samples)
    placeholder = _detect_image_placeholder(processor)
    frames_per_sample = 1 if debug else 4
    collate_fn = make_collate_fn(
        processor,
        frames_per_sample=frames_per_sample,
        image_size=(224, 224),
        use_mock=True,
        placeholder_str=placeholder,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = Qwen2VLCritic(model_name, num_classes=num_classes)
    model.train()

    device = next(model.parameters()).device
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for step, (enc, labels) in enumerate(dataloader):
            for k in enc:
                enc[k] = enc[k].to(device)
            labels = labels.to(device)

            logits = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                pixel_values=enc["pixel_values"],
                image_grid_thw=enc["image_grid_thw"],
            )

            loss = criterion(logits, labels)

            if debug:
                print(f"[DEBUG] loss {loss.item():.4f}")
                return

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")


if __name__ == "__main__":
    train()
