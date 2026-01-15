from typing import Optional, Dict, List, Any, Union
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np


class AlohaRewardDataset(Dataset):
    """数据集类，用于加载 Aloha 机器人的多模态 reward 分类数据"""
    
    def __init__(self, lerobot_ds: LeRobotDataset, max_samples: Optional[int] = None):
        """
        初始化数据集
        
        Args:
            lerobot_ds: LeRobot 数据集对象
            max_samples: 最大样本数，None 表示使用全部数据
        """
        self.lerobot_ds = lerobot_ds
        self.meta = lerobot_ds.meta
        self.info = self.meta.info
        self.cameras = [
            "observation.images.cam_right_wrist",
            "observation.images.cam_left_wrist",
            "observation.images.cam_high",
        ]
        
        total_len = len(self.lerobot_ds)
        self.indices = list(range(total_len))
        if max_samples is not None:
            self.indices = self.indices[:max_samples]
        
        print(f"AlohaRewardDataset initialized with {len(self.indices)} samples")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含 images, text, label 的字典
        """
        real_idx = self.indices[idx]
        row = self.lerobot_ds[real_idx]
        
        # 处理图像
        images = []
        for cam in self.cameras:
            img = row[cam]
            
            # 处理 torch.Tensor 图像
            if isinstance(img, torch.Tensor):
                if img.ndim == 4:
                    img = img[0]  # 取第一帧 [T, C, H, W] -> [C, H, W]
                if img.ndim == 3:
                    img = img.detach().cpu()
                    # LeRobot returns [0,1] float, permute to HWC, scale to [0,255] then to uint8
                    if img.shape[0] == 3:  # [C, H, W]
                        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    else:  # 已经是 [H, W, C] 格式
                        img = (img.numpy() * 255).astype(np.uint8)
                    img = Image.fromarray(img)
                elif img.ndim == 2:
                    # 灰度图
                    img = (img.detach().cpu().numpy() * 255).astype(np.uint8)
                    img = Image.fromarray(img, mode='L')
            
            images.append(img)
        
        # 处理文本
        task_text = str(row.get("task", "task"))
        text = "Task: " + task_text
        
        # 处理标签
        reward_bins = row["reward_bins"]
        if hasattr(reward_bins, "detach"):
            rb = reward_bins.detach()
        else:
            rb = torch.tensor(reward_bins)
        
        if rb.ndim == 0:
            label = int(rb.item())
        else:
            if rb.sum().abs().item() == 0:
                label = 0
            else:
                label = int(rb.argmax().item())
        
        return {"images": images, "text": text, "label": label}


def make_vl_collate_fn(processor: AutoProcessor):
    """
    创建用于多模态数据批处理的 collate 函数
    
    Args:
        processor: Qwen3-VL 的 AutoProcessor
        
    Returns:
        collate 函数
    """
    def collate(batch: List[Dict[str, Any]]) -> tuple:
        """
        批处理函数
        
        Args:
            batch: 样本列表，每个样本包含 images, text, label
            
        Returns:
            (encodings, labels) 元组
        """
        # 提取标签
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        
        texts = []
        image_inputs = []
        
        for sample in batch:
            text = sample["text"]
            images = sample["images"]
            
            # 构造 Qwen3-VL 标准的 messages 格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text,
                        },
                    ]
                }
            ]
            
            # 将图像插入到消息开头（顺序：cam_right_wrist, cam_left_wrist, cam_high）
            for img in images:
                messages[0]["content"].insert(
                    0,
                    {
                        "type": "image",
                        "image": img,
                        "resized_height": 280,
                        "resized_width": 280,
                    },
                )
            
            # 应用 chat template
            chat_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            img_inputs, _ = process_vision_info(messages)
            
            texts.append(chat_text)
            image_inputs.append(img_inputs)
        
        # 批量处理
        enc = processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        return enc, labels

    return collate
