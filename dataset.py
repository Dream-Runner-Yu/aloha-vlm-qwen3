from lerobot.datasets.lerobot_dataset import LeRobotDataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np


class AlohaRewardDataset(Dataset):
    def __init__(self, lerobot_ds: LeRobotDataset, max_samples: int | None = None):
        self.lerobot_ds = lerobot_ds
        self.meta = lerobot_ds.meta
        self.info = self.meta.info
        self.cameras = [
            "observation.images.cam_right_wrist",
            "observation.images.cam_left_wrist",
            "observation.images.cam_high",
        ]
        self.indices = list(range(len(self.lerobot_ds)))
        if max_samples is not None:
            self.indices = self.indices[: max_samples]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        row = self.lerobot_ds[real_idx]
        images = []
        for cam in self.cameras:
            img = row[cam]
            if isinstance(img, torch.Tensor):
                if img.ndim == 4:
                    img = img[0]
                if img.ndim == 3:
                    img = img.detach().cpu()
                    # LeRobot returns [0,1] float, permute to HWC, scale to [0,255] then to uint8
                    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img = Image.fromarray(img)
            images.append(img)
        task_text = str(row.get("task", "task"))
        text = "Task: " + task_text
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
    def collate(batch):
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        texts = []
        image_inputs = []
        for sample in batch:
            text = sample["text"]
            images = sample["images"]
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
            chat_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            img_inputs, _ = process_vision_info(messages)
            texts.append(chat_text)
            image_inputs.append(img_inputs)
        enc = processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        return enc, labels

    return collate
