from lerobot.datasets.lerobot_dataset import LeRobotDataset
from analysis import analyze_metadata
from dataset import AlohaRewardDataset, make_vl_collate_fn
from model import Qwen3VLRewardClassifier
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os


def train_qwen3vl_reward_classifier(ds: LeRobotDataset):
    if ds is None:
        print("Dataset is None, skip training.")
        return
    hf_ds = getattr(ds, "hf_dataset", None)
    if hf_ds is None:
        hf_ds = getattr(ds, "dataset", None)
    first_row = hf_ds[0]
    reward_bins = first_row["reward_bins"]
    if hasattr(reward_bins, "shape"):
        num_bins = reward_bins.shape[0]
    else:
        num_bins = len(reward_bins)
    print("num reward bins:", num_bins)
    max_samples_str = os.environ.get("MAX_TRAIN_SAMPLES", "")
    if max_samples_str and max_samples_str != "-1":
        max_samples = int(max_samples_str)
    else:
        max_samples = None
    print(f"Training with max_samples={max_samples} (None means all)")
    dataset = AlohaRewardDataset(ds, max_samples=max_samples)
    model_name = os.environ.get("QWEN3VL_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
    processor = AutoProcessor.from_pretrained(model_name)
    collate_fn = make_vl_collate_fn(processor)
    batch_size = int(os.environ.get("BATCH_SIZE", "2"))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model = Qwen3VLRewardClassifier(model_name, num_bins)
    device = next(model.parameters()).device
    lr = float(os.environ.get("LR", "1e-4"))
    epochs = int(os.environ.get("EPOCHS", "10"))
    max_steps = int(os.environ.get("MAX_STEPS", "50"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    step = 0
    model.train()
    for epoch in range(epochs):
        for enc, labels in dataloader:
            for k in enc:
                enc[k] = enc[k].to(device)
            labels = labels.to(device)
            logits = model(enc)
            loss = nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % 5 == 0:
                preds = logits.argmax(dim=-1)
                acc = (preds == labels).float().mean().item()
                print("step", step, "loss", float(loss.item()), "acc", float(acc))
            if step >= max_steps:
                break
        if step >= max_steps:
            break
    print("training finished, total steps:", step)
    save_dir = os.environ.get("CKPT_DIR", "checkpoints/qwen3_reward")
    os.makedirs(save_dir, exist_ok=True)
    model.base_model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    torch.save(model.classifier.state_dict(), os.path.join(save_dir, "classifier.pt"))
    print("saved checkpoint to", save_dir)


def chat_qwen3vl():
    model_name = os.environ.get("QWEN3VL_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
    ckpt_dir = os.environ.get("CKPT_DIR", "checkpoints/qwen3_reward")
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, ckpt_dir)
    processor = AutoProcessor.from_pretrained(ckpt_dir)
    model.eval()
    while True:
        try:
            text = input("User: ").strip()
        except EOFError:
            break
        if not text:
            continue
        if text.lower() in ["exit", "quit"]:
            break
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=128)
        gen_ids = []
        for in_ids, out_ids in zip(inputs["input_ids"], outputs):
            gen_ids.append(out_ids[len(in_ids) :])
        texts = processor.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(texts) > 0:
            print("Assistant:", texts[0])


def main():
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "error")
    rl_dataset = "/data/model_checkpoint/clear_brain/notebooks_check/aloha/rl_data"
    rl_dataset_id = "lerobot/realworld_aloha_success_bhc"
    try:
        success_bhc_dataset = LeRobotDataset(repo_id=rl_dataset_id, root=rl_dataset)
        print("Loaded LeRobot dataset:", rl_dataset_id)
    except Exception as e:
        print("Warning: failed to load LeRobot dataset:", rl_dataset_id)
        print("  Details:", e)
        success_bhc_dataset = None
    mode = os.environ.get("MODE", "train")
    if mode == "analyze":
        analyze_metadata(success_bhc_dataset)
    elif mode == "chat":
        chat_qwen3vl()
    else:
        train_qwen3vl_reward_classifier(success_bhc_dataset)


if __name__ == "__main__":
    main()


