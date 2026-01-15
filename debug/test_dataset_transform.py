import torch
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image


def test_transform():
    rl_dataset = "/data/model_checkpoint/clear_brain/notebooks_check/aloha/rl_data"
    rl_dataset_id = "lerobot/realworld_aloha_success_bhc"
    try:
        ds = LeRobotDataset(repo_id=rl_dataset_id, root=rl_dataset)
        print("Dataset loaded.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    item = ds[0]
    cam = "observation.images.cam_high"

    if cam not in item:
        print(f"Camera {cam} not found in item keys: {item.keys()}")
        return

    img_tensor = item[cam]

    print("\n" + "=" * 50)
    print("BEFORE TRANSFORMATION (Raw Tensor)")
    print("=" * 50)
    print(f"Type: {type(img_tensor)}")
    print(f"Shape: {img_tensor.shape}")
    print(f"Dtype: {img_tensor.dtype}")
    if img_tensor.ndim == 3:
        pixel_vals = img_tensor[:, 0, 0]
        print(f"Top-left pixel (CHW): R={pixel_vals[0]:.4f}, G={pixel_vals[1]:.4f}, B={pixel_vals[2]:.4f}")
        print(f"Value range: Min={img_tensor.min():.4f}, Max={img_tensor.max():.4f}")

    print("\n" + "=" * 50)
    print("PERFORMING TRANSFORMATION...")
    print("=" * 50)

    img = img_tensor
    if isinstance(img, torch.Tensor):
        if img.ndim == 4:
            img = img[0]
        if img.ndim == 3:
            img = img.detach().cpu()
            permuted = img.permute(1, 2, 0).numpy()
            scaled = permuted * 255
            img_np = scaled.astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            print(f"Step 1: Permute (CHW -> HWC) shape: {permuted.shape}")
            print(f"Step 2: Scale (* 255) top-left pixel: {scaled[0, 0]}")
            print(f"Step 3: Cast (uint8) top-left pixel: {img_np[0, 0]}")

    print("\n" + "=" * 50)
    print("AFTER TRANSFORMATION (PIL Image -> Numpy)")
    print("=" * 50)

    final_np = np.array(img_pil)
    print(f"Type: {type(img_pil)}")
    print(f"Numpy Shape: {final_np.shape}")
    print(f"Dtype: {final_np.dtype}")
    print(f"Top-left pixel (HWC): R={final_np[0, 0, 0]}, G={final_np[0, 0, 1]}, B={final_np[0, 0, 2]}")
    print(f"Value range: Min={final_np.min()}, Max={final_np.max()}")

    print("\n" + "=" * 50)
    print("VALIDATION")
    print("=" * 50)
    expected_r = int(pixel_vals[0] * 255)
    actual_r = final_np[0, 0, 0]
    print(f"Expected R (approx): {expected_r}")
    print(f"Actual R: {actual_r}")

    if abs(expected_r - actual_r) <= 1:
        print(">> SUCCESS: Values match expectation (within rounding error).")
    else:
        print(">> WARNING: Values do not match!")


if __name__ == "__main__":
    test_transform()

