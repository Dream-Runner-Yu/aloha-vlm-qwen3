from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch


def check_format():
    rl_dataset = "/data/model_checkpoint/clear_brain/notebooks_check/aloha/rl_data"
    rl_dataset_id = "lerobot/realworld_aloha_success_bhc"
    try:
        ds = LeRobotDataset(repo_id=rl_dataset_id, root=rl_dataset)
        print(f"Loaded dataset: {rl_dataset_id}")

        item = ds[0]

        cameras = [
            "observation.images.cam_right_wrist",
            "observation.images.cam_left_wrist",
            "observation.images.cam_high",
        ]

        for cam in cameras:
            if cam in item:
                img = item[cam]
                print(f"Camera: {cam}")
                print(f"Type: {type(img)}")
                if isinstance(img, torch.Tensor):
                    print(f"Shape: {img.shape}")
                    print(f"Dtype: {img.dtype}")
                    print(f"Min: {img.min()}, Max: {img.max()}")
                    shape = img.shape
                    if shape[0] == 3:
                        print("Likely CHW format (Channels first)")
                    elif shape[-1] == 3:
                        print("Likely HWC format (Channels last)")
                    else:
                        print("Ambiguous format")
                else:
                    print("Not a tensor")
                print("-" * 20)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    check_format()

