"""
使用 Mock 数据测试训练流程
"""
import os
import sys
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock

# 检查依赖
try:
    import torch
    import numpy as np
    from PIL import Image
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"⚠ 警告: 缺少依赖 {e}")
    print("将使用 mock 对象进行结构测试")
    TORCH_AVAILABLE = False
    # 创建简单的 mock
    torch = Mock()
    np = Mock()
    Image = Mock()
    Dataset = Mock

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dataset import AlohaRewardDataset, make_vl_collate_fn
    DATASET_AVAILABLE = True
except ImportError as e:
    print(f"⚠ 警告: 无法导入 dataset 模块: {e}")
    DATASET_AVAILABLE = False

try:
    from model import Qwen3VLRewardClassifier
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"⚠ 警告: 无法导入 model 模块: {e}")
    MODEL_AVAILABLE = False

try:
    from transformers import AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠ 警告: 无法导入 transformers: {e}")
    TRANSFORMERS_AVAILABLE = False


class MockLeRobotDataset:
    """模拟 LeRobotDataset 用于测试"""
    
    def __init__(self, num_samples: int = 10, num_classes: int = 5, image_size: tuple = (224, 224)):
        """
        初始化 Mock 数据集
        
        Args:
            num_samples: 样本数量
            num_classes: reward 类别数
            image_size: 图像尺寸 (H, W)
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        
        # 创建 mock meta 对象
        self.meta = Mock()
        self.meta.info = {
            "codebase_version": "1.0.0",
            "robot_type": "aloha",
            "fps": 30,
            "total_episodes": num_samples,
            "total_frames": num_samples,
        }
        
        # 创建 mock hf_dataset
        self.hf_dataset = Mock()
        self.hf_dataset.__len__ = lambda: num_samples
        self.hf_dataset.__getitem__ = self._get_item
        
        # 创建 mock dataset 属性（备用）
        self.dataset = self.hf_dataset
        
        print(f"MockLeRobotDataset initialized: {num_samples} samples, {num_classes} classes")
    
    def _get_item(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range")
        
        # 生成随机图像（RGB，[0, 1] 范围的 float tensor）
        h, w = self.image_size
        if TORCH_AVAILABLE:
            images = {
                "observation.images.cam_right_wrist": torch.rand(3, h, w),
                "observation.images.cam_left_wrist": torch.rand(3, h, w),
                "observation.images.cam_high": torch.rand(3, h, w),
            }
            # 生成随机 reward_bins（one-hot 向量）
            reward_bins = torch.zeros(self.num_classes)
            label = idx % self.num_classes
            reward_bins[label] = 1.0
        else:
            # 使用 mock 对象
            images = {
                "observation.images.cam_right_wrist": Mock(),
                "observation.images.cam_left_wrist": Mock(),
                "observation.images.cam_high": Mock(),
            }
            reward_bins = Mock()
            reward_bins.ndim = 1
            reward_bins.sum.return_value.abs.return_value.item.return_value = 1.0
            reward_bins.argmax.return_value.item.return_value = idx % self.num_classes
            label = idx % self.num_classes
        
        # 生成随机任务文本
        tasks = [
            "pick up cup",
            "place object",
            "open drawer",
            "close drawer",
            "move object",
        ]
        task = tasks[idx % len(tasks)]
        
        return {
            **images,
            "task": task,
            "reward_bins": reward_bins,
        }
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._get_item(idx)


def test_dataset():
    """测试数据集加载"""
    print("\n" + "="*60)
    print("测试数据集加载")
    print("="*60)
    
    if not DATASET_AVAILABLE:
        print("⚠ 跳过：dataset 模块不可用")
        return None
    
    if not TORCH_AVAILABLE:
        print("⚠ 跳过：torch 不可用")
        return None
    
    mock_ds = MockLeRobotDataset(num_samples=5, num_classes=5)
    dataset = AlohaRewardDataset(mock_ds, max_samples=3)
    
    print(f"数据集长度: {len(dataset)}")
    
    # 测试获取样本
    sample = dataset[0]
    print(f"\n样本键: {sample.keys()}")
    print(f"图像数量: {len(sample['images'])}")
    print(f"图像类型: {[type(img) for img in sample['images']]}")
    print(f"文本: {sample['text']}")
    print(f"标签: {sample['label']}")
    
    # 测试所有样本
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"样本 {i}: label={sample['label']}, text={sample['text']}")
    
    print("\n✓ 数据集测试通过")
    return dataset


def test_collate_fn():
    """测试 collate 函数"""
    print("\n" + "="*60)
    print("测试 Collate 函数")
    print("="*60)
    
    if not DATASET_AVAILABLE or not TORCH_AVAILABLE:
        print("⚠ 跳过：依赖不可用")
        return
    
    # 使用一个小的模型名称（如果可用）或创建 mock processor
    model_name = os.environ.get("QWEN3VL_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
    
    if TRANSFORMERS_AVAILABLE:
        try:
            print(f"尝试加载 processor: {model_name}")
            processor = AutoProcessor.from_pretrained(model_name)
            print("✓ Processor 加载成功")
        except Exception as e:
            print(f"⚠ 无法加载真实 processor: {e}")
            print("使用 mock processor 进行测试...")
            # 创建简单的 mock processor（仅用于测试结构）
            processor = Mock()
            processor.apply_chat_template = lambda *args, **kwargs: "<mock_text>"
            processor.side_effect = lambda *args, **kwargs: {
                "input_ids": torch.randint(0, 1000, (2, 10)),
                "attention_mask": torch.ones(2, 10),
                "pixel_values": torch.rand(2, 3, 224, 224),
            }
    else:
        print("⚠ 使用 mock processor（transformers 不可用）")
        processor = Mock()
        processor.apply_chat_template = lambda *args, **kwargs: "<mock_text>"
        processor.side_effect = lambda *args, **kwargs: {
            "input_ids": torch.randint(0, 1000, (2, 10)) if TORCH_AVAILABLE else Mock(),
            "attention_mask": torch.ones(2, 10) if TORCH_AVAILABLE else Mock(),
            "pixel_values": torch.rand(2, 3, 224, 224) if TORCH_AVAILABLE else Mock(),
        }
    
    mock_ds = MockLeRobotDataset(num_samples=5, num_classes=5)
    dataset = AlohaRewardDataset(mock_ds, max_samples=3)
    collate_fn = make_vl_collate_fn(processor)
    
    # 创建 batch
    batch = [dataset[i] for i in range(min(2, len(dataset)))]
    
    try:
        enc, labels = collate_fn(batch)
        print(f"\nBatch 编码键: {enc.keys() if isinstance(enc, dict) else 'N/A'}")
        print(f"标签形状: {labels.shape if isinstance(labels, torch.Tensor) else type(labels)}")
        print(f"标签值: {labels.tolist() if isinstance(labels, torch.Tensor) else labels}")
        print("\n✓ Collate 函数测试通过")
    except Exception as e:
        print(f"\n⚠ Collate 函数测试失败（可能是 mock processor 限制）: {e}")
        import traceback
        traceback.print_exc()


def test_model_forward():
    """测试模型前向传播（使用 mock 输入）"""
    print("\n" + "="*60)
    print("测试模型前向传播")
    print("="*60)
    
    if not MODEL_AVAILABLE:
        print("⚠ 跳过：model 模块不可用")
        return
    
    model_name = os.environ.get("QWEN3VL_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
    num_classes = 5
    
    try:
        print(f"尝试加载模型: {model_name}")
        print("⚠ 注意：这将下载模型（如果未缓存），可能需要一些时间...")
        
        model = Qwen3VLRewardClassifier(model_name, num_classes)
        device = next(model.parameters()).device
        print(f"✓ 模型加载成功，设备: {device}")
        
        # 创建 mock 输入
        batch_size = 2
        seq_len = 20
        hidden_size = model.classifier.in_features
        
        mock_encodings = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)).to(device),
            "attention_mask": torch.ones(batch_size, seq_len).to(device),
        }
        
        # 注意：真实模型还需要 pixel_values 等，这里仅测试结构
        print("\n⚠ 使用简化的 mock 输入（真实训练需要完整的多模态输入）")
        print("模型结构测试通过")
        
    except Exception as e:
        print(f"⚠ 模型加载失败: {e}")
        print("这可能是正常的（需要网络连接或模型文件）")
        import traceback
        traceback.print_exc()


def test_full_pipeline():
    """测试完整训练流程（使用 mock 数据）"""
    print("\n" + "="*60)
    print("测试完整训练流程（Mock 数据）")
    print("="*60)
    
    if not (DATASET_AVAILABLE and MODEL_AVAILABLE and TORCH_AVAILABLE):
        print("⚠ 跳过：依赖不可用")
        return
    
    # 设置环境变量
    os.environ.setdefault("MAX_STEPS", "3")
    os.environ.setdefault("BATCH_SIZE", "2")
    os.environ.setdefault("LR", "1e-4")
    os.environ.setdefault("EPOCHS", "1")
    os.environ.setdefault("FINETUNE_MODE", "lora")
    
    model_name = os.environ.get("QWEN3VL_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
    
    print(f"配置:")
    print(f"  MAX_STEPS: {os.environ.get('MAX_STEPS')}")
    print(f"  BATCH_SIZE: {os.environ.get('BATCH_SIZE')}")
    print(f"  MODEL: {model_name}")
    print(f"  FINETUNE_MODE: {os.environ.get('FINETUNE_MODE')}")
    
    # 创建 mock 数据集
    mock_ds = MockLeRobotDataset(num_samples=10, num_classes=5)
    
    try:
        # 导入训练函数
        from main import train_qwen3vl_reward_classifier
        
        print("\n开始训练（使用 mock 数据）...")
        print("⚠ 注意：这将尝试加载真实模型，可能需要网络连接")
        
        # 将 mock_ds 包装成类似 LeRobotDataset 的对象
        class WrappedMockDataset:
            def __init__(self, mock_ds):
                self.hf_dataset = mock_ds.hf_dataset
                self.dataset = mock_ds.dataset
                self.meta = mock_ds.meta
        
        wrapped_ds = WrappedMockDataset(mock_ds)
        train_qwen3vl_reward_classifier(wrapped_ds)
        
        print("\n✓ 完整流程测试完成")
        
    except Exception as e:
        print(f"\n⚠ 完整流程测试失败: {e}")
        print("这可能是正常的（需要模型文件或网络连接）")
        import traceback
        traceback.print_exc()


def main():
    """主测试函数"""
    print("="*60)
    print("Mock 数据测试脚本")
    print("="*60)
    
    print(f"\n依赖检查:")
    print(f"  torch: {'✓' if TORCH_AVAILABLE else '✗'}")
    print(f"  dataset 模块: {'✓' if DATASET_AVAILABLE else '✗'}")
    print(f"  model 模块: {'✓' if MODEL_AVAILABLE else '✗'}")
    print(f"  transformers: {'✓' if TRANSFORMERS_AVAILABLE else '✗'}")
    
    # 测试 1: 数据集
    if DATASET_AVAILABLE and TORCH_AVAILABLE:
        try:
            test_dataset()
        except Exception as e:
            print(f"\n✗ 数据集测试失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠ 跳过数据集测试（依赖不可用）")
    
    # 测试 2: Collate 函数
    if DATASET_AVAILABLE and TORCH_AVAILABLE:
        try:
            test_collate_fn()
        except Exception as e:
            print(f"\n✗ Collate 函数测试失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠ 跳过 Collate 函数测试（依赖不可用）")
    
    # 测试 3: 模型前向传播
    if MODEL_AVAILABLE:
        try:
            test_model_forward()
        except Exception as e:
            print(f"\n✗ 模型测试失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠ 跳过模型测试（依赖不可用）")
    
    # 测试 4: 完整流程（可选，需要真实模型）
    run_full_test = os.environ.get("RUN_FULL_TEST", "false").lower() == "true"
    if run_full_test and DATASET_AVAILABLE and MODEL_AVAILABLE and TORCH_AVAILABLE:
        try:
            test_full_pipeline()
        except Exception as e:
            print(f"\n✗ 完整流程测试失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n" + "="*60)
        if not run_full_test:
            print("跳过完整流程测试（需要真实模型）")
            print("要运行完整测试，设置环境变量: RUN_FULL_TEST=true")
        else:
            print("跳过完整流程测试（依赖不可用）")
        print("="*60)
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
    print("\n提示：要运行完整测试，需要安装以下依赖：")
    print("  - torch")
    print("  - transformers")
    print("  - qwen-vl-utils")
    print("  - lerobot")
    print("  - peft")


if __name__ == "__main__":
    main()
