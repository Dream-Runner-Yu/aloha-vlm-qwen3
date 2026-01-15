## Qwen3-VL Reward 分类微调（Aloha VLM）

本目录下的代码实现了基于 **Qwen3-VL-2B-Instruct** 的多模态 reward 分类微调，用于对 Aloha 机器人任务中的轨迹/帧进行离散等级打分。核心特点：

- 数据来自 **LeRobot** 的 `realworld_aloha_success_bhc` 数据集；
- 每条样本包含 **三路相机图像 + 文本 task 描述 + reward_bins**；
- 模型骨干是 **Qwen3VLForConditionalGeneration**，在其文本最后 token 上做 pooling，加一个线性分类头；
- 使用 **交叉熵损失** 训练 reward 分类器，支持 LoRA 或全参数微调。

---

## 数据结构

数据读取整体流程在 [dataset.py](./dataset.py) 中：

- 上游数据集：`LeRobotDataset`
  - 通过 `LeRobotDataset(repo_id="lerobot/realworld_aloha_success_bhc", root=rl_dataset)` 加载；
  - 每条样本包含多路相机帧、任务元信息 `task`、奖励 `reward_bins` 等。

- 包装数据集：`AlohaRewardDataset`

  - 相机通道：
    - `"observation.images.cam_right_wrist"`
    - `"observation.images.cam_left_wrist"`
    - `"observation.images.cam_high"`
  - 对每个通道：
    - 如果是 `torch.Tensor`，会从 `[T, C, H, W]` 取第一帧；
    - 将 `[0,1]` 的 float tensor 转成 `HWC uint8`，再转为 `PIL.Image`。
  - 文本：
    - 从 `row["task"]` 读取任务名称，构造 `"Task: {task}"` 文本。
  - 标签 `label`：
    - 来自 `row["reward_bins"]`；
    - 如果是一维向量，取 `argmax` 作为类别索引；
    - 如果是标量或全零向量，则退化为 `0` 类。
  - 返回字典：
    - `"images"`：长度为 3 的 `PIL.Image` 列表；
    - `"text"`：任务描述字符串；
    - `"label"`：`int` 类型类别 id。

---

## 多模态输入构造

多模态打包逻辑在 `make_vl_collate_fn` 中完成：

- 对一个 batch 内的样本，逐个构造 Qwen3-VL 标准的 `messages` 结构：

  - `messages` 的形式为：

    ```python
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img1, ...},
                {"type": "image", "image": img2, ...},
                {"type": "image", "image": img3, ...},
                {"type": "text", "text": text},
            ],
        }
    ]
    ```

  - 三路相机图像 **一起** 放到同一个 message 里，随后是对应的文本描述。

- 使用 Qwen 官方推荐流程构造多模态输入：

  1. `chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)`
  2. `image_inputs, _ = process_vision_info(messages)`
  3. 将所有样本的 `chat_text` 和 `image_inputs` 组成列表，调用：

     ```python
     enc = processor(
         text=texts,
         images=image_inputs,
         padding=True,
         return_tensors="pt",
     )
     ```

- collate 函数最终返回：

  - `enc`：包含 `input_ids`、`attention_mask`、`pixel_values`、`image_grid_thw` 等；
  - `labels`：`LongTensor`，shape 为 `[batch_size]`。

---

## 模型结构

模型定义在 [model.py](./model.py) 中的 `Qwen3VLRewardClassifier`：

- 骨干模型：

  - `Qwen3VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")`
  - 默认从环境变量 `QWEN3VL_MODEL` 读取模型名，缺省为 `"Qwen/Qwen3-VL-2B-Instruct"`。

- LoRA 配置：

  - 环境变量 `FINETUNE_MODE` 控制是否使用 LoRA：
    - `"full"`：全参数微调（不注入 LoRA）；
    - 其他值（默认 `"lora"`）：启用 LoRA。
  - LoRA 目标模块：
    - `"q_proj"`, `"k_proj"`, `"v_proj"`, `"o_proj"`,
    - `"gate_proj"`, `"up_proj"`, `"down_proj"`.

  - LoRA 超参：
    - `r`：秩，默认 `16`；
    - `lora_alpha`：默认 `32`；
    - `lora_dropout`：`0.05`；
    - `bias`：`"none"`；
    - `task_type`：`"CAUSAL_LM"`。

- 分类头：

  - 自动推断 hidden size：
    - 优先 `self.base_model.config.hidden_size`；
    - 或 `self.base_model.config.text_config.hidden_size`；
    - 否则回落到 `2048`。
  - 分类层：
    - `nn.Linear(hidden_size, num_classes, dtype=torch.bfloat16)`；
    - `num_classes` 来自环境变量 `NUM_CLASSES` 或训练入口函数的参数。

- 前向计算：

  - 调用 Qwen3-VL 时开启 `output_hidden_states=True`；
  - 从最后一层 hidden state 中取每个样本 **最后一个非 padding token** 对应的向量：

    ```python
    attn = encodings["attention_mask"]
    idx = attn.long().sum(dim=1) - 1
    pooled = hidden[torch.arange(batch_size, device=device), idx]
    ```

  - 将 `pooled` 送入线性分类头得到 `logits`，shape `[batch_size, num_classes]`。

- 损失函数：

  - 训练代码中使用：

    ```python
    loss = nn.functional.cross_entropy(logits, labels)
    ```

  - 即标准的交叉熵分类 loss。

---

## 训练入口与参数

训练和推理入口定义在 [main.py](./main.py) 中：

- 主入口：`python main.py`

  - 环境变量 `MODE` 控制模式：
    - `"train"`（默认）：训练 Qwen3-VL reward 分类器；
    - `"chat"`：加载 checkpoint，进入交互式对话/生成模式；
    - `"analyze"`：对数据集做基础统计分析。

- 训练函数：`train_qwen3vl_reward_classifier(ds: LeRobotDataset)`

  - 推断类别数：
    - 从第一条样本的 `reward_bins` 推断 bin 数量，作为 `num_classes`。
  - 构建数据集和 DataLoader：
    - `AlohaRewardDataset` + `make_vl_collate_fn(processor)`。
  - 模型与优化器：
    - `Qwen3VLRewardClassifier(model_name, num_bins)`；
    - `torch.optim.AdamW(model.parameters(), lr=LR)`。
  - 训练超参（通过环境变量配置）：
    - `MAX_TRAIN_SAMPLES`：最大训练样本数，`None` 表示全部；
    - `BATCH_SIZE`：每设备 batch size，默认 `2`；
    - `LR`：学习率，默认 `1e-4`；
    - `EPOCHS`：最大训练轮数，默认 `10`；
    - `MAX_STEPS`：最大步骤数（优先级高于 EPOCHS），默认 `50`。

- checkpoint 保存与加载：

  - 环境变量 `CKPT_DIR` 控制保存目录，默认 `"checkpoints/qwen3_reward"`；
  - 若 `CKPT_DIR` 为相对路径，则自动拼到当前文件所在目录下；
  - 训练结束后保存：
    - `model.base_model.save_pretrained(save_dir)`；
    - `processor.save_pretrained(save_dir)`；
    - `classifier.pt`：分类头权重。

- chat / 推理模式：`chat_qwen3vl()`

  - 根据 `FINETUNE_MODE` 决定加载方式：
    - `"full"`：直接从 `CKPT_DIR` 加载全量权重；
    - 其他：从基础模型 `QWEN3VL_MODEL` + LoRA 权重 `CKPT_DIR` 组合加载。
  - 当前实现是文本对话 demo，如需图像 + 文本的 reward 预测，可在此基础上扩展。

---

## 运行示例

在已有 RL 数据和 kuavo_il 环境下，示例训练命令：

```bash
cd /data/model_checkpoint/clear_brain/notebooks_check/aloha/vlm

MAX_STEPS=100 \
MODE=train \
QWEN3VL_MODEL="Qwen/Qwen3-VL-2B-Instruct" \
FINETUNE_MODE="lora" \
conda run -n kuavo_il python main.py
```

训练完成后，可将 `MODE` 切换为 `chat` 进行测试：

```bash
MODE=chat \
CKPT_DIR="checkpoints/qwen3_reward" \
conda run -n kuavo_il python main.py
```

根据需要，你可以调整：

- LoRA vs 全量微调：`FINETUNE_MODE`；
- 模型尺寸：`QWEN3VL_MODEL`；
- 类别数：`NUM_CLASSES`；
- 训练步数和 batch 大小：`MAX_STEPS`、`BATCH_SIZE` 等。

---

## 测试与开发

### Mock 数据测试

项目包含测试脚本 [test_with_mock_data.py](./test_with_mock_data.py)，可用于测试代码结构而无需完整依赖：

```bash
# 基础测试（检查代码结构）
python test_with_mock_data.py

# 完整测试（需要安装依赖）
RUN_FULL_TEST=true python test_with_mock_data.py
```

测试脚本包含：
- ✅ 数据集加载测试
- ✅ Collate 函数测试
- ✅ 模型前向传播测试
- ✅ 完整训练流程测试（可选）

### 代码优化

代码已进行优化，主要改进：

- **类型安全**：添加了完整的类型提示（Type Hints），使用 `Optional` 和 `Union` 明确可选类型
- **代码可维护性**：添加了详细的文档字符串，改进了变量命名，提取了重复逻辑为独立方法
- **训练稳定性**：添加了梯度裁剪（`max_norm=1.0`）防止梯度爆炸
- **日志输出**：改进了训练日志格式，更清晰地显示训练进度

主要特性：
- 支持 LoRA 和全参数微调
- 自动推断 hidden_size
- 梯度裁剪防止训练不稳定
- 清晰的训练日志输出

---

## 依赖要求

完整功能需要以下依赖：

- `torch` - PyTorch 深度学习框架
- `transformers` - Hugging Face Transformers
- `qwen-vl-utils` - Qwen VL 工具包
- `lerobot` - LeRobot 数据集库
- `peft` - Parameter-Efficient Fine-Tuning
- `PIL` / `Pillow` - 图像处理
- `numpy` - 数值计算

---

## 未来改进方向

1. **添加验证集评估**：在训练过程中评估验证集性能
2. **支持混合精度训练**：使用 `torch.cuda.amp` 加速训练
3. **添加 TensorBoard 日志**：可视化训练过程
4. **支持分布式训练**：使用 `torch.distributed` 进行多 GPU 训练
5. **添加数据增强**：图像翻转、颜色抖动等
6. **添加早停机制**：基于验证集性能自动停止训练
