"""
分析图像处理流程，确认三张图像是否正确送入模型
"""
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def analyze_image_flow():
    """
    分析数据流中图像的传递过程
    """
    print("=" * 80)
    print("图像处理流程分析")
    print("=" * 80)
    
    print("\n1. AlohaRewardDataset.__getitem__ (dataset.py:55-80)")
    print("   - 定义三个相机：")
    print("     * observation.images.cam_right_wrist")
    print("     * observation.images.cam_left_wrist")
    print("     * observation.images.cam_high")
    print("   - 循环处理每个相机图像（第57-80行）：")
    print("     for cam in self.cameras:")
    print("         img = row[cam]")
    print("         # 处理为 PIL.Image")
    print("         images.append(img)  # 添加到列表")
    print("   - 返回: {'images': images, 'text': text, 'label': label}")
    print("   - ✅ images 列表应该包含 3 张 PIL.Image")
    
    print("\n2. collate 函数 (dataset.py:114-178)")
    print("   - 对每个样本：")
    print("     images = sample['images']  # 获取3张图像的列表")
    print("   - 构造 messages 结构（第134-145行）：")
    print("     messages = [{'role': 'user', 'content': [{'type': 'text', 'text': text}]}]")
    print("   - 将图像插入到 messages（第147-157行）：")
    print("     for img in images:")
    print("         messages[0]['content'].insert(0, {'type': 'image', 'image': img, ...})")
    print("   - ⚠️  注意：使用 insert(0, ...) 会将图像反向插入")
    print("   - 最终顺序：cam_high, cam_left_wrist, cam_right_wrist（反向）")
    
    print("\n3. process_vision_info (dataset.py:165)")
    print("   - img_inputs, _ = process_vision_info(messages)")
    print("   - ✅ 应该处理 messages 中的所有图像（3张）")
    
    print("\n4. processor 批量处理 (dataset.py:171-176)")
    print("   - enc = processor(text=texts, images=image_inputs, ...)")
    print("   - ✅ image_inputs 列表中每个元素包含3张图像的信息")
    
    print("\n5. 模型 forward (model.py:100)")
    print("   - outputs = self.base_model(**encodings, output_hidden_states=True)")
    print("   - ✅ encodings 包含 pixel_values 等，应该是3张图像的编码")
    
    print("\n" + "=" * 80)
    print("潜在问题检查")
    print("=" * 80)
    
    print("\n✅ 确认点：")
    print("   1. __getitem__ 中确实循环了3个相机（第57行）")
    print("   2. 每个相机图像都被 append 到 images 列表（第80行）")
    print("   3. collate 中遍历所有 images（第148行）")
    print("   4. 每张图像都被插入到 messages（第149-157行）")
    
    print("\n⚠️  需要注意的点：")
    print("   1. 图像顺序：由于使用 insert(0, ...)，最终顺序是反向的")
    print("      - 插入顺序：cam_right_wrist, cam_left_wrist, cam_high")
    print("      - 最终顺序：cam_high, cam_left_wrist, cam_right_wrist")
    print("   2. 需要确认 process_vision_info 是否正确处理了多张图像")
    print("   3. 需要确认 processor 是否正确处理了 image_inputs")
    
    print("\n" + "=" * 80)
    print("建议验证方法")
    print("=" * 80)
    print("""
1. 在 collate 函数中添加调试信息：
   print(f"Number of images in sample: {len(images)}")
   print(f"Number of image entries in messages: {len([x for x in messages[0]['content'] if x['type']=='image'])}")

2. 在 model forward 中添加调试信息：
   if 'pixel_values' in encodings:
       print(f"pixel_values shape: {encodings['pixel_values'].shape}")
   if 'image_grid_thw' in encodings:
       print(f"image_grid_thw: {encodings['image_grid_thw']}")

3. 检查 process_vision_info 的返回值
""")

if __name__ == "__main__":
    analyze_image_flow()
