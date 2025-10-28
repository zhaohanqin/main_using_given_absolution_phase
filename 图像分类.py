"""
图像分类工具 - 支持可变相移步数

功能：
  将结构光扫描的条纹图像按照频率分类到不同的子文件夹，并按顺序重命名

支持的相移步数：
  - 3步相移：总共18张图像（h1-h9, v1-v9），每个文件夹6张（3h+3v）
  - 4步相移：总共24张图像（h1-h12, v1-v12），每个文件夹8张（4h+4v）

分类规则：
  3步相移（N=3）：
    - 频率81: h1-h3, v1-v3 → I1-I6
    - 频率72: h4-h6, v4-v6 → I1-I6
    - 频率64: h7-h9, v7-v9 → I1-I6
  
  4步相移（N=4）：
    - 频率81: h1-h4, v1-v4 → I1-I8
    - 频率72: h5-h8, v5-v8 → I1-I8
    - 频率64: h9-h12, v9-v12 → I1-I8

特性：
  ✓ 自然排序（h1, h2, ..., h10, h11, h12），避免字典序错误
  ✓ 自动验证图像数量
  ✓ 支持复制或移动模式
  ✓ 详细的处理日志
"""

import os
import shutil
import re

def natural_sort_key(text):
    """
    自然排序的键函数，使得 h1, h2, ..., h10, h11, h12 能正确排序
    而不是字典序 h1, h10, h11, h12, h2, ...
    """
    def atoi(text):
        return int(text) if text.isdigit() else text
    
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def classify_images_by_frequency(source_folder, 
                                  phase_shift_steps=4,
                                  folder_names=["81", "72", "64"],
                                  copy_mode=True,
                                  extensions=(".bmp", ".jpg", ".png")):
    """
    将包含图像的文件夹按照频率分类到不同的子文件夹，并重命名为I1-I(2N)
    
    分类规则（以N步相移为例，N=3或4）：
    - 每个频率文件夹包含 2N 张图像（N张h图像 + N张v图像）
    - 3步相移：每个文件夹6张（h×3 + v×3）
      * 频率81: h1-h3, v1-v3 → 重命名为 I1-I6
      * 频率72: h4-h6, v4-v6 → 重命名为 I1-I6
      * 频率64: h7-h9, v7-v9 → 重命名为 I1-I6
    - 4步相移：每个文件夹8张（h×4 + v×4）
      * 频率81: h1-h4, v1-v4 → 重命名为 I1-I8
      * 频率72: h5-h8, v5-v8 → 重命名为 I1-I8
      * 频率64: h9-h12, v9-v12 → 重命名为 I1-I8
    
    参数:
        source_folder: 源文件夹路径
        phase_shift_steps: 相移步数（3或4），决定每个频率的图像数量
        folder_names: 三个子文件夹的名称列表 [频率1, 频率2, 频率3]
        copy_mode: True=复制文件（保留原文件），False=移动文件（删除原文件）
        extensions: 可识别的图像后缀
    """
    if not os.path.exists(source_folder):
        print(f"❌ 错误: 源文件夹不存在: {source_folder}")
        return
    
    # 验证参数
    if len(folder_names) != 3:
        print(f"❌ 错误: folder_names必须包含3个文件夹名称，当前为{len(folder_names)}个")
        return
    
    if phase_shift_steps not in [3, 4]:
        print(f"❌ 错误: phase_shift_steps必须是3或4，当前为{phase_shift_steps}")
        return
    
    # 计算每个频率需要的图像数量
    images_per_freq = phase_shift_steps  # 每个频率的h或v图像数量
    total_images_per_freq = 2 * images_per_freq  # 每个频率总共的图像数（h+v）
    expected_h_images = 3 * images_per_freq  # 期望的h图像总数
    expected_v_images = 3 * images_per_freq  # 期望的v图像总数
    
    print(f"=" * 70)
    print(f"相移步数配置: {phase_shift_steps}步")
    print(f"每个频率文件夹: {total_images_per_freq}张图像 ({images_per_freq}张h + {images_per_freq}张v)")
    print(f"期望图像总数: {expected_h_images + expected_v_images}张 ({expected_h_images}张h + {expected_v_images}张v)")
    print(f"=" * 70)
    
    # 获取所有图像文件
    all_files = [f for f in os.listdir(source_folder) 
                 if f.lower().endswith(extensions) and 
                 os.path.isfile(os.path.join(source_folder, f))]
    
    # 分离h和v图像，使用自然排序（数字大小排序，而非字典序）
    h_images = sorted([f for f in all_files if f.lower().startswith('h')], key=natural_sort_key)
    v_images = sorted([f for f in all_files if f.lower().startswith('v')], key=natural_sort_key)
    
    print(f"\n📂 源文件夹: {source_folder}")
    print(f"📊 找到图像文件: {len(all_files)} 张")
    print(f"   - h图像: {len(h_images)} 张: {h_images if len(h_images) <= 12 else h_images[:12] + ['...']}")
    print(f"   - v图像: {len(v_images)} 张: {v_images if len(v_images) <= 12 else v_images[:12] + ['...']}")
    
    # 检查图像数量
    if len(h_images) < expected_h_images or len(v_images) < expected_v_images:
        print(f"\n⚠️ 警告: 图像数量不足")
        print(f"   h图像: {len(h_images)}/{expected_h_images}, v图像: {len(v_images)}/{expected_v_images}")
        user_input = input("是否继续处理？(y/n): ")
        if user_input.lower() != 'y':
            print("取消处理。")
            return
    
    # 动态定义分类规则（根据相移步数）
    # 每个频率对应的h和v图像范围（使用切片索引）
    classification_rules = {
        folder_names[0]: {  # 频率1（如81）
            'h_range': (0, images_per_freq),
            'v_range': (0, images_per_freq),
        },
        folder_names[1]: {  # 频率2（如72）
            'h_range': (images_per_freq, 2 * images_per_freq),
            'v_range': (images_per_freq, 2 * images_per_freq),
        },
        folder_names[2]: {  # 频率3（如64）
            'h_range': (2 * images_per_freq, 3 * images_per_freq),
            'v_range': (2 * images_per_freq, 3 * images_per_freq),
        }
    }
    
    operation = "复制" if copy_mode else "移动"
    print(f"\n⚙️ 操作模式: {operation}文件")
    print(f"📁 目标文件夹: {folder_names}")
    print("="*70)
    
    # 创建子文件夹并分类图像
    for folder_name, ranges in classification_rules.items():
        # 创建目标文件夹
        target_folder = os.path.join(source_folder, folder_name)
        os.makedirs(target_folder, exist_ok=True)
        
        h_start, h_end = ranges['h_range']
        v_start, v_end = ranges['v_range']
        
        print(f"\n📂 正在处理文件夹: {folder_name}")
        print(f"   h图像索引范围: {h_start}-{h_end-1} (共{h_end-h_start}张)")
        print(f"   v图像索引范围: {v_start}-{v_end-1} (共{v_end-v_start}张)")
        
        # 收集需要处理的图像（h图像 + v图像）
        images_to_process = []
        
        # 添加h图像
        selected_h = []
        for i in range(h_start, h_end):
            if i < len(h_images):
                images_to_process.append(h_images[i])
                selected_h.append(h_images[i])
        
        # 添加v图像
        selected_v = []
        for i in range(v_start, v_end):
            if i < len(v_images):
                images_to_process.append(v_images[i])
                selected_v.append(v_images[i])
        
        print(f"   实际选中的h图像: {selected_h}")
        print(f"   实际选中的v图像: {selected_v}")
        print(f"   将重命名为: I1-I{len(images_to_process)}")
        
        # 复制/移动并重命名为I1-I(2N)
        for new_idx, img_name in enumerate(images_to_process, start=1):
            source_path = os.path.join(source_folder, img_name)
            
            # 获取文件扩展名
            ext = os.path.splitext(img_name)[1]
            new_name = f"I{new_idx}{ext}"
            target_path = os.path.join(target_folder, new_name)
            
            try:
                if copy_mode:
                    shutil.copy2(source_path, target_path)
                    print(f"   ✅ 复制: {img_name} → {folder_name}/{new_name}")
                else:
                    shutil.move(source_path, target_path)
                    print(f"   ✅ 移动: {img_name} → {folder_name}/{new_name}")
            except Exception as e:
                print(f"   ❌ 错误: {img_name} - {str(e)}")
    
    print("\n" + "="*70)
    print("✅ 图像分类和重命名完成！")
    print(f"\n分类结果:")
    for folder_name in folder_names:
        target_folder = os.path.join(source_folder, folder_name)
        if os.path.exists(target_folder):
            files = sorted([f for f in os.listdir(target_folder) 
                           if f.lower().endswith(extensions)])
            count = len(files)
            print(f"  📁 {folder_name}/: {count} 张图像 - {files if count <= 8 else files[:8] + ['...']}")


def rename_images_in_subfolders(root_dir, extensions=(".bmp", ".jpg", ".png")):
    """
    将 root_dir 文件夹下的每个子文件夹中的 8 张图片按顺序重命名为 I1-I8

    参数:
        root_dir: 根目录路径
        extensions: 可识别的图像后缀
    """
    for folder_name in os.listdir(root_dir):
        subfolder = os.path.join(root_dir, folder_name)
        if not os.path.isdir(subfolder):
            continue

        # 获取子文件夹中所有图片文件
        images = [f for f in os.listdir(subfolder)
                  if f.lower().endswith(extensions)]
        images.sort(key=natural_sort_key)  # 使用自然排序（按数字大小排序）

        if len(images) < 8:
            print(f"⚠️ 子文件夹 {folder_name} 中图片数量不足 8 张（实际 {len(images)} 张），跳过。")
            continue

        print(f"📂 正在处理文件夹: {folder_name}")
        for i, img_name in enumerate(images[:8], start=1):
            old_path = os.path.join(subfolder, img_name)
            ext = os.path.splitext(img_name)[1]
            new_name = f"I{i}{ext}"
            new_path = os.path.join(subfolder, new_name)

            os.rename(old_path, new_path)
            print(f"    ✅ {img_name} → {new_name}")

    print("\n✅ 所有子文件夹处理完成！")


if __name__ == "__main__":
    # =========================================================================
    # 使用示例
    # =========================================================================
    
    # 示例1: 4步相移（24张图像：h1-h12, v1-v12）
    source_directory = r"E:\code\images\05_luminance200_pillar\12sided_pillar_01"
    classify_images_by_frequency(
        source_folder=source_directory,
        phase_shift_steps=4,  # 4步相移，每个文件夹8张（4h+4v）
        folder_names=["81", "72", "64"],  # 可以修改这些名称
        copy_mode=True,  # True=复制文件，False=移动文件
        extensions=(".bmp", ".jpg", ".png")
    )
    
    # 示例2: 3步相移（18张图像：h1-h9, v1-v9）
    # classify_images_by_frequency(
    #     source_folder=r"D:\images\my_folder",
    #     phase_shift_steps=3,  # 3步相移，每个文件夹6张（3h+3v）
    #     folder_names=["81", "72", "64"],
    #     copy_mode=True,
    #     extensions=(".bmp", ".jpg", ".png")
    # )
    
    # 示例3: 使用自定义文件夹名称
    # classify_images_by_frequency(
    #     source_folder=r"D:\images\my_folder",
    #     phase_shift_steps=4,
    #     folder_names=["freq_high", "freq_mid", "freq_low"],  # 自定义名称
    #     copy_mode=False,  # 移动文件（删除原文件）
    # )
    
    # 示例4: 原有的重命名功能（如果需要）
    # rename_images_in_subfolders(r"E:\code\images\05_luminance200_pillar\test")
