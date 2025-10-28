import os
import shutil

def classify_images_by_frequency(source_folder, 
                                  folder_names=["81", "72", "64"],
                                  copy_mode=True,
                                  extensions=(".bmp", ".jpg", ".png")):
    """
    将包含24张图像的文件夹按照频率分类到不同的子文件夹，并重命名为I1-I8
    
    分类规则：
    - 频率81文件夹：h1-h4, v1-v4 → 重命名为 I1-I8
    - 频率72文件夹：h5-h8, v5-v8 → 重命名为 I1-I8
    - 频率64文件夹：h9-h12, v9-v12 → 重命名为 I1-I8
    
    参数:
        source_folder: 源文件夹路径（包含24张图像：h1-h12, v1-v12）
        folder_names: 三个子文件夹的名称列表 [频率1, 频率2, 频率3]
        copy_mode: True=复制文件（保留原文件），False=移动文件（删除原文件）
        extensions: 可识别的图像后缀
    """
    if not os.path.exists(source_folder):
        print(f"❌ 错误: 源文件夹不存在: {source_folder}")
        return
    
    if len(folder_names) != 3:
        print(f"❌ 错误: folder_names必须包含3个文件夹名称，当前为{len(folder_names)}个")
        return
    
    # 获取所有图像文件
    all_files = [f for f in os.listdir(source_folder) 
                 if f.lower().endswith(extensions) and 
                 os.path.isfile(os.path.join(source_folder, f))]
    
    # 分离h和v图像
    h_images = sorted([f for f in all_files if f.lower().startswith('h')])
    v_images = sorted([f for f in all_files if f.lower().startswith('v')])
    
    print(f"📂 源文件夹: {source_folder}")
    print(f"📊 找到图像文件: {len(all_files)} 张")
    print(f"   - h图像: {len(h_images)} 张")
    print(f"   - v图像: {len(v_images)} 张")
    
    if len(h_images) < 12 or len(v_images) < 12:
        print(f"⚠️ 警告: h或v图像数量不足12张")
        print(f"   h图像: {len(h_images)}/12, v图像: {len(v_images)}/12")
        user_input = input("是否继续处理？(y/n): ")
        if user_input.lower() != 'y':
            print("取消处理。")
            return
    
    # 定义分类规则和重命名映射
    # 每个频率对应的h和v图像范围（使用切片索引）
    classification_rules = {
        folder_names[0]: {  # 频率81: h1-h4, v1-v4
            'h_range': (0, 4),   # h_images[0:4] = h1-h4
            'v_range': (0, 4),   # v_images[0:4] = v1-v4
        },
        folder_names[1]: {  # 频率72: h5-h8, v5-v8
            'h_range': (4, 8),   # h_images[4:8] = h5-h8
            'v_range': (4, 8),   # v_images[4:8] = v5-v8
        },
        folder_names[2]: {  # 频率64: h9-h12, v9-v12
            'h_range': (8, 12),  # h_images[8:12] = h9-h12
            'v_range': (8, 12),  # v_images[8:12] = v9-v12
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
        print(f"   h图像范围: {h_start+1}-{h_end} (h{h_start+1}-h{h_end})")
        print(f"   v图像范围: {v_start+1}-{v_end} (v{v_start+1}-v{v_end})")
        
        # 收集需要处理的图像（h图像 + v图像）
        images_to_process = []
        
        # 添加h图像
        for i in range(h_start, h_end):
            if i < len(h_images):
                images_to_process.append(h_images[i])
        
        # 添加v图像
        for i in range(v_start, v_end):
            if i < len(v_images):
                images_to_process.append(v_images[i])
        
        # 复制/移动并重命名为I1-I8
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
        images.sort()  # 按文件名排序

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
    
    # 示例1: 将24张图像按频率分类（默认文件夹名称：81, 72, 64）
    source_directory = r"E:\code\images\05_luminance200_pillar\12sided_pillar_01"
    classify_images_by_frequency(
        source_folder=source_directory,
        folder_names=["81", "72", "64"],  # 可以修改这些名称
        copy_mode=True,  # True=复制文件，False=移动文件
        extensions=(".bmp", ".jpg", ".png")
    )
    
    # 示例2: 使用自定义文件夹名称
    # classify_images_by_frequency(
    #     source_folder=r"D:\images\my_folder",
    #     folder_names=["freq_high", "freq_mid", "freq_low"],  # 自定义名称
    #     copy_mode=False,  # 移动文件（删除原文件）
    # )
    
    # 示例3: 原有的重命名功能（如果需要）
    # rename_images_in_subfolders(r"E:\code\images\05_luminance200_pillar\test")
