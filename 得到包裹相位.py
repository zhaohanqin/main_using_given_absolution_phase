"""
四步相移法包裹相位计算模块

功能：
    从指定文件夹读取四张相移图像，使用四步相移算法计算包裹相位

作者: AI Assistant
日期: 2025-10-09
"""

import os
import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List

# 设置matplotlib支持中文显示
try:
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=10)
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
except:
    try:
        font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc", size=10)
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    except:
        try:
            font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=10)
            matplotlib.rcParams['font.sans-serif'] = ['SimSun']
        except:
            print("警告: 找不到中文字体，标题可能无法正确显示")
            font = None

matplotlib.rcParams['axes.unicode_minus'] = False


def read_images_from_folder(folder_path: str, num_images: int = 4) -> List[np.ndarray]:
    """
    从指定文件夹读取图像
    
    参数:
        folder_path: str - 图像文件夹路径
        num_images: int - 期望读取的图像数量，默认为4
        
    返回:
        images: List[np.ndarray] - 图像列表，每个图像为numpy数组
        
    异常:
        FileNotFoundError: 如果文件夹不存在
        ValueError: 如果图像数量不符合预期
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")
    
    # 获取文件夹中的所有文件，按文件名数字排序
    # 支持常见图像格式：.png, .jpg, .jpeg, .bmp, .tif, .tiff
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    files = [f for f in os.listdir(folder_path) 
             if os.path.splitext(f)[1].lower() in valid_extensions]
    
    # 尝试按文件名中的数字排序
    try:
        files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        # 如果文件名不是纯数字，则按字母顺序排序
        files = sorted(files)
    
    # 检查图像数量
    if len(files) < num_images:
        raise ValueError(f"文件夹中图像数量不足: 期望{num_images}张，实际{len(files)}张")
    elif len(files) > num_images:
        print(f"警告: 文件夹中有{len(files)}张图像，只使用前{num_images}张")
        files = files[:num_images]
    
    # 读取图像
    images = []
    for i, filename in enumerate(files):
        file_path = os.path.join(folder_path, filename)
        try:
            # 使用PIL读取图像并转换为灰度图
            img = Image.open(file_path).convert('L')
            img_array = np.array(img, dtype=np.float32)
            images.append(img_array)
        except Exception as e:
            raise IOError(f"无法读取图像 {filename}: {str(e)}")
    
    return images


def calculate_wrapped_phase(images: List[np.ndarray], 
                           phase_shift_step: int = 4,
                           initial_phase: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # 验证图像数量
    if len(images) != phase_shift_step:
        raise ValueError(f"图像数量({len(images)})与相移步数({phase_shift_step})不匹配")
    
    # 将图像列表转换为numpy数组 [N, H, W]
    image_stack = np.stack(images, axis=0)
    
    # 生成相移角度数组: [0, 2π/N, 4π/N, ..., 2π(N-1)/N]
    phase_angles = 2 * np.pi * np.arange(phase_shift_step, dtype=np.float32) / phase_shift_step
    phase_angles = phase_angles.reshape(-1, 1, 1)  # 形状调整为 [N, 1, 1] 便于广播
    
    # 计算正弦项（分子）和余弦项（分母）
    sin_term = np.sum(image_stack * np.sin(phase_angles), axis=0)  # Σ(I_i * sin(θ_i))
    cos_term = np.sum(image_stack * np.cos(phase_angles), axis=0)  # Σ(I_i * cos(θ_i))
    
    # 使用arctan2计算包裹相位，范围[-π, π]
    wrapped_phase = -np.arctan2(sin_term, cos_term)
    
    # 计算调制幅值和亮度偏移
    amplitude = 2 / phase_shift_step * sin_term
    offset = 2 / phase_shift_step * cos_term
    
    # 归一化相位到[0, 1]区间，并减去初始相位
    wrapped_phase = (wrapped_phase + np.pi) / (2 * np.pi) - initial_phase
    
    # 确保相位在[0, 1]区间内
    wrapped_phase = np.mod(wrapped_phase, 1.0)
    
    return wrapped_phase, amplitude, offset


def calculate_phase_difference(phase1: np.ndarray, phase2: np.ndarray) -> np.ndarray:
    """
    计算两个包裹相位之间的差值，并自动显示结果
    
    实现了外差法的核心操作，确保相位差在[0,1]范围内。
    该函数用于多频外差法中，通过计算不同频率相位的差值来降低等效频率。
    
    参数:
        phase1: np.ndarray - 第一个包裹相位图（通常是高频相位）
        phase2: np.ndarray - 第二个包裹相位图（通常是低频相位）
        
    返回:
        phase_diff: np.ndarray - 两个相位的差值，归一化到[0,1]区间
    """
    # 检查输入形状是否一致
    if phase1.shape != phase2.shape:
        raise ValueError(f"两个相位图形状不一致: phase1={phase1.shape}, phase2={phase2.shape}")
    
    # 计算相位差
    result = phase1 - phase2
    
    # 处理负值，保证结果在[0,1]区间
    result[result < 0] += 1
    
    # 打印相位差信息
    print(f"✓ 相位差计算完成，范围: [{np.min(result):.4f}, {np.max(result):.4f}]")
    
    # 显示相位差
    plt.figure(figsize=(10, 8))
    im = plt.imshow(result, cmap='viridis')
    plt.title(f'两个频率的相位差', fontproperties=font, fontsize=14)
    plt.colorbar(im, label='Phase Difference (0-1)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return result


def generate_fringe_from_phase(wrapped_phase: np.ndarray, 
                               frequency: int = 64,
                               amplitude: float = 127.5,
                               offset: float = 127.5,
                               save_path: str = None) -> np.ndarray:
    """
    从包裹相位图生成条纹图像
    
    使用正弦函数将相位值转换为强度值，模拟结构光投影的条纹图案。
    
    参数:
        wrapped_phase: np.ndarray - 包裹相位图，范围[0, 1]
        frequency: int - 条纹频率（周期数），默认64
        amplitude: float - 调制幅度，默认127.5
        offset: float - 亮度偏移，默认127.5
        save_path: str - 保存路径，如果为None则不保存
        
    返回:
        fringe_image: np.ndarray - 生成的条纹图像，uint8类型，范围[0, 255]
        
    数学原理:
        I(x,y) = offset + amplitude * sin(2π * frequency * phase(x,y))
        
    使用示例:
        >>> phase = get_wrapped_phase_from_folder("./phase_images")
        >>> fringe = generate_fringe_from_phase(phase, frequency=64, save_path="output.png")
    """
    print(f"\n正在从包裹相位生成条纹图像 (频率={frequency})...")
    
    # 检查输入相位范围
    if np.min(wrapped_phase) < 0 or np.max(wrapped_phase) > 1:
        print(f"警告: 相位范围超出[0,1]: [{np.min(wrapped_phase):.4f}, {np.max(wrapped_phase):.4f}]")
        wrapped_phase = np.clip(wrapped_phase, 0, 1)
    
    # 使用正弦函数生成条纹图像
    # I = offset + amplitude * sin(2π * frequency * phase)
    fringe_image = offset + amplitude * np.sin(2 * np.pi * frequency * wrapped_phase)
    
    # 确保在有效范围内并转换为uint8
    fringe_image = np.clip(fringe_image, 0, 255).astype(np.uint8)
    
    print(f"✓ 条纹图像生成完成，强度范围: [{np.min(fringe_image)}, {np.max(fringe_image)}]")
    
    # 保存图像
    if save_path:
        Image.fromarray(fringe_image).save(save_path)
        print(f"✓ 条纹图像已保存到: {save_path}")
    
    # 显示结果
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 显示包裹相位
    im1 = axes[0].imshow(wrapped_phase, cmap='viridis')
    axes[0].set_title('输入: 包裹相位', fontproperties=font, fontsize=12)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], label='Phase (0-1)')
    
    # 显示生成的条纹
    im2 = axes[1].imshow(fringe_image, cmap='gray')
    axes[1].set_title(f'输出: 条纹图像 (频率={frequency})', fontproperties=font, fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], label='Intensity (0-255)')
    
    plt.tight_layout()
    plt.show()
    
    return fringe_image


def visualize_phase_periodic_pattern(phase: np.ndarray,
                                     frequency: int = 64,
                                     is_wrapped: bool = True,
                                     save_path: str = None,
                                     row_indices: list = None) -> None:
    """
    显示相位的周期性变化模式（锯齿波图）
    
    从包裹相位或解包裹相位中提取横截面，显示相位在空间上的周期性变化。
    每个周期内相位从0增长到最大值，形成锯齿波图案，用于观察相位的周期性特征。
    
    参数:
        phase: np.ndarray - 相位图（包裹相位范围[0,1]或解包裹相位）
        frequency: int - 条纹频率（周期数），仅用于包裹相位，默认64
        is_wrapped: bool - 是否为包裹相位，True表示包裹相位[0,1]，False表示解包裹相位
        save_path: str - 保存路径，如果为None则不保存
        row_indices: list - 要显示的行索引列表，默认为[height//4, height//2, height*3//4]
        
    返回:
        None - 直接显示图像
        
    使用示例:
        >>> # 从包裹相位生成周期性变化图
        >>> phase_wrapped = cv.imread("phase_1x_wrapped.png", cv.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
        >>> visualize_phase_periodic_pattern(phase_wrapped, frequency=64, is_wrapped=True, save_path="1.png")
        
        >>> # 从解包裹相位生成周期性变化图
        >>> phase_unwrapped = cv.imread("unwrapped_phase.tiff", cv.IMREAD_UNCHANGED).astype(np.float32)
        >>> visualize_phase_periodic_pattern(phase_unwrapped, is_wrapped=False, save_path="2.png")
    """
    print(f"\n正在生成相位周期性变化图...")
    print(f"  - 相位类型: {'包裹相位' if is_wrapped else '解包裹相位'}")
    print(f"  - 相位范围: [{np.min(phase):.4f}, {np.max(phase):.4f}]")
    
    # 处理多通道图像（转换为灰度）
    if len(phase.shape) == 3:
        print(f"  - 检测到多通道图像，形状: {phase.shape}")
        # 如果是3通道，取第一个通道
        phase = phase[:, :, 0]
        print(f"  - 已转换为单通道，新形状: {phase.shape}")
    
    height, width = phase.shape
    
    # 确定要显示的行
    if row_indices is None:
        row_indices = [height // 4, height // 2, height * 3 // 4]
    
    # 处理相位数据：将其转换为周期内的相位变化
    if is_wrapped:
        # 包裹相位：乘以频率得到总相位变化
        phase_periodic = phase * frequency
        period_label = f"相位变化 (0-{frequency})"
    else:
        # 解包裹相位：已经是累积相位，直接显示
        phase_periodic = phase
        period_label = "相位值"
    
    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # ====================================================================
    # 子图1: 横截面线条图（锯齿波图）
    # ====================================================================
    for idx, row_idx in enumerate(row_indices):
        line_data = phase_periodic[row_idx, :]
        axes[0].plot(line_data, label=f'行 {row_idx}', linewidth=1.5, alpha=0.8)
    
    axes[0].set_title('相位横截面 - 周期性变化模式（锯齿波）', fontproperties=font, fontsize=14)
    axes[0].set_xlabel('列索引 (pixels)', fontsize=12)
    axes[0].set_ylabel(period_label, fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # ====================================================================
    # 子图2: 相位图像（伪彩色）
    # ====================================================================
    im = axes[1].imshow(phase_periodic, cmap='viridis', aspect='auto')
    axes[1].set_title('相位图像 - 伪彩色显示', fontproperties=font, fontsize=14)
    axes[1].set_xlabel('列索引 (pixels)', fontsize=12)
    axes[1].set_ylabel('行索引 (pixels)', fontsize=12)
    
    # 标记横截面位置
    for row_idx in row_indices:
        axes[1].axhline(y=row_idx, color='white', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.colorbar(im, ax=axes[1], label=period_label)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 相位周期性变化图已保存到: {save_path}")
    
    plt.show()
    
    print(f"✓ 相位周期性变化图生成完成")
    
    # 统计信息
    print(f"\n统计信息:")
    print(f"  - 图像尺寸: {height} × {width}")
    print(f"  - 显示行数: {len(row_indices)}")
    if is_wrapped:
        print(f"  - 条纹频率: {frequency}")
        print(f"  - 相位范围: [0, {frequency}]")
    else:
        print(f"  - 相位范围: [{np.min(phase):.2f}, {np.max(phase):.2f}]")


def visualize_unwrapped_phase(unwrapped_phase: np.ndarray,
                              mask: np.ndarray = None,
                              output_prefix: str = "unwrapped_phase",
                              save_dir: str = "./") -> dict:
    """
    从解包裹相位生成多种可视化图像，用于检查伽马波纹和最终结果
    
    生成四种可视化图像：
        1. 灰度图（归一化显示）
        2. 伪彩色图（jet colormap）
        3. 横截面切片图（用于查看伽马波纹）
        4. 3D表面图（整体形态）
        
    参数:
        unwrapped_phase: np.ndarray - 解包裹相位图
        mask: np.ndarray - 可选的掩码，用于屏蔽无效区域
        output_prefix: str - 输出文件名前缀
        save_dir: str - 保存目录，默认为当前目录
        
    返回:
        result_dict: dict - 包含所有生成图像的字典
        
    使用示例:
        >>> unwrapped = load_unwrapped_phase("phase.tiff")
        >>> results = visualize_unwrapped_phase(unwrapped, output_prefix="final_phase")
    """
    print(f"\n正在生成解包裹相位的可视化图像...")
    print(f"相位范围: [{np.min(unwrapped_phase):.2f}, {np.max(unwrapped_phase):.2f}]")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 应用掩码（如果提供）
    if mask is not None:
        display_phase = unwrapped_phase.copy()
        display_phase[mask == 0] = np.nan
    else:
        display_phase = unwrapped_phase
    
    result_dict = {}
    
    # ====================================================================
    # 图1: 灰度图（归一化到0-255）
    # ====================================================================
    print("正在生成灰度图...")
    
    # 归一化到0-255
    valid_data = unwrapped_phase[~np.isnan(display_phase)] if mask is not None else unwrapped_phase.flatten()
    phase_min, phase_max = np.min(valid_data), np.max(valid_data)
    normalized_phase = (unwrapped_phase - phase_min) / (phase_max - phase_min) * 255
    gray_image = normalized_phase.astype(np.uint8)
    
    if mask is not None:
        gray_image_masked = gray_image.copy()
        gray_image_masked[mask == 0] = 0
        result_dict['gray'] = gray_image_masked
        save_path_1 = os.path.join(save_dir, f"{output_prefix}_gray.png")
        Image.fromarray(gray_image_masked).save(save_path_1)
    else:
        result_dict['gray'] = gray_image
        save_path_1 = os.path.join(save_dir, f"{output_prefix}_gray.png")
        Image.fromarray(gray_image).save(save_path_1)
    
    print(f"✓ 灰度图已保存: {save_path_1}")
    
    # ====================================================================
    # 图2: 伪彩色图（viridis colormap - 冷色调）
    # ====================================================================
    print("正在生成伪彩色图...")
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(display_phase, cmap='viridis')
    plt.title(f'解包裹相位 - 伪彩色图', fontproperties=font, fontsize=14)
    plt.colorbar(im, label='Phase Value')
    plt.axis('off')
    plt.tight_layout()
    
    save_path_2 = os.path.join(save_dir, f"{output_prefix}_colormap.png")
    plt.savefig(save_path_2, dpi=150, bbox_inches='tight')
    plt.show()
    
    result_dict['colormap_path'] = save_path_2
    print(f"✓ 伪彩色图已保存: {save_path_2}")
    
    # ====================================================================
    # 图3: 横截面切片图（查看伽马波纹）
    # ====================================================================
    print("正在生成横截面切片图...")
    
    height, width = unwrapped_phase.shape
    
    # 选择横向和纵向的中心线，以及1/4和3/4位置的切片
    row_indices = [height // 4, height // 2, height * 3 // 4]
    col_indices = [width // 4, width // 2, width * 3 // 4]

    # 改为你想要的行号，例如：
    row_indices = [100, 200, 300]  # 显示第100、200、300行
    col_indices = [200, 400, 600]  # 显示第200、400、600列
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 横向切片（查看横向伽马波纹）
    for row_idx in row_indices:
        line_data = unwrapped_phase[row_idx, :]
        axes[0].plot(line_data, label=f'行 {row_idx}', linewidth=1.5, alpha=0.8)
    
    axes[0].set_title('横向切片 - 用于检查横向伽马波纹', fontproperties=font, fontsize=12)
    axes[0].set_xlabel('列索引 (pixels)')
    axes[0].set_ylabel('相位值')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 纵向切片（查看纵向伽马波纹）
    for col_idx in col_indices:
        line_data = unwrapped_phase[:, col_idx]
        axes[1].plot(line_data, label=f'列 {col_idx}', linewidth=1.5, alpha=0.8)
    
    axes[1].set_title('纵向切片 - 用于检查纵向伽马波纹', fontproperties=font, fontsize=12)
    axes[1].set_xlabel('行索引 (pixels)')
    axes[1].set_ylabel('相位值')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path_3 = os.path.join(save_dir, f"{output_prefix}_cross_section.png")
    plt.savefig(save_path_3, dpi=150, bbox_inches='tight')
    plt.show()
    
    result_dict['cross_section_path'] = save_path_3
    print(f"✓ 横截面切片图已保存: {save_path_3}")
    
    # ====================================================================
    # 图4: 3D表面图（整体形态）
    # ====================================================================
    print("正在生成3D表面图...")
    
    # 为了提高性能，对大图像进行降采样
    downsample_factor = max(1, max(height, width) // 500)
    if downsample_factor > 1:
        phase_3d = unwrapped_phase[::downsample_factor, ::downsample_factor]
        print(f"  (降采样因子: {downsample_factor}, 用于3D显示)")
    else:
        phase_3d = unwrapped_phase
    
    h_3d, w_3d = phase_3d.shape
    x_3d = np.arange(0, w_3d)
    y_3d = np.arange(0, h_3d)
    X_3d, Y_3d = np.meshgrid(x_3d, y_3d)
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X_3d, Y_3d, phase_3d, cmap='viridis', 
                           linewidth=0, antialiased=True, alpha=0.9)
    
    ax.set_title('解包裹相位 - 3D表面图', fontproperties=font, fontsize=14, pad=20)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel('Phase Value')
    
    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    save_path_4 = os.path.join(save_dir, f"{output_prefix}_3d_surface.png")
    plt.savefig(save_path_4, dpi=150, bbox_inches='tight')
    plt.show()
    
    result_dict['3d_surface_path'] = save_path_4
    print(f"✓ 3D表面图已保存: {save_path_4}")
    
    # ====================================================================
    # 汇总输出
    # ====================================================================
    print("\n" + "=" * 70)
    print("✓ 所有可视化图像生成完成！")
    print("=" * 70)
    print(f"  1. 灰度图:       {save_path_1}")
    print(f"  2. 伪彩色图:     {save_path_2}")
    print(f"  3. 横截面切片图: {save_path_3}")
    print(f"  4. 3D表面图:     {save_path_4}")
    print("=" * 70 + "\n")
    
    return result_dict


def compare_two_folders(folder1: str, folder2: str, num_images: int = 8) -> bool:
    """
    比较两个文件夹中的前N张图像是否完全相同
    
    读取两个文件夹的前N张图像，按顺序逐一对比，判断是否完全一致。
    通过计算图像差值来判断：如果差值全为0（全黑），则图像相同。
    
    参数:
        folder1: str - 第一个文件夹路径
        folder2: str - 第二个文件夹路径
        num_images: int - 要比较的图像数量，默认为8
        
    返回:
        is_identical: bool - True表示所有图像完全相同，False表示存在差异
        
    使用示例:
        >>> result = compare_two_folders(r"E:\\data\\folder1", r"E:\\data\\folder2")
        >>> if result:
        ...     print("两个文件夹的图像完全相同")
        ... else:
        ...     print("两个文件夹的图像存在差异")
    """
    print("=" * 70)
    print(f"开始比较两个文件夹的前 {num_images} 张图像")
    print("=" * 70)
    print(f"文件夹1: {folder1}")
    print(f"文件夹2: {folder2}")
    print("-" * 70)
    
    try:
        # 读取两个文件夹的图像
        print(f"\n正在读取文件夹1的图像...")
        images1 = read_images_from_folder(folder1, num_images=num_images)
        
        print(f"\n正在读取文件夹2的图像...")
        images2 = read_images_from_folder(folder2, num_images=num_images)
        
        # 检查图像数量是否一致
        if len(images1) != len(images2):
            print(f"\n⚠ 警告: 图像数量不一致！文件夹1有{len(images1)}张，文件夹2有{len(images2)}张")
            return False
        
        # 逐一比较图像
        print(f"\n开始逐一比较图像...")
        all_identical = True
        diff_images = []
        
        for i, (img1, img2) in enumerate(zip(images1, images2)):
            # 检查形状是否一致
            if img1.shape != img2.shape:
                print(f"  [图像 {i+1}] ✗ 形状不一致: {img1.shape} vs {img2.shape}")
                all_identical = False
                continue
            
            # 计算差值
            diff = cv.absdiff(img1, img2)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            # 判断是否相同（差值是否全为0）
            is_same = (max_diff == 0)
            
            if is_same:
                print(f"  [图像 {i+1}] ✓ 完全相同 (差值: 0)")
            else:
                print(f"  [图像 {i+1}] ✗ 存在差异 (最大差值: {max_diff:.2f}, 平均差值: {mean_diff:.4f})")
                all_identical = False
            
            # 保存差值图像用于可视化
            diff_images.append(diff)
        
        # 可视化结果
        print(f"\n生成对比可视化...")
        n_cols = min(4, num_images)
        n_rows = (num_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        if num_images == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, diff in enumerate(diff_images):
            if i < len(axes):
                # 归一化差值图像以便显示
                diff_normalized = cv.normalize(diff, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
                
                axes[i].imshow(diff_normalized, cmap='hot')
                axes[i].set_title(f'图像 {i+1} 差值\n{"相同" if np.max(diff) == 0 else f"最大差:{np.max(diff):.1f}"}', 
                                fontproperties=font, fontsize=10)
                axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(len(diff_images), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'两个文件夹图像差值对比\n{"✓ 所有图像完全相同" if all_identical else "✗ 存在差异"}', 
                     fontproperties=font, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # 输出最终结果
        print("\n" + "=" * 70)
        if all_identical:
            print("✓ 结论: 两个文件夹的所有图像完全相同！")
        else:
            print("✗ 结论: 两个文件夹的图像存在差异！")
        print("=" * 70 + "\n")
        
        return all_identical
        
    except Exception as e:
        print(f"\n✗ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def get_wrapped_phase_from_folder(folder_path: str,
                                  phase_shift_step: int = 4,
                                  initial_phase: float = 0.0) -> np.ndarray:
    """
    主函数：从文件夹读取图像并计算包裹相位
    
    参数:
        folder_path: str - 包含相移图像的文件夹路径
        phase_shift_step: int - 相移步数，默认为4
        initial_phase: float - 初始相位偏移，默认为0.0
        
    返回:
        wrapped_phase: np.ndarray - 包裹相位图，归一化到[0,1]区间
        
    使用示例:
        >>> phase = get_wrapped_phase_from_folder("./phase_images")
        >>> phase = get_wrapped_phase_from_folder("./data/images", phase_shift_step=4)
    """
    print(f"正在处理文件夹: {folder_path}")
    
    # 步骤1: 读取图像
    images = read_images_from_folder(folder_path, num_images=phase_shift_step)
    
    # 步骤2: 计算包裹相位
    wrapped_phase, _, _ = calculate_wrapped_phase(
        images, 
        phase_shift_step=phase_shift_step,
        initial_phase=initial_phase
    )
    
    print(f"✓ 包裹相位计算完成，范围: [{np.min(wrapped_phase):.4f}, {np.max(wrapped_phase):.4f}]")
    
    # 步骤3: 显示包裹相位
    plt.figure(figsize=(10, 8))
    im = plt.imshow(wrapped_phase, cmap='viridis')
    plt.title(f'包裹相位', fontproperties=font, fontsize=14)
    plt.colorbar(im, label='Phase (0-1)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return wrapped_phase


# ============================================================================
# 主程序入口（用于测试）
# ============================================================================
if __name__ == "__main__":
    """
    使用示例
    """
    
    # ========================================
    # 示例1: 比较两个文件夹的图像是否完全相同
    # ========================================
    # result = compare_two_folders(
    #     folder1=r"C:\Users\Administrator\Desktop\images\images\64\debug_fringe_images",
    #     folder2=r"C:\Users\Administrator\Desktop\images\test_images\64",
    #     num_images=8
    # )
    
    # if result:
    #     print("✓ 结果: 两个文件夹的图像完全相同")
    # else:
    #     print("✗ 结果: 两个文件夹的图像存在差异")
    
    # ========================================
    # 示例2: 计算包裹相位
    # ========================================
    # phase_64 = get_wrapped_phase_from_folder(r"C:\Users\Administrator\Desktop\images\images\SLMasterimages\30")
    # phase_63 = get_wrapped_phase_from_folder(r"C:\Users\Administrator\Desktop\images\images\SLMasterimages\29")
    # phase_64 = get_wrapped_phase_from_folder(r"E:\\data\\64")
    
    # ========================================
    # 示例3: 计算相位差
    # ========================================
    # phase_diff_15_14 = calculate_phase_difference(phase_64, phase_63)
    # phase_diff_56_49 = calculate_phase_difference(phase_56, phase_49)
    
    # ========================================
    # 示例4: 从TIFF文件显示相位周期性变化（锯齿波图）
    # ========================================
    # 外差相位（第一次）
    tiff_path = r"three_freq_phase_unwrap_results\2_first_heterodyne\phase_12x.tiff"
    frequency = 8  # |64-56| = 8
    
    # 检查文件是否存在
    if not os.path.exists(tiff_path):
        print(f"✗ 错误：文件不存在")
        print(f"   路径: {tiff_path}")
        print(f"   请检查路径是否正确，或先运行相位解包裹程序生成TIFF文件")
    else:
        print(f"\n{'='*70}")
        print(f"正在处理: {tiff_path}")
        print(f"{'='*70}")
        
        # 加载TIFF文件
        phase = cv.imread(tiff_path, cv.IMREAD_UNCHANGED)
        
        if phase is None:
            print(f"✗ 无法读取文件: {tiff_path}")
        else:
            # 转换为float32
            phase = phase.astype(np.float32)
            
            print(f"✓ 文件加载成功")
            print(f"  - 形状: {phase.shape}")
            print(f"  - 类型: {phase.dtype}")
            print(f"  - 范围: [{np.min(phase):.6f}, {np.max(phase):.6f}]")
            print(f"  - 唯一值数量: {len(np.unique(phase))}")
            
            # 显示周期性变化（锯齿波图）
            visualize_phase_periodic_pattern(
                phase,
                frequency=frequency,
                is_wrapped=True,
                save_path="1.png"
            )
            
            print(f"✓ 可视化完成，已保存到: 1.png")
            print(f"{'='*70}\n")
    
    
    # ========================================
    # 示例5: 从解包裹相位生成可视化图像
    # ========================================
    # 使用方法：
    # 1. 加载解包裹相位图
    unwrapped = cv.imread(r"three_freq_phase_unwrap_results\5_final_results\unwrap_phase2_x_12.tiff", cv.IMREAD_UNCHANGED)
    if unwrapped.dtype == np.uint16:
        unwrapped = unwrapped.astype(np.float32)
    
    # 2. 可选：加载掩码（如果不需要掩码，设为None）
    # mask = cv.imread("three_freq_phase_unwrap_results/mask/Final Mask.png", cv.IMREAD_GRAYSCALE)
    mask = None  # 不使用掩码，显示完整的相位数据
    
    # 3. 生成可视化图像（生成4.png, 5.png, 6.png, 7.png）
    results = visualize_unwrapped_phase(
        unwrapped, 
        mask=mask,
        output_prefix="unwrapped_phase",
        save_dir="./"
    )
    # 输出文件：
    #   - unwrapped_phase_gray.png         (对应 4.png - 灰度图)
    #   - unwrapped_phase_colormap.png     (对应 5.png - 伪彩色图)
    #   - unwrapped_phase_cross_section.png (对应 6.png - 横截面切片图，查看伽马波纹)
    #   - unwrapped_phase_3d_surface.png   (对应 7.png - 3D表面图)

    # 显示解包裹相位在空间上的周期性变化
    visualize_phase_periodic_pattern(
        unwrapped, 
        is_wrapped=False,   # 解包裹相位
        save_path="2.png"   # 保存路径
    )
