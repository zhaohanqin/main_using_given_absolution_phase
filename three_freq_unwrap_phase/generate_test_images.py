import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import argparse
from scipy import ndimage

def create_realistic_object_mask(width, height):
    """
    创建一个扩大的投影区域掩膜，使投影区域占大部分面积
    
    参数:
        width: 图像宽度
        height: 图像高度
        
    返回:
        mask: 投影区域掩膜，True表示物体表面，False表示背景
        depth_map: 深度图，用于模拟物体的3D形状
    """
    # 创建坐标网格
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # 创建更大面积的投影区域 - 覆盖图像的绝大部分区域
    # 进一步扩大物体区域：覆盖95%以上的图像区域
    main_object = (X > -0.95) & (X < 0.95) & (Y > -0.90) & (Y < 0.90)
    
    # 创建更大的椭圆形状
    # 使用更大的椭圆，几乎覆盖整个图像
    ellipse_mask = ((X/0.98)**2 + (Y/0.95)**2) < 1.0
    
    # 结合矩形和椭圆，创建最大的投影区域
    combined_mask = main_object | ellipse_mask
    
    # 添加轻微的边缘变化，但减少噪声
    mask_rough = combined_mask.astype(np.float32)
    mask_smooth = ndimage.gaussian_filter(mask_rough, sigma=8)
    
    # 进一步减少随机噪声
    noise = np.random.normal(0, 0.01, (height, width))  # 从0.02进一步减少到0.01
    mask_with_noise = mask_smooth + noise
    
    # 创建最终的掩膜，进一步调整阈值以保证最大面积覆盖
    mask = mask_with_noise > 0.1  # 从0.3进一步降低到0.1，最大化投影区域
    
    # 创建更平缓的深度图
    # 减少深度变化，使条纹更加清晰
    depth_base = 0.1 + 0.2 * X  # 减小倾斜程度，从0.3+0.4*X改为0.1+0.2*X
    
    # 大幅减少局部高度变化
    depth_variation = 0.03 * np.sin(2 * X) * np.cos(1.5 * Y)  # 从0.1减少到0.03
    
    # 组合深度信息
    depth_map = depth_base + depth_variation
    
    # 只在物体区域保留深度信息
    depth_map = depth_map * mask
    
    # 更强的平滑处理，减少深度变化对条纹的影响
    depth_map = ndimage.gaussian_filter(depth_map, sigma=8)  # 从5增加到8
    
    return mask, depth_map

def create_realistic_background(width, height):
    """
    创建更干净的背景，减少噪声
    
    参数:
        width: 图像宽度
        height: 图像高度
        
    返回:
        background: 背景图像
    """
    # 基础背景 - 深色但不是纯黑，进一步减少随机变化
    base_level = 10 + np.random.normal(0, 1, (height, width))  # 从12+2进一步减少到10+1
    
    # 添加轻微的照明不均匀性
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # 进一步减少环境光照的渐变强度
    illumination = 3 * np.exp(-(X**2 + Y**2) / 4)  # 从5减少到3
    
    # 进一步减少背景噪声
    noise = np.random.normal(0, 1.5, (height, width))  # 从3减少到1.5
    
    # 组合背景
    background = base_level + illumination + noise
    
    # 确保值在合理范围内，保持背景更暗
    background = np.clip(background, 0, 25)  # 从35进一步减少到25
    
    return background

def validate_and_adjust_frequency(frequency, width, height, direction='horizontal'):
    """
    验证和调整频率，确保不会出现采样问题
    
    参数:
        frequency: 原始频率
        width: 图像宽度
        height: 图像高度
        direction: 条纹方向
        
    返回:
        adjusted_frequency: 调整后的频率
        warning_message: 警告信息（如果有调整）
    """
    # 计算奈奎斯特频率限制
    if direction == 'horizontal':
        # 水平条纹，受图像高度限制
        max_safe_frequency = height / 4  # 保证至少4个像素采样一个周期
        dimension = height
    else:
        # 垂直条纹，受图像宽度限制
        max_safe_frequency = width / 4   # 保证至少4个像素采样一个周期
        dimension = width
    
    warning_message = ""
    adjusted_frequency = frequency
    
    if frequency > max_safe_frequency:
        adjusted_frequency = max_safe_frequency
        warning_message = f"频率 {frequency} 过高，已调整为 {adjusted_frequency:.1f} (最大安全频率，基于{direction}方向{dimension}像素)"
    
    return adjusted_frequency, warning_message

def generate_realistic_fringe_image(width, height, frequency, phase_shift, direction='horizontal'):
    """
    生成真实的相移条纹图像，包含物体形状、背景和真实的光照效果
    
    参数:
        width: 图像宽度
        height: 图像高度
        frequency: 条纹频率（表示在图像尺寸范围内的条纹周期数）
        phase_shift: 相位偏移，单位为弧度
        direction: 条纹方向，'horizontal'或'vertical'
        
    返回:
        fringe_image: 生成的真实条纹图像，灰度值范围为[0, 255]
    """
    # 验证和调整频率
    adjusted_frequency, warning_msg = validate_and_adjust_frequency(frequency, width, height, direction)
    if warning_msg:
        print(f"  警告: {warning_msg}")
    
    # 使用调整后的频率
    frequency = adjusted_frequency
    # 创建物体掩膜和深度图
    object_mask, depth_map = create_realistic_object_mask(width, height)
    
    # 创建背景
    background = create_realistic_background(width, height)
    
    # 创建坐标网格 - 修正频率计算方式
    if direction == 'horizontal':
        # 水平条纹，相位沿Y方向变化
        # 频率表示在图像高度范围内的条纹周期数
        x = np.linspace(0, 1, width)
        y = np.linspace(0, frequency, height)  # 修正：Y方向范围为[0, frequency]
        X, Y = np.meshgrid(x, y)
        
        # 基础相位计算
        base_phase = 2 * np.pi * Y + phase_shift
        # 进一步减小深度引起的相位调制，使条纹更清晰
        phase_modulation = 2 * np.pi * depth_map * 0.08  # 从0.2进一步减少到0.08
        phase = base_phase + phase_modulation
    else:
        # 垂直条纹，相位沿X方向变化
        # 频率表示在图像宽度范围内的条纹周期数
        x = np.linspace(0, frequency, width)  # 修正：X方向范围为[0, frequency]
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # 基础相位计算
        base_phase = 2 * np.pi * X + phase_shift
        # 进一步减小深度引起的相位调制，使条纹更清晰
        phase_modulation = 2 * np.pi * depth_map * 0.08  # 从0.2进一步减少到0.08
        phase = base_phase + phase_modulation
    
    # 生成基础条纹图案
    fringe_pattern = 0.5 + 0.5 * np.cos(phase)
    
    # 模拟真实的投影亮度，提高基础亮度和对比度
    # 投影区域的基础亮度 - 增加亮度以提高条纹可见性
    projected_intensity = 140 + 110 * fringe_pattern  # 从120+100改为140+110
    
    # 进一步减少光照不均匀性，使条纹更加清晰
    illumination_variation = 4 * np.sin(np.pi * X) * np.sin(np.pi * Y)  # 从8进一步减少到4
    projected_intensity += illumination_variation
    
    # 进一步减少投影噪声，使画面更干净
    projection_noise = np.random.normal(0, 1.5, (height, width))  # 从3进一步减少到1.5
    projected_intensity += projection_noise
    
    # 创建最终图像
    final_image = np.zeros((height, width))
    
    # 在物体区域应用投影条纹
    final_image[object_mask] = projected_intensity[object_mask]
    
    # 在背景区域应用背景
    final_image[~object_mask] = background[~object_mask]
    
    # 添加更轻微的边缘光晕效果
    edge_kernel = np.ones((9, 9)) / 81  # 进一步增大核尺寸，使光晕更平滑
    edge_effect = cv.filter2D(object_mask.astype(np.float32), -1, edge_kernel)
    edge_glow = (edge_effect - object_mask.astype(np.float32)) * 8  # 从15进一步减少到8，减少光晕强度
    final_image += edge_glow
    
    # 确保值在[0, 255]范围内
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)
    
    return final_image

def generate_fringe_image(width, height, frequency, phase_shift, direction='horizontal'):
    """
    保持原有接口的包装函数
    """
    return generate_realistic_fringe_image(width, height, frequency, phase_shift, direction)

def main():
    """
    生成真实的三频相移条纹图像用于测试
    
    基于实际照片特征生成24张图像:
    - 3个频率（高、中、低）
    - 每个频率4张相移图像（相位偏移为0°, 90°, 180°, 270°）
    - 2个方向（水平条纹和垂直条纹）
    - 包含真实的物体形状、背景和光照效果
    
    注意：图像顺序与get_abs_phase.py中的处理顺序匹配
    - 图像1-12: 水平条纹图像（用于解算垂直方向相位）
    - 图像13-24: 垂直条纹图像（用于解算水平方向相位）
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成真实的三频相移条纹图像')
    parser.add_argument('--output_dir', type=str, default='../test_images',
                       help='输出图像的文件夹路径')
    parser.add_argument('--width', type=int, default=800,
                       help='图像宽度')
    parser.add_argument('--height', type=int, default=600,
                       help='图像高度')
    parser.add_argument('--show_preview', action='store_true',
                       help='是否显示预览图像')
    args = parser.parse_args()
    
    # 创建输出文件夹
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 设置频率参数（从高到低），基于实际应用的合理频率
    frequencies = [81, 72, 64]  # 调整为更适合掩膜测试的频率
    
    # 设置相移参数（4步相移）
    phase_shifts = [0, np.pi/2, np.pi, 3*np.pi/2]  # 0°, 90°, 180°, 270°
    
    print("=== 生成真实的三频相移条纹图像 ===")
    print(f"输出目录: {args.output_dir}")
    print(f"图像尺寸: {args.width} x {args.height}")
    print(f"原始频率设置: {frequencies}")
    
    # 计算和显示频率限制信息
    max_h_freq = args.height / 4  # 水平条纹的最大安全频率
    max_v_freq = args.width / 4   # 垂直条纹的最大安全频率
    print(f"最大安全频率: 水平条纹 {max_h_freq:.1f}, 垂直条纹 {max_v_freq:.1f}")
    print("注意: 超过最大安全频率的设置将被自动调整以避免采样问题")
    print()
    
    # 生成水平方向条纹图像（1-12）- 用于解算垂直方向相位
    print("正在生成水平方向条纹图像（用于解算垂直方向相位）...")
    image_index = 1
    
    for freq_idx, freq in enumerate(frequencies):
        freq_name = ["高频", "中频", "低频"][freq_idx]
        print(f"  生成{freq_name}水平条纹 (频率: {freq})...")
        
        for phase_idx, phase in enumerate(phase_shifts):
            # 生成水平条纹图像（相位沿Y方向变化）
            image = generate_fringe_image(args.width, args.height, freq, phase, 'horizontal')
            
            # 保存图像
            filename = os.path.join(args.output_dir, f"{image_index}.png")
            cv.imwrite(filename, image)
            print(f"    已保存: {image_index}.png - 相移: {phase/np.pi*180:.0f}°")
            
            # 预览图像（只显示每个频率的第一张图像）
            if args.show_preview and phase_idx == 0:
                plt.figure(figsize=(10, 8))
                plt.imshow(image, cmap='gray', vmin=0, vmax=255)
                plt.title(f"水平条纹{freq_name} - 频率: {freq}, 相移: {phase/np.pi*180:.0f}°")
                plt.colorbar()
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            
            image_index += 1
    
    # 生成垂直方向条纹图像（13-24）- 用于解算水平方向相位
    print("正在生成垂直方向条纹图像（用于解算水平方向相位）...")
    
    for freq_idx, freq in enumerate(frequencies):
        freq_name = ["高频", "中频", "低频"][freq_idx]
        print(f"  生成{freq_name}垂直条纹 (频率: {freq})...")
        
        for phase_idx, phase in enumerate(phase_shifts):
            # 生成垂直条纹图像（相位沿X方向变化）
            image = generate_fringe_image(args.width, args.height, freq, phase, 'vertical')
            
            # 保存图像
            filename = os.path.join(args.output_dir, f"{image_index}.png")
            cv.imwrite(filename, image)
            print(f"    已保存: {image_index}.png - 相移: {phase/np.pi*180:.0f}°")
            
            # 预览图像（只显示每个频率的第一张图像）
            if args.show_preview and phase_idx == 0:
                plt.figure(figsize=(10, 8))
                plt.imshow(image, cmap='gray', vmin=0, vmax=255)
                plt.title(f"垂直条纹{freq_name} - 频率: {freq}, 相移: {phase/np.pi*180:.0f}°")
                plt.colorbar()
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            
            image_index += 1
    
    print()
    print("=== 生成完成 ===")
    print(f"共生成 {image_index-1} 张真实条纹图像")
    print(f"保存位置: {os.path.abspath(args.output_dir)}")
    print()
    print("图像序列说明:")
    print("1-4:   水平条纹高频相移图像（用于解算垂直方向相位）")
    print("5-8:   水平条纹中频相移图像（用于解算垂直方向相位）")
    print("9-12:  水平条纹低频相移图像（用于解算垂直方向相位）")
    print("13-16: 垂直条纹高频相移图像（用于解算水平方向相位）")
    print("17-20: 垂直条纹中频相移图像（用于解算水平方向相位）")
    print("21-24: 垂直条纹低频相移图像（用于解算水平方向相位）")
    print()
    print("特征说明:")
    print("- 包含真实的物体形状和深度变化")
    print("- 具有黑色背景和投影区域的明显对比")
    print("- 模拟了真实的光照不均匀性和噪声")
    print("- 条纹会根据物体表面形状产生相应变形")
    print("- 适合测试掩膜生成和相位解包裹功能")

if __name__ == "__main__":
    main() 