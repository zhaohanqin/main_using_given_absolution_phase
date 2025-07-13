import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import argparse

def generate_fringe_image(width, height, frequency, phase_shift, direction='horizontal'):
    """
    生成相移条纹图像
    
    参数:
        width: 图像宽度
        height: 图像高度
        frequency: 条纹频率
        phase_shift: 相位偏移，单位为弧度
        direction: 条纹方向，'horizontal'或'vertical'
        
    返回:
        fringe_image: 生成的条纹图像，灰度值范围为[0, 255]
    """
    # 创建坐标网格
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    
    # 根据条纹方向生成相应的条纹图像
    if direction == 'horizontal':
        # 水平条纹，相位沿Y方向变化
        phase = 2 * np.pi * frequency * Y / height + phase_shift
    else:
        # 垂直条纹，相位沿X方向变化
        phase = 2 * np.pi * frequency * X / width + phase_shift
    
    # 生成正弦条纹图像
    fringe = 0.5 + 0.5 * np.cos(phase)
    
    # 转换为8位灰度图像
    fringe_image = (fringe * 255).astype(np.uint8)
    
    return fringe_image

def main():
    """
    生成三频相移条纹图像用于测试
    
    生成24张图像:
    - 3个频率（高、中、低）
    - 每个频率4张相移图像（相位偏移为0°, 90°, 180°, 270°）
    - 2个方向（水平条纹和垂直条纹）
    
    注意：图像顺序与get_abs_phase.py中的处理顺序匹配
    - 图像1-12: 水平条纹图像（用于解算垂直方向相位）
    - 图像13-24: 垂直条纹图像（用于解算水平方向相位）
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成三频相移条纹图像')
    parser.add_argument('--output_dir', type=str, default='./test_images',
                       help='输出图像的文件夹路径')
    parser.add_argument('--width', type=int, default=1024,
                       help='图像宽度')
    parser.add_argument('--height', type=int, default=768,
                       help='图像高度')
    parser.add_argument('--show_preview', action='store_true',
                       help='是否显示预览图像')
    args = parser.parse_args()
    
    # 创建输出文件夹
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 设置频率参数（从高到低）
    frequencies = [71, 64, 58]  # 频率值
    
    # 设置相移参数（4步相移）
    phase_shifts = [0, np.pi/2, np.pi, 3*np.pi/2]  # 0°, 90°, 180°, 270°
    
    # 生成水平方向条纹图像（1-12）- 用于解算垂直方向相位
    print("正在生成水平方向条纹图像（用于解算垂直方向相位）...")
    image_index = 1
    
    for freq in frequencies:
        for phase in phase_shifts:
            # 生成水平条纹图像（相位沿Y方向变化）
            image = generate_fringe_image(args.width, args.height, freq, phase, 'horizontal')
            
            # 保存图像
            filename = os.path.join(args.output_dir, f"{image_index}.png")
            cv.imwrite(filename, image)
            print(f"已保存: {filename} - 水平条纹, 频率: {freq}, 相移: {phase/np.pi*180}°")
            
            # 预览图像
            if args.show_preview and image_index % 4 == 1:  # 只显示每个频率的第一张图像
                plt.figure(figsize=(8, 6))
                plt.imshow(image, cmap='gray')
                plt.title(f"水平条纹 - 频率: {freq}, 相移: {phase/np.pi*180}°")
                plt.colorbar()
                plt.show()
            
            image_index += 1
    
    # 生成垂直方向条纹图像（13-24）- 用于解算水平方向相位
    print("正在生成垂直方向条纹图像（用于解算水平方向相位）...")
    
    for freq in frequencies:
        for phase in phase_shifts:
            # 生成垂直条纹图像（相位沿X方向变化）
            image = generate_fringe_image(args.width, args.height, freq, phase, 'vertical')
            
            # 保存图像
            filename = os.path.join(args.output_dir, f"{image_index}.png")
            cv.imwrite(filename, image)
            print(f"已保存: {filename} - 垂直条纹, 频率: {freq}, 相移: {phase/np.pi*180}°")
            
            # 预览图像
            if args.show_preview and image_index % 4 == 1:  # 只显示每个频率的第一张图像
                plt.figure(figsize=(8, 6))
                plt.imshow(image, cmap='gray')
                plt.title(f"垂直条纹 - 频率: {freq}, 相移: {phase/np.pi*180}°")
                plt.colorbar()
                plt.show()
            
            image_index += 1
    
    print(f"生成完成！共生成 {image_index-1} 张条纹图像，保存在 {args.output_dir} 文件夹中")
    print("图像顺序:")
    print("1-4:   水平条纹高频相移图像（用于解算垂直方向相位）")
    print("5-8:   水平条纹中频相移图像（用于解算垂直方向相位）")
    print("9-12:  水平条纹低频相移图像（用于解算垂直方向相位）")
    print("13-16: 垂直条纹高频相移图像（用于解算水平方向相位）")
    print("17-20: 垂直条纹中频相移图像（用于解算水平方向相位）")
    print("21-24: 垂直条纹低频相移图像（用于解算水平方向相位）")

if __name__ == "__main__":
    main() 