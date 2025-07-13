import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from get_abs_phase import multi_phase
from read_image import read_img

def main():
    """
    使用三频外差法对结构光相移图像进行相位解包裹的测试程序
    
    该程序接收用户输入的图像文件夹路径，读取相移图像，进行相位解包裹，
    并将结果保存为相位图文件和可视化结果。
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='三频外差法相位解包裹测试程序')
    parser.add_argument('--input_dir', type=str, 
                        help='包含相移图像的文件夹路径')
    parser.add_argument('--output_dir', type=str,
                        help='结果输出文件夹路径')
    parser.add_argument('--show_results', action='store_true', 
                        help='是否显示结果图像')
    args = parser.parse_args()
    
    # 交互式获取参数（如果命令行参数未提供）
    if args.input_dir is None:
        args.input_dir = input("请输入包含相移图像的文件夹路径: ")
    
    if args.output_dir is None:
        default_output = './output'
        output_input = input(f"请输入结果输出文件夹路径 (直接回车使用默认路径 {default_output}): ")
        args.output_dir = output_input if output_input.strip() else default_output
    
    if not args.show_results:
        show_results_input = input("是否显示结果图像? (y/n): ").lower()
        args.show_results = show_results_input.startswith('y')
    
    # 创建输出文件夹
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 读取相移图像
    print(f"正在读取图像文件夹: {args.input_dir}")
    try:
        images = read_img(args.input_dir)
        print(f"成功读取 {len(images)} 张图像")
    except Exception as e:
        print(f"读取图像失败: {str(e)}")
        return
    
    # 检查图像数量是否符合要求
    if len(images) < 24:
        print(f"错误: 需要至少24张图像用于三频相位解包裹 (当前: {len(images)}张)")
        print("图像要求: 每个频率4张相移图像 × 3个频率 × 2个方向(水平和垂直) = 24张")
        return
    
    # 设置相位解包裹参数
    # 频率值从高到低排序
    fx = [64, 8, 1]  # 水平方向的三个频率
    fy = [64, 8, 1]  # 垂直方向的三个频率
    phase_step = 4   # 4步相移
    ph0 = 0.5        # 初始相位偏移
    
    print("正在进行相位解包裹...")
    
    # 创建多频相位解包裹对象并处理
    phase_processor = multi_phase(f=fx, step=phase_step, images=images, ph0=ph0)
    unwarp_phase_y, unwarp_phase_x, ratio = phase_processor.get_phase()
    
    print("相位解包裹完成")
    
    # 保存结果
    print(f"正在保存结果到: {args.output_dir}")
    cv.imwrite(os.path.join(args.output_dir, "unwrapped_phase_vertical.tiff"), unwarp_phase_y)
    cv.imwrite(os.path.join(args.output_dir, "unwrapped_phase_horizontal.tiff"), unwarp_phase_x)
    cv.imwrite(os.path.join(args.output_dir, "phase_quality.tiff"), ratio)
    
    # 可视化结果
    if args.show_results:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.title("垂直方向展开相位")
        plt.imshow(unwarp_phase_y, cmap='jet')
        plt.colorbar()
        
        plt.subplot(132)
        plt.title("水平方向展开相位")
        plt.imshow(unwarp_phase_x, cmap='jet')
        plt.colorbar()
        
        plt.subplot(133)
        plt.title("相位质量图")
        plt.imshow(ratio, cmap='viridis')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "phase_visualization.png"))
        plt.show()
    
    print("处理完成！")
    print(f"结果文件已保存至: {args.output_dir}")

if __name__ == "__main__":
    main() 