import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
from PIL import Image
import argparse
from get_abs_phase import multi_phase
from read_image import read_img

# 设置matplotlib支持中文显示 - 使用更可靠的方法
try:
    # 尝试使用SimHei字体
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=10)
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
except:
    try:
        # 备选方案1: 使用微软雅黑
        font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc", size=10)
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    except:
        try:
            # 备选方案2: 使用宋体
            font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=10)
            matplotlib.rcParams['font.sans-serif'] = ['SimSun']
        except:
            print("警告: 找不到中文字体，标题可能无法正确显示")
            font = None

# 解决保存图像时负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False

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
    fx = [71, 64, 58]  # 水平方向的三个频率
    fy = [71, 64, 58]  # 垂直方向的三个频率
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
        fig = plt.figure(figsize=(15, 5))
        
        ax1 = plt.subplot(131)
        plt.title("垂直方向展开相位", fontproperties=font)
        im1 = plt.imshow(unwarp_phase_y, cmap='jet')
        plt.colorbar()
        
        ax2 = plt.subplot(132)
        plt.title("水平方向展开相位", fontproperties=font)
        im2 = plt.imshow(unwarp_phase_x, cmap='jet')
        plt.colorbar()
        
        ax3 = plt.subplot(133)
        plt.title("相位质量图", fontproperties=font)
        im3 = plt.imshow(ratio, cmap='viridis')
        plt.colorbar()
        
        # 添加鼠标交互功能
        annot = ax1.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)
        
        # 为每个相位图创建一个鼠标悬停处理函数
        def update_annot(ax, im, event, phase_data, title):
            if event.inaxes == ax:
                annot.set_visible(True)
                # 获取像素坐标
                x, y = int(event.xdata), int(event.ydata)
                if 0 <= x < phase_data.shape[1] and 0 <= y < phase_data.shape[0]:
                    # 获取相位值
                    phase_value = phase_data[y, x]
                    # 计算周期值 (相位值除以2π)
                    period_value = phase_value / (2 * np.pi)
                    
                    annot.xy = (x, y)
                    annot.set_text(f"{title}\n位置: ({x}, {y})\n相位值: {phase_value:.2f}\n周期值: {period_value:.2f}")
                    annot.get_bbox_patch().set_alpha(0.9)
                    fig.canvas.draw_idle()
            else:
                annot.set_visible(False)
                fig.canvas.draw_idle()
                
        def hover(event):
            if event.inaxes == ax1:
                update_annot(ax1, im1, event, unwarp_phase_y, "垂直方向")
            elif event.inaxes == ax2:
                update_annot(ax2, im2, event, unwarp_phase_x, "水平方向")
            elif event.inaxes == ax3:
                update_annot(ax3, im3, event, ratio, "相位质量")
            else:
                annot.set_visible(False)
                fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect("motion_notify_event", hover)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "phase_visualization.png"))
        plt.show()
    
    print("处理完成！")
    print(f"结果文件已保存至: {args.output_dir}")

if __name__ == "__main__":
    main() 