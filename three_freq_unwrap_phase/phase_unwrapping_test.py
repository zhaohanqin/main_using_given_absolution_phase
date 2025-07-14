import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
import argparse
import traceback
from get_abs_phase import multi_phase
from read_image import read_img
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QStatusBar, QSplitter
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont

# 设置matplotlib后端和字体
matplotlib.use('Qt5Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

class InteractivePhaseCanvas(FigureCanvas):
    """交互式相位图显示画布"""
    mouse_moved = pyqtSignal(float, float, float)  # 发送坐标和相位值
    
    def __init__(self, phase_data, title, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.phase_data = phase_data
        
        super(InteractivePhaseCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        # 显示相位图
        self.im = self.axes.imshow(self.phase_data, cmap='jet')
        self.axes.set_title(title)
        self.fig.colorbar(self.im, ax=self.axes, label='相位值')
        self.fig.tight_layout()
        
        # 连接鼠标事件
        self.mpl_connect('motion_notify_event', self.on_mouse_move)
    
    def on_mouse_move(self, event):
        if event.inaxes == self.axes:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < self.phase_data.shape[1] and 0 <= y < self.phase_data.shape[0]:
                phase_value = self.phase_data[y, x]
                self.mouse_moved.emit(x, y, phase_value)

class CombinedPhaseCanvas(FigureCanvas):
    """组合相位图显示画布"""
    mouse_moved = pyqtSignal(float, float, float, float)  # 发送坐标和两个方向的相位值
    
    def __init__(self, h_phase, v_phase, title, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.h_phase = h_phase
        self.v_phase = v_phase
        
        super(CombinedPhaseCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        # 生成组合相位图
        h_norm = (h_phase - np.min(h_phase)) / (np.max(h_phase) - np.min(h_phase))
        v_norm = (v_phase - np.min(v_phase)) / (np.max(v_phase) - np.min(v_phase))
        
        h, w = h_phase.shape
        combined_rgb = np.zeros((h, w, 3), dtype=np.float32)
        combined_rgb[:,:,0] = h_norm  # 红色通道为水平方向
        combined_rgb[:,:,1] = v_norm  # 绿色通道为垂直方向
        combined_rgb[:,:,2] = (h_norm + v_norm) / 2  # 蓝色通道为两者平均
        
        # 显示组合相位图
        self.im = self.axes.imshow(combined_rgb)
        self.axes.set_title(title)
        self.fig.colorbar(self.im, ax=self.axes, label='归一化相位值')
        self.fig.tight_layout()
        
        # 连接鼠标事件
        self.mpl_connect('motion_notify_event', self.on_mouse_move)
    
    def on_mouse_move(self, event):
        if event.inaxes == self.axes:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < self.h_phase.shape[1] and 0 <= y < self.h_phase.shape[0]:
                h_phase_value = self.h_phase[y, x]
                v_phase_value = self.v_phase[y, x]
                self.mouse_moved.emit(x, y, h_phase_value, v_phase_value)

class PhaseUnwrapViewer(QMainWindow):
    """相位解包裹结果查看器"""
    
    def __init__(self, unwrap_phase_y, unwrap_phase_x, ratio, output_dir):
        super().__init__()
        self.unwrap_phase_y = unwrap_phase_y
        self.unwrap_phase_x = unwrap_phase_x
        self.ratio = ratio
        self.output_dir = output_dir
        
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("相位解包裹结果查看器")
        self.setGeometry(100, 100, 1500, 600)  # 更宽的窗口以适应水平布局
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建水平分割器 (所有图像水平排列)
        splitter = QSplitter(Qt.Horizontal)
        
        # 创建左侧面板 - 水平方向相位
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_title = QLabel("水平方向展开相位")
        left_title.setAlignment(Qt.AlignCenter)
        left_title.setFont(QFont("Arial", 12, QFont.Bold))
        left_layout.addWidget(left_title)
        
        self.h_phase_canvas = InteractivePhaseCanvas(self.unwrap_phase_x, "水平方向展开相位")
        self.h_phase_canvas.mouse_moved.connect(self.update_h_status)
        left_layout.addWidget(self.h_phase_canvas)
        
        splitter.addWidget(left_widget)
        
        # 创建中间面板 - 垂直方向相位
        middle_widget = QWidget()
        middle_layout = QVBoxLayout(middle_widget)
        middle_title = QLabel("垂直方向展开相位")
        middle_title.setAlignment(Qt.AlignCenter)
        middle_title.setFont(QFont("Arial", 12, QFont.Bold))
        middle_layout.addWidget(middle_title)
        
        self.v_phase_canvas = InteractivePhaseCanvas(self.unwrap_phase_y, "垂直方向展开相位")
        self.v_phase_canvas.mouse_moved.connect(self.update_v_status)
        middle_layout.addWidget(self.v_phase_canvas)
        
        splitter.addWidget(middle_widget)
        
        # 创建右侧面板 - 组合相位图
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_title = QLabel("水平和垂直方向相位组合图")
        right_title.setAlignment(Qt.AlignCenter)
        right_title.setFont(QFont("Arial", 12, QFont.Bold))
        right_layout.addWidget(right_title)
        
        self.combined_canvas = CombinedPhaseCanvas(self.unwrap_phase_x, self.unwrap_phase_y, "水平和垂直方向相位组合图")
        self.combined_canvas.mouse_moved.connect(self.update_combined_status)
        right_layout.addWidget(self.combined_canvas)
        
        splitter.addWidget(right_widget)
        
        # 设置分割器的初始大小 (均匀分布)
        splitter.setSizes([500, 500, 500])
        
        # 添加分割器到主布局
        main_layout.addWidget(splitter)
        
        # 创建底部控制面板
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        
        # 创建保存按钮
        save_button = QPushButton("保存所有图像")
        save_button.setMinimumHeight(30)
        save_button.clicked.connect(self.save_images)
        control_layout.addWidget(save_button)
        
        # 添加控制面板到主布局
        main_layout.addWidget(control_panel)
        
        # 创建状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪")
    
    @pyqtSlot(float, float, float)
    def update_h_status(self, x, y, phase_value):
        """更新水平方向相位的状态信息"""
        period_value = phase_value / (2 * np.pi)
        self.statusBar.showMessage(f"水平方向 - 坐标: ({int(x)}, {int(y)})   相位值: {phase_value:.4f} rad   周期值: {period_value:.4f}")
    
    @pyqtSlot(float, float, float)
    def update_v_status(self, x, y, phase_value):
        """更新垂直方向相位的状态信息"""
        period_value = phase_value / (2 * np.pi)
        self.statusBar.showMessage(f"垂直方向 - 坐标: ({int(x)}, {int(y)})   相位值: {phase_value:.4f} rad   周期值: {period_value:.4f}")
    
    @pyqtSlot(float, float, float, float)
    def update_combined_status(self, x, y, h_phase_value, v_phase_value):
        """更新组合图的状态信息，显示两个方向的相位值"""
        h_period = h_phase_value / (2 * np.pi)
        v_period = v_phase_value / (2 * np.pi)
        self.statusBar.showMessage(
            f"坐标: ({int(x)}, {int(y)})   "
            f"水平相位: {h_phase_value:.4f} rad   水平周期: {h_period:.4f}   "
            f"垂直相位: {v_phase_value:.4f} rad   垂直周期: {v_period:.4f}"
        )
    
    def save_images(self):
        """保存所有图像"""
        try:
            # 确保输出目录存在
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 保存水平方向相位图
            self.h_phase_canvas.fig.savefig(os.path.join(self.output_dir, "unwrapped_phase_horizontal.png"), 
                                           dpi=300, bbox_inches='tight')
            
            # 保存垂直方向相位图
            self.v_phase_canvas.fig.savefig(os.path.join(self.output_dir, "unwrapped_phase_vertical.png"), 
                                           dpi=300, bbox_inches='tight')
            
            # 保存组合相位图
            self.combined_canvas.fig.savefig(os.path.join(self.output_dir, "combined_phase.png"), 
                                            dpi=300, bbox_inches='tight')
            
            # 保存原始相位数据
            cv.imwrite(os.path.join(self.output_dir, "unwrapped_phase_horizontal.tiff"), self.unwrap_phase_x)
            cv.imwrite(os.path.join(self.output_dir, "unwrapped_phase_vertical.tiff"), self.unwrap_phase_y)
            cv.imwrite(os.path.join(self.output_dir, "phase_quality.tiff"), self.ratio)
            
            self.statusBar.showMessage(f"所有图像已保存至: {self.output_dir}")
        except Exception as e:
            self.statusBar.showMessage(f"保存图像时出错: {str(e)}")
            traceback.print_exc()

def visualize_unwrapped_phase(unwrap_phase_y, unwrap_phase_x, ratio, output_dir):
    """
    可视化解包裹相位结果
    
    参数:
        unwrap_phase_y: 垂直方向解包裹相位
        unwrap_phase_x: 水平方向解包裹相位
        ratio: 相位质量图
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 显示2D相位图
    plt.figure(figsize=(15, 5))
    
    # 水平方向相位图
    plt.subplot(131)
    plt.imshow(unwrap_phase_x, cmap='jet')
    plt.colorbar(label='相位值')
    plt.title('水平方向展开相位')
    
    # 垂直方向相位图
    plt.subplot(132)
    plt.imshow(unwrap_phase_y, cmap='jet')
    plt.colorbar(label='相位值')
    plt.title('垂直方向展开相位')
    
    # 相位质量图
    plt.subplot(133)
    plt.imshow(ratio, cmap='viridis')
    plt.colorbar(label='质量值')
    plt.title('相位质量图')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phase_2d_view.png"), dpi=300, bbox_inches='tight')
    
    # 生成组合相位图 (使用unwrap_phase_example.py中的方法)
    # 归一化两个相位图
    h_norm = (unwrap_phase_x - np.min(unwrap_phase_x)) / (np.max(unwrap_phase_x) - np.min(unwrap_phase_x))
    v_norm = (unwrap_phase_y - np.min(unwrap_phase_y)) / (np.max(unwrap_phase_y) - np.min(unwrap_phase_y))
    
    # 组合两个方向的相位图得到伪彩色图像
    h, w = unwrap_phase_x.shape
    combined_rgb = np.zeros((h, w, 3), dtype=np.float32)
    combined_rgb[:,:,0] = h_norm  # 红色通道为水平方向
    combined_rgb[:,:,1] = v_norm  # 绿色通道为垂直方向
    combined_rgb[:,:,2] = (h_norm + v_norm) / 2  # 蓝色通道为两者平均
    
    plt.figure(figsize=(10, 8))
    plt.imshow(combined_rgb)
    plt.title('水平和垂直方向相位组合图')
    plt.colorbar(label='归一化相位值')
    plt.savefig(os.path.join(output_dir, "combined_phase.png"), dpi=300, bbox_inches='tight')
    
    # 显示所有图形
    plt.show()
    
    return combined_rgb

def show_interactive_viewer(unwrap_phase_y, unwrap_phase_x, ratio, output_dir):
    """显示交互式查看器"""
    app = QApplication([])
    viewer = PhaseUnwrapViewer(unwrap_phase_y, unwrap_phase_x, ratio, output_dir)
    viewer.show()
    app.exec_()

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
    parser.add_argument('--interactive', action='store_true',
                        help='是否使用交互式查看器')
    parser.add_argument('--hide_intermediate', action='store_true', 
                        help='是否隐藏中间过程图像，仅显示最终结果')
    args = parser.parse_args()
    
    # 交互式获取参数（如果命令行参数未提供）
    if args.input_dir is None:
        args.input_dir = input("请输入包含相移图像的文件夹路径: ")
    
    if args.output_dir is None:
        default_output = './output'
        output_input = input(f"请输入结果输出文件夹路径 (直接回车使用默认路径 {default_output}): ")
        args.output_dir = output_input if output_input.strip() else default_output
    
    if not args.show_results and not args.interactive:
        show_results_input = input("是否显示结果图像? (y/n): ").lower()
        args.show_results = show_results_input.startswith('y')
        
        if args.show_results:
            interactive_input = input("是否使用交互式查看器? (y/n): ").lower()
            args.interactive = interactive_input.startswith('y')
    
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
        traceback.print_exc()
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
    phase_step = 4     # 4步相移
    ph0 = 0.5          # 初始相位偏移
    
    print("正在进行相位解包裹...")
    
    # 创建多频相位解包裹对象并处理
    phase_processor = multi_phase(f=fx, step=phase_step, images=images, ph0=ph0)
    
    # 临时禁用matplotlib显示，避免中间过程显示图像
    if args.hide_intermediate:
        plt.ioff()  # 关闭交互模式
    
    # 执行相位解包裹
    unwrap_phase_y, unwrap_phase_x, ratio = phase_processor.get_phase()
    
    # 恢复matplotlib交互模式
    if args.hide_intermediate:
        plt.close('all')  # 关闭所有图形
        plt.ion()  # 重新开启交互模式
    
    print("相位解包裹完成")
    
    # 保存结果
    print(f"正在保存结果到: {args.output_dir}")
    cv.imwrite(os.path.join(args.output_dir, "unwrapped_phase_vertical.tiff"), unwrap_phase_y)
    cv.imwrite(os.path.join(args.output_dir, "unwrapped_phase_horizontal.tiff"), unwrap_phase_x)
    cv.imwrite(os.path.join(args.output_dir, "phase_quality.tiff"), ratio)
    
    # 可视化结果
    if args.interactive:
        print("显示交互式查看器...")
        show_interactive_viewer(unwrap_phase_y, unwrap_phase_x, ratio, args.output_dir)
    elif args.show_results:
        print("显示解包裹相位结果...")
        visualize_unwrapped_phase(unwrap_phase_y, unwrap_phase_x, ratio, args.output_dir)
    
    print("处理完成！")
    print(f"结果文件已保存至: {args.output_dir}")

if __name__ == "__main__":
    main() 