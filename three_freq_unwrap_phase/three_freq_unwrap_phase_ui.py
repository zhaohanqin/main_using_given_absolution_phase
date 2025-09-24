#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
三频外差法相位解包裹程序UI界面

基于PySide6构建的界面，用于便捷地进行三频外差法相位解包裹操作。
支持水平和垂直方向的相位解包裹，基于get_abs_phase.py中的算法。
"""

import sys
import os
import numpy as np
import cv2 as cv
from typing import List, Optional, Tuple, Dict
from enum import Enum
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob # Added for folder scanning
import re # Added for folder scanning
import traceback # Added for improved error handling

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QGroupBox, 
    QRadioButton, QButtonGroup, QMessageBox, QProgressBar,
    QScrollArea, QSplitter, QFrame, QTabWidget,
    QSpinBox, QDoubleSpinBox, QFormLayout, QCheckBox, QComboBox
)
from PySide6.QtGui import QPixmap, QImage, QColor, QPalette, QFont, QPainter, QPen
from PySide6.QtCore import Qt, Signal, Slot, QThread

# 导入三频相位解包裹模块
from get_abs_phase import multi_phase, generate_projection_mask_three_freq, PhaseShiftingAlgorithm


class UnwrapDirection(Enum):
    """解包裹方向枚举"""
    HORIZONTAL = 0    # 水平方向
    VERTICAL = 1      # 垂直方向
    BOTH = 2          # 两个方向


class UnwrappingWorker(QThread):
    """相位解包裹处理线程"""
    # 定义信号
    progress_updated = Signal(int)
    processing_done = Signal(dict)
    error_occurred = Signal(str)
    
    def __init__(self, 
                output_dir: str = "output",
                freq_data: Optional[Dict[int, Dict[str, List[str]]]] = None,
                phase_step: int = 4,
                ph0: float = 0.5,
                filter_kernel_size: int = 9,
                unwrap_direction: UnwrapDirection = UnwrapDirection.BOTH,
                use_mask: bool = True,
                mask_method: str = "otsu",
                mask_confidence: float = 0.5):
        super().__init__()
        self.output_dir = output_dir
        self.freq_data = freq_data  # 格式: {freq: {'h': [paths], 'v': [paths]}}
        self.phase_step = phase_step
        self.ph0 = ph0
        self.filter_kernel_size = filter_kernel_size
        self.unwrap_direction = unwrap_direction
        self.use_mask = use_mask
        self.mask_method = mask_method
        self.mask_confidence = mask_confidence
        
    def run(self):
        try:
            result = {}
            os.makedirs(self.output_dir, exist_ok=True)
            
            self.progress_updated.emit(10)
            if self.freq_data is None:
                raise ValueError("未提供频率图像数据")
            
            # 获取三个频率
            frequencies = sorted(self.freq_data.keys())
            if len(frequencies) != 3:
                raise ValueError(f"三频法需要3个频率的图像，当前提供了{len(frequencies)}个")
                
            # 根据解包裹方向确定需要处理的图像
            process_horizontal = self.unwrap_direction in [UnwrapDirection.HORIZONTAL, UnwrapDirection.BOTH]
            process_vertical = self.unwrap_direction in [UnwrapDirection.VERTICAL, UnwrapDirection.BOTH]
            
            # 加载图像
            horizontal_images = []
            vertical_images = []
            combined_images = []  # 用于存储按照特定顺序的所有图像
            
            # 为了排错，记录图像信息
            image_info = []
            
            if process_horizontal:
                for freq in frequencies:
                    if 'h' not in self.freq_data[freq] or not self.freq_data[freq]['h']:
                        raise ValueError(f"频率 {freq} 缺少水平方向图像")
                    for i, path in enumerate(self.freq_data[freq]['h']):
                        # 增加路径检查
                        if not os.path.exists(path):
                            raise ValueError(f"图像文件不存在: {path}")
                        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
                        if img is None or img.size == 0:
                            raise ValueError(f"无法读取图像或图像为空: {path}")
                        image_info.append(f"水平-频率{freq}-图像{i+1}: 形状={img.shape}, 类型={img.dtype}")
                        horizontal_images.append(img.copy())  # 使用copy()确保数据独立
            
            if process_vertical:
                for freq in frequencies:
                    if 'v' not in self.freq_data[freq] or not self.freq_data[freq]['v']:
                        raise ValueError(f"频率 {freq} 缺少垂直方向图像")
                    for i, path in enumerate(self.freq_data[freq]['v']):
                        # 增加路径检查
                        if not os.path.exists(path):
                            raise ValueError(f"图像文件不存在: {path}")
                        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
                        if img is None or img.size == 0:
                            raise ValueError(f"无法读取图像或图像为空: {path}")
                        image_info.append(f"垂直-频率{freq}-图像{i+1}: 形状={img.shape}, 类型={img.dtype}")
                        vertical_images.append(img.copy())  # 使用copy()确保数据独立
            
            # 检查加载的图像数量
            n_freq = len(frequencies)
            
            if process_horizontal and len(horizontal_images) != n_freq * self.phase_step:
                raise ValueError(f"水平方向图像数量不正确，应为{n_freq * self.phase_step}张，实际为{len(horizontal_images)}张")
                
            if process_vertical and len(vertical_images) != n_freq * self.phase_step:
                raise ValueError(f"垂直方向图像数量不正确，应为{n_freq * self.phase_step}张，实际为{len(vertical_images)}张")
            
            # 组织图像数组：
            # 顺序：垂直高频(4张) + 垂直中频(4张) + 垂直低频(4张) + 水平高频(4张) + 水平中频(4张) + 水平低频(4张)
            # 注意：frequencies已按从高到低排序
            
            # 对于三频法，需要构建24张图像数组，按特定顺序排列
            # 准备处理时，先保证数组初始化为空
            combined_images = []
            
            # 首先添加垂直方向的图像（如果有）
            if process_vertical:
                for freq in frequencies:  # 按频率顺序添加
                    for img_path in self.freq_data[freq]['v']:
                        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
                        if img is None or img.size == 0:
                            raise ValueError(f"处理垂直图像时发现空图像: {img_path}")
                        combined_images.append(img)
            else:
                # 如果不处理垂直方向，添加12个空白图像占位
                dummy_img = np.zeros((600, 800), dtype=np.uint8)  # 创建一个适当大小的空白图像
                for _ in range(3 * self.phase_step):
                    combined_images.append(dummy_img.copy())
            
            # 然后添加水平方向的图像（如果有）
            if process_horizontal:
                for freq in frequencies:  # 按频率顺序添加
                    for img_path in self.freq_data[freq]['h']:
                        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
                        if img is None or img.size == 0:
                            raise ValueError(f"处理水平图像时发现空图像: {img_path}")
                        combined_images.append(img)
            else:
                # 如果不处理水平方向，添加12个空白图像占位
                dummy_img = np.zeros((600, 800), dtype=np.uint8)  # 创建一个适当大小的空白图像
                for _ in range(3 * self.phase_step):
                    combined_images.append(dummy_img.copy())
            
            # 验证图像数量
            if len(combined_images) != 6 * self.phase_step:
                raise ValueError(f"组合后的图像数量不正确，应为{6 * self.phase_step}张，实际为{len(combined_images)}张")
            
            # 保存图像信息日志，便于排错
            with open(os.path.join(self.output_dir, "image_info.log"), "w") as f:
                f.write("\n".join(image_info))
            
            self.progress_updated.emit(30)
            
            # 创建三频相位解包裹对象并处理
            unwarp_phase_y = None
            unwarp_phase_x = None
            ratio_y = None
            ratio_x = None
            wrapped_phase_y = None
            wrapped_phase_x = None
            
            # 创建处理对象，正确传递图像数组
            try:
                processor = multi_phase(
                    f=frequencies, 
                    step=self.phase_step, 
                    images=combined_images, 
                    ph0=self.ph0,
                    use_mask=self.use_mask,
                    mask_method=self.mask_method,
                    mask_confidence=self.mask_confidence,
                    output_dir=self.output_dir
                )
            
                # 调用get_phase()方法获取解包裹结果
                self.progress_updated.emit(50)
                unwarp_phase_y, unwarp_phase_x, ratio, wrapped_phase_y, wrapped_phase_x = processor.get_phase()
                
                # 检查输出结果
                if process_vertical and (unwarp_phase_y is None or unwarp_phase_y.size == 0):
                    raise ValueError("垂直方向相位解包裹结果为空")
                    
                if process_horizontal and (unwarp_phase_x is None or unwarp_phase_x.size == 0):
                    raise ValueError("水平方向相位解包裹结果为空")
                    
                if ratio is None or ratio.size == 0:
                    raise ValueError("相位质量图结果为空")
                    
                # 如果只处理一个方向，将另一个方向的结果置为None
                if not process_vertical:
                    unwarp_phase_y = None
                    ratio_y = None
                    wrapped_phase_y = None
                else:
                    ratio_y = ratio
                    
                if not process_horizontal:
                    unwarp_phase_x = None
                    ratio_x = None
                    wrapped_phase_x = None
                else:
                    ratio_x = ratio
                    
            except Exception as e:
                import traceback  # 确保在此作用域内导入traceback
                traceback.print_exc()
                raise ValueError(f"相位解包裹处理失败: {str(e)}")
            
            plt.close('all')  # 关闭所有中间过程的图像窗口
            plt.ion()  # 重新开启交互模式
            
            self.progress_updated.emit(80)
            
            # 保存掩膜（如果使用了掩膜）
            if self.use_mask and hasattr(processor, 'mask') and processor.mask is not None:
                try:
                    mask_dir = os.path.join(self.output_dir, "mask")
                    os.makedirs(mask_dir, exist_ok=True)
                    
                    # 保存最终掩膜
                    mask_img = (processor.mask.astype(np.uint8) * 255)
                    cv.imwrite(os.path.join(mask_dir, "final_mask.png"), mask_img)
                    
                    # 保存掩膜特征图（如果有的话）
                    # 这里可以保存振幅、调制度等特征图，用于调试
                    print(f"掩膜已保存至: {mask_dir}")
                except Exception as e:
                    print(f"保存掩膜时出错: {e}")
            
            # 保存结果
            if process_vertical and unwarp_phase_y is not None and unwarp_phase_y.size > 0 and ratio_y is not None and ratio_y.size > 0:
                try:
                    cv.imwrite(os.path.join(self.output_dir, "unwrapped_phase_vertical.tiff"), unwarp_phase_y)
                    cv.imwrite(os.path.join(self.output_dir, "phase_quality_vertical.tiff"), ratio_y)
                    
                    # 保存2D伪彩色图像
                    plt.figure(figsize=(10, 8))
                    plt.imshow(unwarp_phase_y, cmap='jet')
                    plt.colorbar(label='相位值')
                    plt.title('垂直方向解包裹相位2D视图')
                    plt.savefig(os.path.join(self.output_dir, "unwrapped_phase_vertical_2d.png"))
                    plt.close()
                    
                    # 保存包裹相位的2D伪彩色图像
                    if wrapped_phase_y is not None and wrapped_phase_y.size > 0:
                        plt.figure(figsize=(10, 8))
                        plt.imshow(wrapped_phase_y, cmap='jet')
                        plt.colorbar(label='相位值')
                        plt.title('垂直方向包裹相位2D视图')
                        plt.savefig(os.path.join(self.output_dir, "wrapped_phase_vertical_2d.png"))
                        plt.close()
                        
                        # 仅保存包裹相位数据，不含其他元素
                        wrapped_phase_y_normalized = cv.normalize(wrapped_phase_y, None, 0, 1, cv.NORM_MINMAX)
                        plt.figure(figsize=(10, 8), frameon=False)
                        plt.imshow(wrapped_phase_y_normalized, cmap='jet')
                        plt.axis('off')
                        plt.savefig(os.path.join(self.output_dir, "wrapped_phase_vertical_only.png"), 
                                   bbox_inches='tight', pad_inches=0)
                        plt.close()
                except Exception as e:
                    self.progress_updated.emit(90)
                    import traceback  # 确保在此作用域内导入traceback
                    traceback.print_exc()
                    raise ValueError(f"保存垂直方向相位图失败: {str(e)}")
            
            if process_horizontal and unwarp_phase_x is not None and unwarp_phase_x.size > 0 and ratio_x is not None and ratio_x.size > 0:
                try:
                    cv.imwrite(os.path.join(self.output_dir, "unwrapped_phase_horizontal.tiff"), unwarp_phase_x)
                    cv.imwrite(os.path.join(self.output_dir, "phase_quality_horizontal.tiff"), ratio_x)
                    
                    # 保存2D伪彩色图像
                    plt.figure(figsize=(10, 8))
                    plt.imshow(unwarp_phase_x, cmap='jet')
                    plt.colorbar(label='相位值')
                    plt.title('水平方向解包裹相位2D视图')
                    plt.savefig(os.path.join(self.output_dir, "unwrapped_phase_horizontal_2d.png"))
                    plt.close()
                    
                    # 保存包裹相位的2D伪彩色图像
                    if wrapped_phase_x is not None and wrapped_phase_x.size > 0:
                        plt.figure(figsize=(10, 8))
                        plt.imshow(wrapped_phase_x, cmap='jet')
                        plt.colorbar(label='相位值')
                        plt.title('水平方向包裹相位2D视图')
                        plt.savefig(os.path.join(self.output_dir, "wrapped_phase_horizontal_2d.png"))
                        plt.close()
                        
                        # 仅保存包裹相位数据，不含其他元素
                        wrapped_phase_x_normalized = cv.normalize(wrapped_phase_x, None, 0, 1, cv.NORM_MINMAX)
                        plt.figure(figsize=(10, 8), frameon=False)
                        plt.imshow(wrapped_phase_x_normalized, cmap='jet')
                        plt.axis('off')
                        plt.savefig(os.path.join(self.output_dir, "wrapped_phase_horizontal_only.png"), 
                                   bbox_inches='tight', pad_inches=0)
                        plt.close()
                except Exception as e:
                    self.progress_updated.emit(90)
                    import traceback  # 确保在此作用域内导入traceback
                    traceback.print_exc()
                    raise ValueError(f"保存水平方向相位图失败: {str(e)}")
                    
            # 如果两个方向都处理完成，保存组合相位图
            if process_horizontal and process_vertical and unwarp_phase_x is not None and unwarp_phase_y is not None:
                try:
                    # 创建组合伪彩色图像 (H -> Red, V -> Green)
                    h_norm = cv.normalize(unwarp_phase_x, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
                    v_norm = cv.normalize(unwarp_phase_y, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
                    
                    # 创建一个RGB图像
                    combined_img = np.zeros((*unwarp_phase_x.shape, 3), dtype=np.uint8)
                    combined_img[:,:,0] = h_norm  # 红色通道 - 水平
                    combined_img[:,:,1] = v_norm  # 绿色通道 - 垂直
                    
                    # 保存组合图像
                    cv.imwrite(os.path.join(self.output_dir, "combined_phase_map.png"), combined_img)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"保存组合相位图失败: {str(e)}")  # 不中断处理，只打印错误
            
            # 创建水平方向的3D视图
            if process_horizontal and unwarp_phase_x is not None and unwarp_phase_x.size > 0:
                try:
                    plt.figure(figsize=(10, 8))
                    ax = plt.axes(projection='3d')
                    h, w = unwarp_phase_x.shape
                    X, Y = np.meshgrid(range(w), range(h))
                    stride = 10  # 控制网格密度
                    ax.plot_surface(X[::stride, ::stride], Y[::stride, ::stride], 
                                    unwarp_phase_x[::stride, ::stride], cmap='jet')
                    ax.set_title('水平方向解包裹相位3D视图')
                    plt.savefig(os.path.join(self.output_dir, "unwrapped_phase_horizontal_3d.png"))
                    plt.close()
                except Exception as e:
                    import traceback  # 确保在此作用域内导入traceback
                    traceback.print_exc()
                    print(f"创建水平方向3D视图失败: {str(e)}")  # 不中断处理，只打印错误
            
            # 创建垂直方向的3D视图
            if process_vertical and unwarp_phase_y is not None and unwarp_phase_y.size > 0:
                try:
                    plt.figure(figsize=(10, 8))
                    ax = plt.axes(projection='3d')
                    h, w = unwarp_phase_y.shape
                    X, Y = np.meshgrid(range(w), range(h))
                    stride = 10  # 控制网格密度
                    ax.plot_surface(X[::stride, ::stride], Y[::stride, ::stride], 
                                    unwarp_phase_y[::stride, ::stride], cmap='jet')
                    ax.set_title('垂直方向解包裹相位3D视图')
                    plt.savefig(os.path.join(self.output_dir, "unwrapped_phase_vertical_3d.png"))
                    plt.close()
                except Exception as e:
                    import traceback  # 确保在此作用域内导入traceback
                    traceback.print_exc()
                    print(f"创建垂直方向3D视图失败: {str(e)}")  # 不中断处理，只打印错误
            
            self.progress_updated.emit(100)
            
            # 准备返回结果
            result = {}
            
            if process_horizontal and unwarp_phase_x is not None and unwarp_phase_x.size > 0 and ratio_x is not None and ratio_x.size > 0:
                result["horizontal"] = {
                    "unwrapped_phase": unwarp_phase_x,
                    "wrapped_phase": wrapped_phase_x,
                    "output_dir": self.output_dir,
                    "quality_map": ratio_x
                }
            
            if process_vertical and unwarp_phase_y is not None and unwarp_phase_y.size > 0 and ratio_y is not None and ratio_y.size > 0:
                result["vertical"] = {
                    "unwrapped_phase": unwarp_phase_y,
                    "wrapped_phase": wrapped_phase_y,
                    "output_dir": self.output_dir,
                    "quality_map": ratio_y
                }
            
            if not result:
                self.error_occurred.emit("处理完成，但未能生成有效的相位数据。请检查输入图像。")
                return
            
            self.processing_done.emit(result)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))


class InteractiveImageLabel(QLabel):
    """一个带有十字线交互的图像标签"""
    mouse_moved = Signal(object) # 发出鼠标移动事件
    mouse_left = Signal()      # 发出鼠标离开事件

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.crosshair_pos = None

    def mouseMoveEvent(self, event):
        self.crosshair_pos = event.pos()
        self.mouse_moved.emit(event)
        self.update() 

    def leaveEvent(self, event):
        self.crosshair_pos = None
        self.mouse_left.emit()
        self.update() 

    def paintEvent(self, event):
        super().paintEvent(event)
        
        pixmap = self.pixmap()
        if pixmap and not pixmap.isNull() and self.crosshair_pos:
            label_size = self.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            pixmap_w, pixmap_h = scaled_pixmap.width(), scaled_pixmap.height()
            offset_x = (label_size.width() - pixmap_w) / 2
            offset_y = (label_size.height() - pixmap_h) / 2
            
            x = self.crosshair_pos.x()
            y = self.crosshair_pos.y()

            if offset_x <= x < offset_x + pixmap_w and offset_y <= y < offset_y + pixmap_h:
                painter = QPainter(self)
                pen = QPen(QColor(220, 220, 220, 180)) 
                pen.setStyle(Qt.DashLine)
                pen.setWidth(1)
                painter.setPen(pen)
                painter.drawLine(int(offset_x), y, int(offset_x + pixmap_w), y)
                painter.drawLine(x, int(offset_y), x, int(offset_y + pixmap_h))


class PhaseImageViewer(QWidget):
    """相位图像查看器组件"""
    mouse_hover_info = Signal(str)
    
    def __init__(self, title: str = "图像查看器"):
        super().__init__()
        self.title = title
        self.phase_data = None
        self.quality_map = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        title_label = QLabel(self.title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)
        
        self.image_label = InteractiveImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ddd;")
        
        self.image_label.mouse_moved.connect(self._on_mouse_move)
        self.image_label.mouse_left.connect(self._on_mouse_leave)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.image_label)
        layout.addWidget(scroll)
        self.setLayout(layout)
    
    def set_image(self, image_path: str):
        if not os.path.exists(image_path):
            self.image_label.setText(f"图像不存在: {image_path}")
            return
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.image_label.setText(f"无法加载图像: {image_path}")
            return
        pixmap = pixmap.scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)

    def set_interactive_image(self, phase_data: np.ndarray, quality_map: Optional[np.ndarray]):
        self.phase_data = phase_data
        self.quality_map = quality_map
        self.set_numpy_image(phase_data)

    def _on_mouse_leave(self):
        self.mouse_hover_info.emit("")

    def _on_mouse_move(self, event):
        if self.phase_data is None: return
        pixmap = self.image_label.pixmap()
        if pixmap is None or pixmap.isNull(): return
        
        label_pos = event.pos()
        label_w, label_h = self.image_label.width(), self.image_label.height()
        pixmap_w, pixmap_h = pixmap.width(), pixmap.height()
        pixmap_x_offset = (label_w - pixmap_w) / 2
        pixmap_y_offset = (label_h - pixmap_h) / 2
        pixmap_x = label_pos.x() - pixmap_x_offset
        pixmap_y = label_pos.y() - pixmap_y_offset

        if not (0 <= pixmap_x < pixmap_w and 0 <= pixmap_y < pixmap_h):
            self.mouse_hover_info.emit("")
            return

        data_h, data_w = self.phase_data.shape
        data_ix = int(pixmap_x * data_w / pixmap_w)
        data_iy = int(pixmap_y * data_h / pixmap_h)

        if not (0 <= data_ix < data_w and 0 <= data_iy < data_h):
            self.mouse_hover_info.emit("")
            return

        phase_value = self.phase_data[data_iy, data_ix]
        period_value = phase_value / (2 * np.pi)  # 计算周期值
        
        if self.quality_map is not None:
            quality_value = self.quality_map[data_iy, data_ix]
            info_str = f"坐标: ({data_ix}, {data_iy})   相位值: {phase_value:.4f} rad   周期值: {period_value:.4f}   质量: {quality_value:.4f}"
        else:
            info_str = f"坐标: ({data_ix}, {data_iy})   相位值: {phase_value:.4f} rad   周期值: {period_value:.4f}"
        
        self.mouse_hover_info.emit(info_str)

    def set_numpy_image(self, image: np.ndarray, colormap=cv.COLORMAP_JET):
        if image is None:
            self.image_label.setText("图像数据为空")
            return
        
        if image.size == 0:
            self.image_label.setText("图像数据大小为0")
            return
        
        try:
            # 确保正确处理相位数据
                # 相位图往往是浮点数，需要先归一化到0-255区间
            if image.dtype != np.uint8:
                # 将相位数据缩放到0-255区间
                if np.max(image) > np.min(image):
                    img_normalized = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
                else:
                    # 如果所有像素值相同，避免除零问题
                    img_normalized = np.zeros_like(image, dtype=np.uint8)
                img_normalized = img_normalized.astype(np.uint8)
            else:
                img_normalized = image
            
            # 应用彩色映射，以可视化相位值
            if len(img_normalized.shape) == 2:
                img_color = cv.applyColorMap(img_normalized, colormap)
                img_color = cv.cvtColor(img_color, cv.COLOR_BGR2RGB)
            else:
                img_color = cv.cvtColor(img_normalized, cv.COLOR_BGR2RGB)
            
            height, width, channel = img_color.shape
            bytes_per_line = channel * width
            q_image = QImage(img_color.data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(q_image)
            pixmap = pixmap.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)
        except Exception as e:
            import traceback  # 确保在此作用域内导入traceback
            traceback.print_exc()
            self.image_label.setText(f"图像处理错误: {str(e)}")


class PhaseViewerContainer(QWidget):
    """相位查看器容器，包含2D和3D视图"""
    def __init__(self, title: str = "相位数据"):
        super().__init__()
        self.title = title
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        title_label = QLabel(self.title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)
        
        self.tab_widget = QTabWidget()
        self.viewer_2d = PhaseImageViewer("2D 视图")
        self.tab_widget.addTab(self.viewer_2d, "2D 视图")
        self.viewer_3d = PhaseImageViewer("3D 视图")
        self.tab_widget.addTab(self.viewer_3d, "3D 视图")
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
    
    def set_interactive_2d_image(self, phase_data: np.ndarray, quality_map: Optional[np.ndarray]):
        self.viewer_2d.set_interactive_image(phase_data, quality_map)

    def set_3d_image(self, image_path: str):
        if os.path.exists(image_path):
            self.viewer_3d.set_image(image_path)
            self.tab_widget.setTabVisible(1, True)
        else:
            self.tab_widget.setTabVisible(1, False)
    
    def reset(self):
        self.viewer_2d.image_label.setText("暂无图像")
        self.viewer_3d.image_label.setText("暂无图像")
        self.tab_widget.setTabVisible(1, False)


class InteractiveCombinedPhaseViewer(QLabel):
    """
    一个可交互的组合相位图像查看器。
    - 支持鼠标悬停显示十字准星。
    - 实时显示鼠标位置的坐标和相位值。
    - 存储原始相位数据以便精确查找。
    """
    # 信号：当鼠标移动并需要更新外部信息标签时发出
    info_updated = Signal(str)
    
    def __init__(self, title: str = "图像查看器"):
        super().__init__()
        self.title = title
        self.phase_data_h = None  # 水平方向的原始相位数据
        self.phase_data_v = None  # 垂直方向的原始相位数据
        self.wrapped_phase_data_h = None # 水平包裹相位
        self.wrapped_phase_data_v = None # 垂直包裹相位
        self.pixmap = None
        self.mouse_pos = None
        self.ph0 = 0.5  # 添加ph0属性并设置默认值
        
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ddd;")
        
        self.setText(f"{self.title}\n\n(暂无图像)")

    def mouseMoveEvent(self, event):
        """处理鼠标移动事件，更新位置并触发重绘。"""
        self.mouse_pos = event.pos()
        self.update()

    def set_phase_data(self, h_phase, v_phase, h_wrapped, v_wrapped, ph0):
        """设置并显示相位数据。"""
        self.phase_data_h = h_phase
        self.phase_data_v = v_phase
        self.wrapped_phase_data_h = h_wrapped
        self.wrapped_phase_data_v = v_wrapped
        self.ph0 = ph0  # 保存ph0
        
        if self.phase_data_h is not None and self.phase_data_v is not None:
            # 创建一个组合的伪彩色图像 (H -> Red, V -> Green)
            h_norm = cv.normalize(self.phase_data_h, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
            v_norm = cv.normalize(self.phase_data_v, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
            
            # 创建一个RGB图像
            img_color = np.zeros((*self.phase_data_h.shape, 3), dtype=np.uint8)
            img_color[:,:,0] = h_norm
            img_color[:,:,1] = v_norm
            
            height, width, channel = img_color.shape
            bytes_per_line = channel * width
            q_image = QImage(img_color.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(q_image)
            self.update()
        else:
            self.reset()


    def paintEvent(self, event):
        """重写绘制事件以添加十字准星和文本。"""
        super().paintEvent(event)
        if not self.pixmap:
            self.setText(f"{self.title}\n\n(暂无图像)")
            return

        painter = QPainter(self)
        
        # 计算图像在Label中的实际显示区域（保持宽高比）
        label_size = self.size()
        scaled_pixmap = self.pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x_offset = (label_size.width() - scaled_pixmap.width()) / 2
        y_offset = (label_size.height() - scaled_pixmap.height()) / 2
        
        # 绘制图像
        painter.drawPixmap(int(x_offset), int(y_offset), scaled_pixmap)
        
        # 绘制标题
        if self.title:
            painter.setPen(QColor(0, 0, 0))
            painter.setFont(QFont("Arial", 12, QFont.Bold))
            painter.drawText(self.rect(), Qt.AlignHCenter | Qt.AlignTop, self.title)

        # 当鼠标离开时，mouse_pos为None，不执行后续操作
        if self.mouse_pos is None:
            return

        # 检查鼠标是否在图像区域内
        if not (x_offset <= self.mouse_pos.x() < x_offset + scaled_pixmap.width() and
                y_offset <= self.mouse_pos.y() < y_offset + scaled_pixmap.height()):
            self.info_updated.emit("")
            return
            
        # 绘制十字准星
        pen = QPen(QColor(220, 220, 220, 180)) 
        pen.setStyle(Qt.DashLine)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(int(x_offset), self.mouse_pos.y(), int(x_offset + scaled_pixmap.width()), self.mouse_pos.y())
        painter.drawLine(self.mouse_pos.x(), int(y_offset), self.mouse_pos.x(), int(y_offset + scaled_pixmap.height()))

        # 将Label坐标转换为图像坐标
        img_x = self.mouse_pos.x() - x_offset
        img_y = self.mouse_pos.y() - y_offset
        
        # 将缩放后的图像坐标转换回原始图像坐标
        orig_x = int(img_x * (self.pixmap.width() / scaled_pixmap.width()))
        orig_y = int(img_y * (self.pixmap.height() / scaled_pixmap.height()))
            
        # 确保坐标在原始数据范围内
        if not (0 <= orig_x < self.phase_data_h.shape[1] and 0 <= orig_y < self.phase_data_h.shape[0]):
             self.info_updated.emit("")
             return

        info_text = f"坐标: ({orig_x}, {orig_y})"
        
        if self.phase_data_h is not None and self.phase_data_v is not None:
            phase_h = self.phase_data_h[orig_y, orig_x]
            phase_v = self.phase_data_v[orig_y, orig_x]
            info_text += f" | 水平相位: {phase_h:.3f}"
            info_text += f" | 垂直相位: {phase_v:.3f}"

            if self.wrapped_phase_data_h is not None and self.wrapped_phase_data_v is not None:
                # 注意：包裹相位已经归一化到[0,1]，需要转换回[-pi, pi]来计算条纹序数
                wrapped_h = (self.wrapped_phase_data_h[orig_y, orig_x] + self.ph0) * 2 * np.pi - np.pi
                wrapped_v = (self.wrapped_phase_data_v[orig_y, orig_x] + self.ph0) * 2 * np.pi - np.pi
                # 绝对相位也需要转换
                abs_phase_h = phase_h * 2 * np.pi
                abs_phase_v = phase_v * 2 * np.pi
                
                k_h = np.round((abs_phase_h - wrapped_h) / (2 * np.pi))
                k_v = np.round((abs_phase_v - wrapped_v) / (2 * np.pi))
                
                # 确保周期值为非负数
                k_h = int(abs(k_h))
                k_v = int(abs(k_v))
                info_text += f" | 周期(H): {k_h}, (V): {k_v}"
        
        self.info_updated.emit(info_text)

    def leaveEvent(self, event):
        """处理鼠标离开事件。"""
        self.mouse_pos = None
        self.info_updated.emit("") # 清空信息
        self.update()
    
    def reset(self):
        """重置视图。"""
        self.phase_data_h = None
        self.phase_data_v = None
        self.wrapped_phase_data_h = None
        self.wrapped_phase_data_v = None
        self.pixmap = None
        self.mouse_pos = None
        self.setText(f"{self.title}\n\n(暂无图像)")
        self.update()


class CombinedViewerWindow(QWidget):
    """一个用于显示交互式组合相位图的新窗口"""
    def __init__(self, h_phase, v_phase, h_wrapped, v_wrapped, ph0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("组合相位图 (H-红, V-绿)")
        self.setMinimumSize(600, 500)

        layout = QVBoxLayout(self)

        viewer = InteractiveCombinedPhaseViewer("组合相位图")
        viewer.set_phase_data(h_phase, v_phase, h_wrapped, v_wrapped, ph0)
        
        info_label = QLabel("将鼠标悬停在图像上以查看详细信息")
        info_label.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)

        viewer.info_updated.connect(lambda text: info_label.setText(text or "将鼠标悬停在图像上以查看详细信息"))

        layout.addWidget(viewer, 1)
        layout.addWidget(info_label, 0)

        self.setAttribute(Qt.WA_DeleteOnClose)


class ThreeFreqPhaseUnwrapperUI(QMainWindow):
    """三频外差法相位解包裹程序主界面"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("三频外差法相位解包裹程序")
        self.setMinimumSize(1200, 800)
        
        self.freq_widgets = []  # 存储频率相关的UI组件
        self.unwrap_direction = UnwrapDirection.BOTH
        self.output_dir = "three_freq_phase_unwrap_results"
        self.n_steps = 4  # 4步相移
        self.frequencies = [64, 56, 49]  # 默认频率值
        self.ph0 = 0.5  # 初始相位偏移
        
        # 掩膜相关参数
        self.use_mask = True
        self.mask_method = "otsu"
        self.mask_confidence = 0.5
        self.permanent_status_message = "就绪"
        self.combined_viewer_window = None # 用于持有对新窗口的引用
        
        self.set_application_style()
        self.init_ui()
    
    def set_application_style(self):
        QApplication.setStyle("Fusion")
        style_sheet = """
        QMainWindow, QWidget { background-color: #f7f7f7; }
        QPushButton {
            background-color: #d5e8f8; border: 1px solid #a0c0e0;
            border-radius: 4px; padding: 6px 12px;
            color: #2c3e50; font-weight: bold;
        }
        QPushButton:hover { background-color: #bbd5f1; }
        QPushButton:pressed { background-color: #a0c0e0; }
        QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
            border: 1px solid #a0c0e0; border-radius: 4px;
            padding: 4px; background-color: white;
        }
        QGroupBox {
            border: 1px solid #a0c0e0; border-radius: 6px;
            margin-top: 12px; font-weight: bold; color: #2c3e50;
        }
        QGroupBox::title {
            subcontrol-origin: margin; subcontrol-position: top center;
            padding: 0 5px; background-color: #f7f7f7;
        }
        QRadioButton, QLabel { color: #2c3e50; }
        QProgressBar {
            border: 1px solid #a0c0e0; border-radius: 4px;
            text-align: center; background-color: white;
        }
        QProgressBar::chunk { background-color: #3498db; width: 1px; }
        """
        QApplication.instance().setStyleSheet(style_sheet)
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        control_panel = self.create_control_panel()
        main_layout.addLayout(control_panel)
        
        status_panel = self.create_status_panel()
        main_layout.addLayout(status_panel)
        
        image_display = self.create_image_display()
        main_layout.addWidget(image_display, 1)
    
    def create_control_panel(self):
        control_layout = QHBoxLayout()
        control_layout.setSpacing(20)

        # --- 左侧：参数设置 ---
        settings_widget = self.create_settings_panel()
        control_layout.addWidget(settings_widget, 2)

        # --- 右侧：操作和方向 ---
        right_panel_widget = self.create_operation_panel()
        control_layout.addWidget(right_panel_widget, 1)
        
        return control_layout

    def create_settings_panel(self):
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.setContentsMargins(0, 0, 0, 0)

        general_group = QGroupBox("参数设置")
        general_form_layout = QFormLayout(general_group)
        
        # 相移步数
        self.n_steps_spinbox = QSpinBox()
        self.n_steps_spinbox.setRange(3, 20)
        self.n_steps_spinbox.setValue(self.n_steps)
        self.n_steps_spinbox.setToolTip("设置相移的步数(N)。")
        self.n_steps_spinbox.valueChanged.connect(self.update_n_steps)
        general_form_layout.addRow("相移步数 (N):", self.n_steps_spinbox)
        
        # 初始相位偏移
        self.ph0_spinbox = QDoubleSpinBox()
        self.ph0_spinbox.setRange(0.0, 1.0)
        self.ph0_spinbox.setSingleStep(0.1)
        self.ph0_spinbox.setValue(self.ph0)
        self.ph0_spinbox.setToolTip("设置初始相位偏移量。")
        self.ph0_spinbox.valueChanged.connect(lambda v: setattr(self, 'ph0', v))
        general_form_layout.addRow("初始相位偏移:", self.ph0_spinbox)
        
        settings_layout.addWidget(general_group)
        
        # 频率设置
        freq_group = QGroupBox("频率设置")
        freq_form_layout = QVBoxLayout(freq_group)
        
        # 高频设置
        high_freq_group = QGroupBox("高频 (F1)")
        high_freq_layout = QHBoxLayout(high_freq_group)
        self.freq1_spinbox = QSpinBox()
        self.freq1_spinbox.setRange(1, 200)
        self.freq1_spinbox.setValue(self.frequencies[0])
        self.freq1_spinbox.setToolTip("设置高频值。")
        self.freq1_spinbox.valueChanged.connect(lambda v: self.update_frequency(0, v))
        high_freq_layout.addWidget(QLabel("频率:"))
        high_freq_layout.addWidget(self.freq1_spinbox)
        self.freq1_folder_btn = QPushButton("选择文件夹 (0)")
        self.freq1_folder_btn.setToolTip("选择包含高频相移图像的文件夹。")
        self.freq1_folder_btn.clicked.connect(lambda: self.select_freq_folder(0))
        high_freq_layout.addWidget(self.freq1_folder_btn, 1)
        freq_form_layout.addWidget(high_freq_group)
        
        # 中频设置
        mid_freq_group = QGroupBox("中频 (F2)")
        mid_freq_layout = QHBoxLayout(mid_freq_group)
        self.freq2_spinbox = QSpinBox()
        self.freq2_spinbox.setRange(1, 200)
        self.freq2_spinbox.setValue(self.frequencies[1])
        self.freq2_spinbox.setToolTip("设置中频值。")
        self.freq2_spinbox.valueChanged.connect(lambda v: self.update_frequency(1, v))
        mid_freq_layout.addWidget(QLabel("频率:"))
        mid_freq_layout.addWidget(self.freq2_spinbox)
        self.freq2_folder_btn = QPushButton("选择文件夹 (0)")
        self.freq2_folder_btn.setToolTip("选择包含中频相移图像的文件夹。")
        self.freq2_folder_btn.clicked.connect(lambda: self.select_freq_folder(1))
        mid_freq_layout.addWidget(self.freq2_folder_btn, 1)
        freq_form_layout.addWidget(mid_freq_group)
        
        # 低频设置
        low_freq_group = QGroupBox("低频 (F3)")
        low_freq_layout = QHBoxLayout(low_freq_group)
        self.freq3_spinbox = QSpinBox()
        self.freq3_spinbox.setRange(1, 200)
        self.freq3_spinbox.setValue(self.frequencies[2])
        self.freq3_spinbox.setToolTip("设置低频值。")
        self.freq3_spinbox.valueChanged.connect(lambda v: self.update_frequency(2, v))
        low_freq_layout.addWidget(QLabel("频率:"))
        low_freq_layout.addWidget(self.freq3_spinbox)
        self.freq3_folder_btn = QPushButton("选择文件夹 (0)")
        self.freq3_folder_btn.setToolTip("选择包含低频相移图像的文件夹。")
        self.freq3_folder_btn.clicked.connect(lambda: self.select_freq_folder(2))
        low_freq_layout.addWidget(self.freq3_folder_btn, 1)
        freq_form_layout.addWidget(low_freq_group)
        
        settings_layout.addWidget(freq_group)
        
        # 滤波设置
        filter_group = QGroupBox("滤波设置")
        filter_form_layout = QFormLayout(filter_group)
        self.filter_size_spinbox = QSpinBox()
        self.filter_size_spinbox.setRange(3, 21)
        self.filter_size_spinbox.setSingleStep(2)
        self.filter_size_spinbox.setValue(9)
        self.filter_size_spinbox.setToolTip("设置用于平滑数据的滤波器大小。")
        filter_form_layout.addRow("滤波器尺寸:", self.filter_size_spinbox)
        settings_layout.addWidget(filter_group)
        
        # 掩膜设置
        mask_group = QGroupBox("投影区域掩膜设置")
        mask_form_layout = QFormLayout(mask_group)
        
        # 是否使用掩膜
        self.use_mask_checkbox = QCheckBox("启用投影区域掩膜")
        self.use_mask_checkbox.setChecked(self.use_mask)
        self.use_mask_checkbox.setToolTip("启用后，只在投影有效区域内进行相位解包裹计算，避免环境干扰。")
        self.use_mask_checkbox.stateChanged.connect(self.update_use_mask)
        mask_form_layout.addRow(self.use_mask_checkbox)
        
        # 掩膜生成方法
        self.mask_method_combo = QComboBox()
        self.mask_method_combo.addItem("Otsu 自适应阈值 (推荐)", "otsu")
        self.mask_method_combo.addItem("自适应阈值", "adaptive")
        self.mask_method_combo.addItem("相对百分位阈值", "relative")
        self.mask_method_combo.setCurrentIndex(0)  # 默认选择otsu
        self.mask_method_combo.currentIndexChanged.connect(self.update_mask_method)
        self.mask_method_combo.setToolTip("选择用于生成投影区域掩膜的方法：\n"
                                        "• Otsu 自适应阈值 (推荐): 基于Otsu算法，稳定可靠\n"
                                        "• 自适应阈值: 结合多特征的智能阈值化\n"
                                        "• 相对百分位阈值: 基于百分位的相对阈值方法")
        mask_form_layout.addRow("掩膜生成方法:", self.mask_method_combo)
        
        # 掩膜置信度
        self.mask_confidence_spinbox = QDoubleSpinBox()
        self.mask_confidence_spinbox.setRange(0.1, 0.9)
        self.mask_confidence_spinbox.setSingleStep(0.1)
        self.mask_confidence_spinbox.setDecimals(1)
        self.mask_confidence_spinbox.setValue(self.mask_confidence)
        self.mask_confidence_spinbox.valueChanged.connect(self.update_mask_confidence)
        self.mask_confidence_spinbox.setToolTip("输入 0.1-0.9 的数值以设置掩膜置信度\n\n"
                                              "• 自适应方法：结合振幅、调制度、相位稳定性等多特征\n"
                                              "• Otsu方法：置信度对此方法影响较小\n"
                                              "• 相对阈值方法：基于百分位的阈值选择")
        mask_form_layout.addRow("掩膜置信度:", self.mask_confidence_spinbox)
        
        # 置信度说明标签
        self.mask_confidence_info_label = QLabel("推荐范围: 0.4-0.6 (平衡掩膜质量)")
        self.mask_confidence_info_label.setStyleSheet("color: #666; font-size: 11px; font-style: italic;")
        mask_form_layout.addRow(self.mask_confidence_info_label)
        
        settings_layout.addWidget(mask_group)
        
        # 初始化频率组件数据
        self.freq_widgets = [
            {"spinbox": self.freq1_spinbox, "btn": self.freq1_folder_btn, "h_paths": [], "v_paths": []},
            {"spinbox": self.freq2_spinbox, "btn": self.freq2_folder_btn, "h_paths": [], "v_paths": []},
            {"spinbox": self.freq3_spinbox, "btn": self.freq3_folder_btn, "h_paths": [], "v_paths": []}
        ]
        
        # 初始化掩膜信息显示
        self.update_mask_confidence_info()
        
        settings_layout.addStretch()
        return settings_widget

    def create_operation_panel(self):
        right_panel_widget = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_widget)
        right_panel_layout.setContentsMargins(0, 0, 0, 0)

        operation_group = QGroupBox("操作")
        operation_layout = QVBoxLayout(operation_group)
        operation_layout.setSpacing(15)
        start_btn = QPushButton("开始处理")
        start_btn.setToolTip("根据当前设置开始执行相位解包裹。")
        start_btn.setMinimumHeight(40)
        start_btn.clicked.connect(self.start_processing)
        view_results_btn = QPushButton("查看结果文件夹")
        view_results_btn.clicked.connect(self.open_result_folder)
        reset_btn = QPushButton("重置")
        reset_btn.clicked.connect(self.reset_ui)
        operation_layout.addWidget(start_btn)
        operation_layout.addWidget(view_results_btn)
        operation_layout.addWidget(reset_btn)

        self.direction_group = QGroupBox("解包裹方向")
        direction_layout = QVBoxLayout(self.direction_group)
        self.horizontal_radio = QRadioButton("仅水平方向")
        self.vertical_radio = QRadioButton("仅垂直方向")
        self.both_radio = QRadioButton("两个方向")
        self.both_radio.setChecked(True)
        direction_layout.addWidget(self.horizontal_radio)
        direction_layout.addWidget(self.vertical_radio)
        direction_layout.addWidget(self.both_radio)
        self.direction_button_group = QButtonGroup(self)
        self.direction_button_group.addButton(self.horizontal_radio, 0)
        self.direction_button_group.addButton(self.vertical_radio, 1)
        self.direction_button_group.addButton(self.both_radio, 2)
        self.direction_button_group.idClicked.connect(self.update_unwrap_direction)

        output_group = QGroupBox("输出设置")
        output_layout = QHBoxLayout(output_group)
        self.output_dir_label = QLabel(self.output_dir)
        self.output_dir_label.setWordWrap(True)
        select_output_dir_btn = QPushButton("选择...")
        select_output_dir_btn.clicked.connect(self.select_output_dir)
        output_layout.addWidget(self.output_dir_label)
        output_layout.addWidget(select_output_dir_btn)

        right_panel_layout.addWidget(operation_group)
        right_panel_layout.addWidget(self.direction_group)
        right_panel_layout.addWidget(output_group)
        right_panel_layout.addStretch()
        return right_panel_widget
    
    def create_status_panel(self):
        status_layout = QHBoxLayout()
        self.status_label = QLabel(self.permanent_status_message)
        status_layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)
        return status_layout
    
    def create_image_display(self):
        splitter = QSplitter(Qt.Horizontal)
        self.horizontal_viewer = PhaseViewerContainer("水平方向解包裹相位")
        self.horizontal_viewer.viewer_2d.mouse_hover_info.connect(self._update_hover_info)
        splitter.addWidget(self.horizontal_viewer)
        
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        splitter.addWidget(line)
        
        self.vertical_viewer = PhaseViewerContainer("垂直方向解包裹相位")
        self.vertical_viewer.viewer_2d.mouse_hover_info.connect(self._update_hover_info)
        splitter.addWidget(self.vertical_viewer)
        
        splitter.setSizes([500, 10, 500])
        return splitter

    @Slot(int)
    def update_n_steps(self, value: int):
        if self.n_steps != value:
            self.n_steps = value
            self.reset_ui()
            QMessageBox.information(self, "提示", f"相移步数已更新为 {self.n_steps}。\n图像选择已重置，请重新加载。")
    
    def update_frequency(self, index: int, value: int):
        self.frequencies[index] = value
    
    @Slot(int)
    def update_unwrap_direction(self, direction_id: int):
        self.unwrap_direction = UnwrapDirection(direction_id)
    
    @Slot(int)
    def update_use_mask(self, state: int):
        """更新是否使用掩膜"""
        self.use_mask = state == 2  # Qt.Checked = 2
        
        # 启用/禁用掩膜相关的控件
        self.mask_method_combo.setEnabled(self.use_mask)
        self.mask_confidence_spinbox.setEnabled(self.use_mask)
        self.update_mask_confidence_info()
    
    @Slot(int)
    def update_mask_method(self, index: int):
        """更新掩膜方法"""
        self.mask_method = self.mask_method_combo.itemData(index)
        self.update_mask_confidence_info()
    
    @Slot(float)
    def update_mask_confidence(self, value: float):
        """更新掩膜置信度"""
        self.mask_confidence = value
        self.update_mask_confidence_info()
    
    def update_mask_confidence_info(self):
        """更新置信度说明信息"""
        if not self.use_mask:
            self.mask_confidence_info_label.setText("掩膜功能已禁用")
            self.mask_confidence_info_label.setStyleSheet("color: #999; font-size: 11px; font-style: italic;")
            return
            
        confidence = self.mask_confidence
        method = self.mask_method
        
        # 根据不同方法提供不同的建议
        if method == "adaptive":
            if confidence < 0.4:
                info = f"当前值: {confidence:.1f} (自适应-宽松，保留更多区域但可能含噪声)"
                color = "#ff9500"  # 橙色
            elif confidence <= 0.6:
                info = f"当前值: {confidence:.1f} (自适应-推荐，智能多特征平衡)"
                color = "#51cf66"  # 绿色
            else:
                info = f"当前值: {confidence:.1f} (自适应-严格，仅保留高质量区域)"
                color = "#339af0"  # 蓝色
        elif method == "otsu":
            if confidence < 0.4:
                info = f"当前值: {confidence:.1f} (Otsu方法，置信度对此方法影响较小)"
                color = "#868e96"  # 灰色
            elif confidence <= 0.6:
                info = f"当前值: {confidence:.1f} (Otsu方法，传统自动阈值化)"
                color = "#51cf66"  # 绿色
            else:
                info = f"当前值: {confidence:.1f} (Otsu方法，置信度对此方法影响较小)"
                color = "#868e96"  # 灰色
        else:  # relative
            if confidence < 0.4:
                info = f"当前值: {confidence:.1f} (相对阈值-宽松，保留更多百分位)"
                color = "#ff6b6b"  # 红色
            elif confidence <= 0.6:
                info = f"当前值: {confidence:.1f} (相对阈值-推荐，平衡百分位选择)"
                color = "#51cf66"  # 绿色
            else:
                info = f"当前值: {confidence:.1f} (相对阈值-严格，仅保留高百分位)"
                color = "#ffd43b"  # 黄色
        
        self.mask_confidence_info_label.setText(info)
        self.mask_confidence_info_label.setStyleSheet(f"color: {color}; font-size: 11px; font-style: italic;")

    @Slot()
    def select_freq_folder(self, freq_index: int):
        folder = QFileDialog.getExistingDirectory(self, f"为频率 {freq_index+1} 选择图像文件夹", "")
        if not folder: return

        try:
            image_files = []
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
                image_files.extend(
                    glob.glob(os.path.join(folder, f"*{ext}"), recursive=False) + 
                    glob.glob(os.path.join(folder, f"*{ext.upper()}"), recursive=False)
                )
            image_files = sorted(list(set(image_files)))
            
            n = self.n_steps
            def get_img_num(path):
                match = re.search(r'[iI](\d+)\.', os.path.basename(path))
                return int(match.group(1)) if match else -1
            all_imgs_map = {get_img_num(p): p for p in image_files if get_img_num(p) != -1}

            h_paths = [all_imgs_map.get(i) for i in range(1, n + 1) if all_imgs_map.get(i)]
            v_paths = [all_imgs_map.get(i) for i in range(n + 1, 2 * n + 1) if all_imgs_map.get(i)]

            found_h = len(h_paths) == n
            found_v = len(v_paths) == n

            self.freq_widgets[freq_index]["h_paths"] = h_paths if found_h else []
            self.freq_widgets[freq_index]["v_paths"] = v_paths if found_v else []
            
            if found_h and found_v:
                btn_text, info_text = f"水平+垂直 ({2*n}张)", f"频率{freq_index+1}: 已加载水平和垂直图像。"
            elif found_h:
                btn_text, info_text = f"仅水平 ({n}张)", f"频率{freq_index+1}: 已加载水平图像。"
            elif found_v:
                btn_text, info_text = f"仅垂直 ({n}张)", f"频率{freq_index+1}: 已加载垂直图像。"
            else:
                btn_text, info_text = "选择文件夹 (0)", f"频率{freq_index+1}: 未找到符合命名规则(I1-I{2*n})的完整图像集。"
                QMessageBox.warning(self, "未找到图像", info_text)

            self.freq_widgets[freq_index]["btn"].setText(btn_text)
            self.status_label.setText(info_text)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图像时出错:\n{str(e)}")
            import traceback  # 确保在此作用域内导入traceback
            traceback.print_exc()
    
    @Slot()
    def select_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if folder:
            self.output_dir = folder
            self.output_dir_label.setText(self.output_dir)
    
    def start_processing(self):
        freq_data = {}
        
        # 构建处理数据
        for i, widget_set in enumerate(self.freq_widgets):
            freq = widget_set["spinbox"].value()
            data = {}
            
            if widget_set["h_paths"]:
                data["h"] = widget_set["h_paths"]
                
            if widget_set["v_paths"]:
                data["v"] = widget_set["v_paths"]
                
            if data:  # 如果有数据
                freq_data[freq] = data
        
        # 检查数据是否足够
        if len(freq_data) < 3:
            QMessageBox.warning(self, "数据不足", f"三频法需要3个频率的图像，当前只有 {len(freq_data)} 个频率。")
            return
        
        # 根据选择的方向确定处理模式
        process_horizontal = self.unwrap_direction in [UnwrapDirection.HORIZONTAL, UnwrapDirection.BOTH]
        process_vertical = self.unwrap_direction in [UnwrapDirection.VERTICAL, UnwrapDirection.BOTH]
        
        # 检查是否有足够的数据进行处理
        can_process_h = all('h' in freq_data[freq] for freq in freq_data)
        can_process_v = all('v' in freq_data[freq] for freq in freq_data)
        
        if process_horizontal and not can_process_h:
            QMessageBox.warning(self, "数据不足", "水平方向处理需要所有频率都有水平方向的图像。")
            return
            
        if process_vertical and not can_process_v:
            QMessageBox.warning(self, "数据不足", "垂直方向处理需要所有频率都有垂直方向的图像。")
            return
        
        # 准备处理参数
        worker_params = {
            "output_dir": self.output_dir,
            "freq_data": freq_data,
            "phase_step": self.n_steps,
            "ph0": self.ph0,
            "filter_kernel_size": self.filter_size_spinbox.value(),
            "unwrap_direction": self.unwrap_direction,
            "use_mask": self.use_mask,
            "mask_method": self.mask_method,
            "mask_confidence": self.mask_confidence
        }
        
        # 创建并启动处理线程
        self.worker = UnwrappingWorker(**worker_params)
        self.worker.processing_done.connect(self.handle_processing_finished)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.error_occurred.connect(self.handle_error)
        
        self.permanent_status_message = "正在处理..."
        self.status_label.setText(self.permanent_status_message)
        self.progress_bar.setValue(0)
        self.worker.start()
    
    @Slot(int)
    def update_progress(self, value: int):
        self.progress_bar.setValue(value)
    
    def handle_processing_finished(self, result: dict):
        self.permanent_status_message = "三频处理完成"
        self.status_label.setText(self.permanent_status_message)

        if "horizontal" in result:
            res_data = result["horizontal"]
            d3_path = os.path.join(self.output_dir, "unwrapped_phase_horizontal_3d.png")
            self.horizontal_viewer.set_interactive_2d_image(res_data["unwrapped_phase"], res_data.get("quality_map"))
            self.horizontal_viewer.set_3d_image(d3_path)
        else:
            self.horizontal_viewer.reset()

        if "vertical" in result:
            res_data = result["vertical"]
            d3_path = os.path.join(self.output_dir, "unwrapped_phase_vertical_3d.png")
            self.vertical_viewer.set_interactive_2d_image(res_data["unwrapped_phase"], res_data.get("quality_map"))
            self.vertical_viewer.set_3d_image(d3_path)
        else:
            self.vertical_viewer.reset()

        # 如果两个方向都处理完成，则在单独的窗口中显示组合图
        if "horizontal" in result and "vertical" in result:
            h_data = result.get("horizontal", {})
            v_data = result.get("vertical", {})
            h_phase = h_data.get("unwrapped_phase")
            v_phase = v_data.get("unwrapped_phase")
            h_wrapped = h_data.get("wrapped_phase")
            v_wrapped = v_data.get("wrapped_phase")

            if h_phase is not None and v_phase is not None:
                # 在新窗口中显示交互式2D组合图
                # 将窗口引用存储到self中，以防止它被垃圾回收
                self.combined_viewer_window = CombinedViewerWindow(
                    h_phase=h_phase,
                    v_phase=v_phase,
                    h_wrapped=h_wrapped,
                    v_wrapped=v_wrapped,
                    ph0=self.ph0  # 传递ph0
                )
                self.combined_viewer_window.show()

        QMessageBox.information(self, "成功", "三频相位解包裹处理完成")

    @Slot(str)
    def handle_error(self, error_msg: str):
        self.permanent_status_message = f"错误: {error_msg}"
        self.status_label.setText(self.permanent_status_message)
        QMessageBox.critical(self, "错误", f"处理过程中发生错误:\n{error_msg}")
    
    @Slot()
    def open_result_folder(self):
        if not os.path.exists(self.output_dir):
            QMessageBox.warning(self, "警告", "输出目录不存在")
            return
        import platform, subprocess
        if platform.system() == "Windows":
            os.startfile(self.output_dir)
        elif platform.system() == "Darwin":
            subprocess.call(["open", self.output_dir])
        else:
            subprocess.call(["xdg-open", self.output_dir])
    
    @Slot(str)
    def _update_hover_info(self, info: str):
        if info:
            self.status_label.setText(info)
        else:
            self.status_label.setText(self.permanent_status_message)

    @Slot()
    def reset_ui(self):
        for widget_set in self.freq_widgets:
            widget_set["h_paths"], widget_set["v_paths"] = [], []
            widget_set["btn"].setText("选择文件夹 (0)")
        self.horizontal_viewer.reset()
        self.vertical_viewer.reset()
        self.progress_bar.setValue(0)
        self.permanent_status_message = "就绪"
        self.status_label.setText(self.permanent_status_message)


def main():
    app = QApplication(sys.argv)
    window = ThreeFreqPhaseUnwrapperUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 