#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于相位解包裹的投影仪标定系统 - 图形用户界面

该程序提供基于PySide6的图形用户界面，用于执行投影仪标定过程。
界面设计采用浅色主题，布局合理，操作直观。

作者: [Your Name]
日期: [Current Date]
"""

import os
import sys
import threading
from datetime import datetime
import numpy as np
import cv2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QTabWidget, QTextEdit, QGroupBox, QFormLayout, 
    QCheckBox, QMessageBox, QProgressBar, QScrollArea, QSplitter
)
from PySide6.QtGui import QPixmap, QImage, QColor, QPalette, QFont, QIcon
from PySide6.QtCore import Qt, Signal, Slot, QThread, QSize, QTimer

# 导入投影仪标定模块
try:
    import 其它文件.projector_calibration_phase as cal
except ImportError:
    print("Error: 无法导入投影仪标定模块。请确保 projector_calibration_phase_based.py 在当前目录中。")
    sys.exit(1)

# 配色方案
COLORS = {
    "primary": "#4a6fa5",      # 主色调（蓝色）
    "secondary": "#f8f9fa",    # 次要色调（浅灰色）
    "accent": "#6c757d",       # 强调色（灰色）
    "success": "#28a745",      # 成功色（绿色）
    "warning": "#ffc107",      # 警告色（黄色）
    "danger": "#dc3545",       # 危险色（红色）
    "background": "#ffffff",   # 背景色（白色）
    "text": "#343a40"          # 文字色（深灰色）
}

class LogRedirector:
    """重定向标准输出到QTextEdit的类"""
    
    def __init__(self, text_edit):
        self.text_edit = text_edit
        self.buffer = ""
        
    def write(self, text):
        self.buffer += text
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            for line in lines[:-1]:
                self.text_edit.append(line)
            self.buffer = lines[-1]
            
    def flush(self):
        if self.buffer:
            self.text_edit.append(self.buffer)
            self.buffer = ""

class CalibrationThread(QThread):
    """执行标定过程的线程类"""
    
    # 定义信号
    progress_update = Signal(int)
    status_update = Signal(str)
    calibration_complete = Signal(dict)
    calibration_error = Signal(str)
    image_update = Signal(str, QPixmap)
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        
    def run(self):
        try:
            # 重定向标准输出到我们的信号
            # 保存原始print函数，以防重入
            if not hasattr(cal, 'original_print'):
                cal.original_print = print
            
            def custom_print(*args, **kwargs):
                message = " ".join(map(str, args))
                self.status_update.emit(message)
                # 不再调用原始print，避免在GUI日志和控制台重复输出
                # original_print(*args, **kwargs)
            
            # 将cal模块的print重定向到我们的信号发射函数
            cal.print = custom_print
            
            # 发送开始信息
            self.status_update.emit("开始标定过程...")
            self.progress_update.emit(10)
            
            # 加载相机参数
            camera_params_file = self.params.get("camera_params_file")
            if camera_params_file:
                 self.status_update.emit(f"加载相机标定参数: {os.path.basename(camera_params_file)}")
            
            self.progress_update.emit(20)
            
            # 执行标定
            calibration, calibration_file = cal.phase_based_projector_calibration(
                projector_width=self.params.get("projector_width"),
                projector_height=self.params.get("projector_height"),
                camera_params_file=camera_params_file,
                phase_images_folder=self.params.get("phase_images_folder"),
                board_type=self.params.get("board_type"),
                chessboard_size=(self.params.get("chessboard_width"), self.params.get("chessboard_height")),
                square_size=self.params.get("square_size"),
                output_folder=self.params.get("output_folder"),
                visualize=self.params.get("visualize", False), # 添加默认值
                global_optimization=self.params.get("global_optimization"),
                sampling_step=self.params.get("sampling_step"),
                adaptive_threshold=self.params.get("adaptive_threshold"),
                n_steps=self.params.get("n_steps"),
                print_func=custom_print # 传递自定义打印函数
            )
            
            # 发送完成信息
            self.progress_update.emit(100)
            self.status_update.emit(f"标定完成！结果已保存至: {calibration_file}")
            
            # 加载生成的图像，如果有的话
            output_dir = os.path.join(self.params.get("output_folder"), "phase_unwrapping")
            if os.path.exists(output_dir):
                # 尝试加载组合相位图
                combined_path = os.path.join(output_dir, "combined_phase.png")
                if os.path.exists(combined_path):
                    pixmap = QPixmap(combined_path)
                    self.image_update.emit("combined", pixmap)
                
                # 尝试加载水平和垂直相位图
                h_unwrapped_path = os.path.join(output_dir, "horizontal", "unwrapped_phase.png")
                if os.path.exists(h_unwrapped_path):
                    pixmap = QPixmap(h_unwrapped_path)
                    self.image_update.emit("horizontal", pixmap)
                    
                v_unwrapped_path = os.path.join(output_dir, "vertical", "unwrapped_phase.png")
                if os.path.exists(v_unwrapped_path):
                    pixmap = QPixmap(v_unwrapped_path)
                    self.image_update.emit("vertical", pixmap)
            
            # 发送标定结果
            result_dict = {
                "projector_matrix": calibration.projector_matrix.tolist() if calibration.projector_matrix is not None else None,
                "projector_dist": calibration.projector_dist.tolist() if calibration.projector_dist is not None else None,
                "rotation_matrix": calibration.R.tolist() if calibration.R is not None else None,
                "translation_vector": calibration.T.tolist() if calibration.T is not None else None,
                "calibration_file": calibration_file
            }
            self.calibration_complete.emit(result_dict)
            
        except Exception as e:
            self.calibration_error.emit(str(e))
            import traceback
            traceback.print_exc()
        finally:
            # 恢复原始print函数
            if hasattr(cal, 'original_print'):
                cal.print = cal.original_print
            self.status_update.emit("标定线程结束。")

class ImageViewer(QWidget):
    """图像查看器小部件"""
    
    def __init__(self, title="图像预览"):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        # 标题标签
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setBold(True)
        self.title_label.setFont(font)
        self.layout.addWidget(self.title_label)
        
        # 图像标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet(f"background-color: {COLORS['secondary']}; border: 1px solid {COLORS['accent']};")
        
        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidget(self.image_label)
        scroll.setWidgetResizable(True)
        self.layout.addWidget(scroll)
        
    def set_image(self, pixmap):
        """设置要显示的图像"""
        if pixmap:
            # 调整图像大小以适应标签
            scaled_pixmap = pixmap.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.clear()
            self.image_label.setText("无图像可显示")

class CalibrationResultWidget(QWidget):
    """标定结果显示小部件"""
    
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        # 结果文本区域
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.layout.addWidget(self.result_text)
        
    def set_result(self, result):
        """设置要显示的标定结果"""
        if not result:
            self.result_text.setText("无标定结果可显示")
            return
            
        text = "### 投影仪标定结果 ###\n\n"
        
        # 投影仪内参矩阵
        if "projector_matrix" in result and result["projector_matrix"]:
            text += "投影仪内参矩阵:\n"
            for row in result["projector_matrix"]:
                text += f"{row}\n"
            text += "\n"
        
        # 投影仪畸变系数
        if "projector_dist" in result and result["projector_dist"]:
            text += "投影仪畸变系数:\n"
            text += f"{result['projector_dist']}\n\n"
        
        # 旋转矩阵
        if "rotation_matrix" in result and result["rotation_matrix"]:
            text += "从投影仪到相机的旋转矩阵:\n"
            for row in result["rotation_matrix"]:
                text += f"{row}\n"
            text += "\n"
        
        # 平移向量
        if "translation_vector" in result and result["translation_vector"]:
            text += "从投影仪到相机的平移向量 (mm):\n"
            text += f"{result['translation_vector']}\n\n"
        
        # 保存位置
        if "calibration_file" in result:
            text += f"结果保存位置: {result['calibration_file']}\n"
        
        self.result_text.setText(text)

class ProjectorCalibrationGUI(QMainWindow):
    """投影仪标定系统的主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于相位解包裹的投影仪标定系统")
        self.setMinimumSize(1200, 800)
        
        # 设置应用样式
        self.setup_style()
        
        # 创建中央部件和主布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 使用QSplitter分割参数区域和结果区域
        self.splitter = QSplitter(Qt.Horizontal)
        
        # 创建参数设置区域
        self.params_widget = QWidget()
        self.params_layout = QVBoxLayout(self.params_widget)
        self.params_layout.setContentsMargins(10, 10, 10, 10)
        self.params_layout.setSpacing(15)
        
        # 创建结果显示区域
        self.results_widget = QTabWidget()
        
        # 添加参数和结果区域到分割器
        self.splitter.addWidget(self.params_widget)
        self.splitter.addWidget(self.results_widget)
        
        # 设置分割器的初始大小比例
        self.splitter.setSizes([400, 800])
        
        # 主布局
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.addWidget(self.splitter)
        
        # 添加所有UI元素
        self.setup_ui()
        
        # 初始化标定线程为None
        self.calibration_thread = None
        
        # 初始化标定结果
        self.calibration_result = None
        
    def setup_style(self):
        """设置应用样式"""
        # 设置应用字体
        app = QApplication.instance()
        font = QFont("Segoe UI", 10)
        app.setFont(font)
        
        # 设置样式表
        stylesheet = f"""
        QMainWindow, QWidget {{
            background-color: {COLORS['background']};
            color: {COLORS['text']};
        }}
        
        QPushButton {{
            background-color: {COLORS['primary']};
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
        }}
        
        QPushButton:hover {{
            background-color: #3d5d8a;
        }}
        
        QPushButton:pressed {{
            background-color: #2c4361;
        }}
        
        QPushButton:disabled {{
            background-color: {COLORS['accent']};
        }}
        
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            border: 1px solid {COLORS['accent']};
            padding: 5px;
            border-radius: 4px;
            background-color: white;
        }}
        
        /* 自定义QSpinBox和QDoubleSpinBox的上下箭头样式 */
        QSpinBox::up-button, QDoubleSpinBox::up-button {{
            background-color: #E0E0E0;
            border-top-right-radius: 3px;
            width: 16px;
            height: 10px;
        }}
        
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            background-color: #E0E0E0;
            border-bottom-right-radius: 3px;
            width: 16px;
            height: 10px;
        }}
        
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
            background-color: #D0D0D0;
        }}
        
        QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
        QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
            background-color: #C0C0C0;
        }}
        
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
            width: 10px;
            height: 10px;
        }}
        
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
            width: 10px;
            height: 10px;
        }}
        
        QGroupBox {{
            border: 1px solid {COLORS['accent']};
            border-radius: 4px;
            margin-top: 1em;
            padding-top: 10px;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }}
        
        QTabWidget::pane {{
            border: 1px solid {COLORS['accent']};
            border-radius: 4px;
        }}
        
        QTabBar::tab {{
            background-color: {COLORS['secondary']};
            border: 1px solid {COLORS['accent']};
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }}
        
        QTabBar::tab:selected {{
            background-color: {COLORS['primary']};
            color: white;
        }}
        
        QScrollArea, QTextEdit {{
            border: 1px solid {COLORS['accent']};
            border-radius: 4px;
        }}
        
        QProgressBar {{
            border: 1px solid {COLORS['accent']};
            border-radius: 4px;
            text-align: center;
        }}
        
        QProgressBar::chunk {{
            background-color: {COLORS['primary']};
            width: 10px;
            margin: 0.5px;
        }}
        """
        self.setStyleSheet(stylesheet)
    
    def setup_ui(self):
        """设置UI元素"""
        # 基本参数区域
        self.create_basic_params_group()
        
        # 标定板参数区域
        self.create_board_params_group()
        
        # 高级参数区域
        self.create_advanced_params_group()
        
        # 操作按钮区域
        self.create_action_buttons()
        
        # 添加弹性空间
        self.params_layout.addStretch()
        
        # 创建结果选项卡
        self.create_result_tabs()
        
        # 连接信号和槽
        self.connect_signals_slots()
        
    def create_basic_params_group(self):
        """创建基本参数组"""
        group = QGroupBox("基本参数")
        layout = QFormLayout()
        layout.setSpacing(10)
        
        # 相机参数文件
        self.camera_params_edit = QLineEdit()
        self.camera_params_edit.setPlaceholderText("选择相机标定参数文件...")
        self.camera_params_edit.setToolTip("选择之前由相机标定程序生成的相机内参和畸变系数文件 (.json, .npy)。")
        self.camera_params_btn = QPushButton("浏览...")
        camera_params_layout = QHBoxLayout()
        camera_params_layout.addWidget(self.camera_params_edit, 3)
        camera_params_layout.addWidget(self.camera_params_btn, 1)
        layout.addRow("相机参数文件:", camera_params_layout)
        
        # 相位图像文件夹
        self.phase_images_edit = QLineEdit()
        self.phase_images_edit.setPlaceholderText("选择包含多个姿态子文件夹的主文件夹...")
        self.phase_images_edit.setToolTip(
            "请选择一个主文件夹，该文件夹内包含多个子文件夹，每个子文件夹代表一个标定姿态。\n"
            "每个子文件夹内应有 2*N 张图像 (N为相移步数)。\n"
            "例如，对于N=4步相移，需要I1-I4 (水平)和I5-I8 (垂直)。"
        )
        self.phase_images_btn = QPushButton("浏览...")
        phase_images_layout = QHBoxLayout()
        phase_images_layout.addWidget(self.phase_images_edit, 3)
        phase_images_layout.addWidget(self.phase_images_btn, 1)
        layout.addRow("相位图像文件夹:", phase_images_layout)
        
        # 输出文件夹
        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setPlaceholderText("选择输出结果文件夹...")
        self.output_folder_edit.setToolTip("选择一个文件夹用于保存标定日志、中间相位图和最终的标定结果文件。")
        self.output_folder_btn = QPushButton("浏览...")
        output_folder_layout = QHBoxLayout()
        output_folder_layout.addWidget(self.output_folder_edit, 3)
        output_folder_layout.addWidget(self.output_folder_btn, 1)
        layout.addRow("输出文件夹:", output_folder_layout)
        
        # 投影仪分辨率
        res_layout = QHBoxLayout()
        self.proj_width_spin = QSpinBox()
        self.proj_width_spin.setRange(100, 10000)
        self.proj_width_spin.setValue(1280)
        self.proj_width_spin.setToolTip("输入投影仪的水平分辨率（宽度），单位为像素。")
        self.proj_height_spin = QSpinBox()
        self.proj_height_spin.setRange(100, 10000)
        self.proj_height_spin.setValue(720)
        self.proj_height_spin.setToolTip("输入投影仪的垂直分辨率（高度），单位为像素。")
        res_layout.addWidget(QLabel("宽:"))
        res_layout.addWidget(self.proj_width_spin)
        res_layout.addWidget(QLabel("高:"))
        res_layout.addWidget(self.proj_height_spin)
        layout.addRow("投影仪分辨率:", res_layout)
        
        group.setLayout(layout)
        self.params_layout.addWidget(group)
        
    def create_board_params_group(self):
        """创建标定板参数组"""
        group = QGroupBox("标定板参数")
        layout = QFormLayout()
        layout.setSpacing(10)
        
        # 标定板类型
        self.board_type_combo = QComboBox()
        self.board_type_combo.addItems(["棋盘格标定板", "圆形标定板", "环形标定板"])
        self.board_type_combo.setToolTip("根据您使用的物理标定板选择其类型。")
        layout.addRow("标定板类型:", self.board_type_combo)
        
        # 标定板尺寸
        board_size_layout = QHBoxLayout()
        self.board_width_spin = QSpinBox()
        self.board_width_spin.setRange(2, 30)
        self.board_width_spin.setValue(9)
        self.board_width_spin.setToolTip("输入标定板内部角点的数量（宽度方向）。")
        self.board_height_spin = QSpinBox()
        self.board_height_spin.setRange(2, 30)
        self.board_height_spin.setValue(6)
        self.board_height_spin.setToolTip("输入标定板内部角点的数量（高度方向）。")
        board_size_layout.addWidget(QLabel("宽:"))
        board_size_layout.addWidget(self.board_width_spin)
        board_size_layout.addWidget(QLabel("高:"))
        board_size_layout.addWidget(self.board_height_spin)
        layout.addRow("标定板点数:", board_size_layout)
        
        # 方格尺寸或圆心间距
        self.square_size_spin = QDoubleSpinBox()
        self.square_size_spin.setRange(1.0, 1000.0)
        self.square_size_spin.setValue(20.0)
        self.square_size_spin.setSuffix(" mm")
        self.square_size_spin.setToolTip("输入标定板上一个方格的边长，或圆心之间的距离，单位为毫米(mm)。")
        layout.addRow("方格尺寸或圆心间距:", self.square_size_spin)
        
        group.setLayout(layout)
        self.params_layout.addWidget(group)
    
    def create_advanced_params_group(self):
        """创建高级参数组"""
        group = QGroupBox("高级参数")
        layout = QFormLayout()
        layout.setSpacing(10)
        
        # 全局优化
        self.global_opt_check = QCheckBox("使用全局优化方法提高精度")
        self.global_opt_check.setChecked(True)
        self.global_opt_check.setToolTip("推荐勾选。该方法会联合优化相机和投影仪的参数，通常可以获得更精确的结果。")
        layout.addRow("", self.global_opt_check)
        
        # 相位图采样步长
        self.sampling_step_spin = QSpinBox()
        self.sampling_step_spin.setRange(1, 50)
        self.sampling_step_spin.setValue(4)
        self.sampling_step_spin.setToolTip("在从相位图中提取对应点时，每隔N个像素采一个点。值越小，点越多，计算越慢。")
        layout.addRow("相位图采样步长:", self.sampling_step_spin)
        
        # 相移步数
        self.n_steps_spin = QSpinBox()
        self.n_steps_spin.setRange(3, 16) # 3步到16步
        self.n_steps_spin.setValue(4)
        self.n_steps_spin.setToolTip("设置相移的步数(N)。\n每个姿态子文件夹内需要包含 2*N 张图像。")
        layout.addRow("相移步数 (N):", self.n_steps_spin)

        # 自适应阈值
        self.adaptive_threshold_check = QCheckBox("使用自适应质量阈值")
        self.adaptive_threshold_check.setChecked(True)
        self.adaptive_threshold_check.setToolTip("自动根据当前图像的质量来确定一个阈值，以过滤掉低质量的相位点。")
        layout.addRow("", self.adaptive_threshold_check)
        
        # 可视化结果
        self.visualize_check = QCheckBox("显示过程可视化图像")
        self.visualize_check.setChecked(False)
        self.visualize_check.setToolTip("勾选后，在标定过程中会弹出显示中间结果的图像窗口，可用于调试。")
        layout.addRow("", self.visualize_check)
        
        group.setLayout(layout)
        self.params_layout.addWidget(group)
    
    def create_action_buttons(self):
        """创建操作按钮区域"""
        buttons_layout = QHBoxLayout()
        
        # 开始标定按钮
        self.start_btn = QPushButton("开始标定")
        self.start_btn.setMinimumHeight(40)
        font = QFont()
        font.setBold(True)
        self.start_btn.setFont(font)
        buttons_layout.addWidget(self.start_btn)
        
        # 取消按钮
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setEnabled(False)
        buttons_layout.addWidget(self.cancel_btn)
        
        self.params_layout.addLayout(buttons_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.params_layout.addWidget(self.progress_bar)
    
    def create_result_tabs(self):
        """创建结果选项卡"""
        # 日志选项卡
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.results_widget.addTab(self.log_text, "处理日志")
        
        # 重定向标准输出到日志文本框
        self.log_redirector = LogRedirector(self.log_text)
        sys.stdout = self.log_redirector
        
        # 相位图选项卡
        self.image_tabs = QTabWidget()
        
        # 组合相位图
        self.combined_viewer = ImageViewer("组合相位图")
        self.image_tabs.addTab(self.combined_viewer, "组合相位图")
        
        # 水平相位图
        self.horizontal_viewer = ImageViewer("水平相位图")
        self.image_tabs.addTab(self.horizontal_viewer, "水平相位图")
        
        # 垂直相位图
        self.vertical_viewer = ImageViewer("垂直相位图")
        self.image_tabs.addTab(self.vertical_viewer, "垂直相位图")
        
        self.results_widget.addTab(self.image_tabs, "相位图")
        
        # 标定结果选项卡
        self.result_widget = CalibrationResultWidget()
        self.results_widget.addTab(self.result_widget, "标定结果")
    
    def connect_signals_slots(self):
        """连接信号和槽"""
        # 文件和文件夹选择按钮
        self.camera_params_btn.clicked.connect(self.select_camera_params)
        self.phase_images_btn.clicked.connect(self.select_phase_images)
        self.output_folder_btn.clicked.connect(self.select_output_folder)
        
        # 标定板类型下拉框
        self.board_type_combo.currentIndexChanged.connect(self.update_board_type_label)
        
        # 操作按钮
        self.start_btn.clicked.connect(self.start_calibration)
        self.cancel_btn.clicked.connect(self.cancel_calibration)
    
    def select_camera_params(self):
        """选择相机参数文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择相机标定参数文件", "", "JSON文件 (*.json);;NPY文件 (*.npy);;所有文件 (*)"
        )
        if file_path:
            self.camera_params_edit.setText(file_path)
    
    def select_phase_images(self):
        """选择相位图像文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "选择主图像文件夹 (其内部应包含多个姿态子文件夹)"
        )
        if folder_path:
            self.phase_images_edit.setText(folder_path)
            
            # 如果输出文件夹未指定，设置为相位图像文件夹的子目录
            if not self.output_folder_edit.text():
                self.output_folder_edit.setText(os.path.join(folder_path, "projector_calibration_results"))
    
    def select_output_folder(self):
        """选择输出文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "选择输出结果文件夹"
        )
        if folder_path:
            self.output_folder_edit.setText(folder_path)
    
    def update_board_type_label(self):
        """更新标定板类型相关标签"""
        board_type_idx = self.board_type_combo.currentIndex()
        
        if board_type_idx == 0:  # 棋盘格
            self.square_size_spin.setPrefix("")
            self.square_size_spin.setSuffix(" mm")
        elif board_type_idx == 1:  # 圆形标定板
            self.square_size_spin.setPrefix("")
            self.square_size_spin.setSuffix(" mm")
        elif board_type_idx == 2:  # 环形标定板
            self.square_size_spin.setPrefix("")
            self.square_size_spin.setSuffix(" mm")
    
    def validate_inputs(self):
        """验证输入参数"""
        # 检查相机参数文件
        camera_params = self.camera_params_edit.text().strip()
        if not camera_params:
            return False, "请选择相机标定参数文件"
        
        # 检查相位图像文件夹
        phase_images = self.phase_images_edit.text().strip()
        if not phase_images or not os.path.isdir(phase_images):
            return False, "请选择有效的相位图像文件夹"
        
        return True, ""
    
    def get_calibration_params(self):
        """获取标定参数"""
        # 获取标定板类型字符串
        board_type_idx = self.board_type_combo.currentIndex()
        board_types = ["chessboard", "circles", "ring_circles"]
        board_type = board_types[board_type_idx]
        
        # 构建参数字典
        params = {
            "camera_params_file": self.camera_params_edit.text().strip(),
            "phase_images_folder": self.phase_images_edit.text().strip(),
            "output_folder": self.output_folder_edit.text().strip(),
            "projector_width": self.proj_width_spin.value(),
            "projector_height": self.proj_height_spin.value(),
            "board_type": board_type,
            "chessboard_width": self.board_width_spin.value(),
            "chessboard_height": self.board_height_spin.value(),
            "square_size": self.square_size_spin.value(),
            "global_optimization": self.global_opt_check.isChecked(),
            "sampling_step": self.sampling_step_spin.value(),
            "adaptive_threshold": self.adaptive_threshold_check.isChecked(),
            "n_steps": self.n_steps_spin.value()
        }
        
        return params
    
    def start_calibration(self):
        """开始标定过程"""
        # 验证输入
        valid, message = self.validate_inputs()
        if not valid:
            QMessageBox.warning(self, "输入错误", message)
            return
        
        # 获取标定参数
        params = self.get_calibration_params()
        
        # 清空日志和结果
        self.log_text.clear()
        self.log_text.append("准备开始标定过程...")
        self.result_widget.set_result(None)
        self.combined_viewer.set_image(None)
        self.horizontal_viewer.set_image(None)
        self.vertical_viewer.set_image(None)
        
        # 创建并启动标定线程
        self.calibration_thread = CalibrationThread(params)
        self.calibration_thread.progress_update.connect(self.update_progress)
        self.calibration_thread.status_update.connect(self.update_status_and_log)
        self.calibration_thread.calibration_complete.connect(self.handle_calibration_complete)
        self.calibration_thread.calibration_error.connect(self.handle_calibration_error)
        self.calibration_thread.image_update.connect(self.handle_image_update)
        
        self.calibration_thread.start()
        
        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        
        # 切换到日志选项卡
        self.results_widget.setCurrentIndex(0)
    
    def cancel_calibration(self):
        """取消标定过程"""
        if self.calibration_thread and self.calibration_thread.isRunning():
            # 弹出确认对话框
            reply = QMessageBox.question(
                self, "取消标定", 
                "确定要取消正在进行的标定过程吗？", 
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.calibration_thread.terminate()
                self.log_text.append("标定过程已取消")
                
                # 更新UI状态
                self.start_btn.setEnabled(True)
                self.cancel_btn.setEnabled(False)
                self.progress_bar.setValue(0)
    
    @Slot(int)
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    @Slot(str)
    def update_status(self, message):
        """更新状态消息"""
        # 我们使用一个新函数同时更新日志和状态
        pass

    @Slot(str)
    def update_status_and_log(self, message):
        """更新日志和状态消息"""
        self.log_text.append(message)
    
    @Slot(dict)
    def handle_calibration_complete(self, result):
        """处理标定完成信号"""
        self.calibration_result = result
        self.result_widget.set_result(result)
        
        # 更新UI状态
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        
        # 切换到结果选项卡
        self.results_widget.setCurrentIndex(2)
        
        # 显示完成消息
        QMessageBox.information(self, "标定完成", "投影仪标定过程已成功完成！")
    
    @Slot(str)
    def handle_calibration_error(self, error_message):
        """处理标定错误信号"""
        self.log_text.append(f"<font color='red'>错误: {error_message}</font>")
        
        # 更新UI状态
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        
        # 显示错误消息
        QMessageBox.critical(self, "标定错误", f"标定过程发生错误:\n{error_message}")
    
    @Slot(str, QPixmap)
    def handle_image_update(self, image_type, pixmap):
        """处理图像更新信号"""
        if image_type == "combined":
            self.combined_viewer.set_image(pixmap)
            self.image_tabs.setCurrentIndex(0)
        elif image_type == "horizontal":
            self.horizontal_viewer.set_image(pixmap)
        elif image_type == "vertical":
            self.vertical_viewer.set_image(pixmap)
        
        # 切换到图像选项卡
        self.results_widget.setCurrentIndex(1)
    
    def closeEvent(self, event):
        """处理窗口关闭事件"""
        # 恢复标准输出
        sys.stdout = sys.__stdout__
        
        # 如果标定线程正在运行，终止它
        if self.calibration_thread and self.calibration_thread.isRunning():
            self.calibration_thread.terminate()
        
        event.accept()

if __name__ == "__main__":
    # 检查 PySide6 是否可用
    try:
        from PySide6 import __version__ as pyside_version
        print(f"PySide6 版本: {pyside_version}")
    except ImportError:
        print("错误: PySide6 未安装。请使用 'pip install PySide6' 安装。")
        sys.exit(1)
    
    # 检查投影仪标定模块是否可用
    if not hasattr(cal, "phase_based_projector_calibration"):
        print("错误: 投影仪标定模块不完整。请确保 projector_calibration_phase_based.py 正确导入。")
        sys.exit(1)
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 使用Fusion风格以获得跨平台一致的外观
    
    # 设置应用程序图标（如果有的话）
    # app.setWindowIcon(QIcon("icon.png"))
    
    window = ProjectorCalibrationGUI()
    window.show()
    
    sys.exit(app.exec()) 