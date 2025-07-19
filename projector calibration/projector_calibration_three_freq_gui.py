#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于三频外差相位解包裹的投影仪标定系统 - 图形用户界面

该程序提供基于PySide6的图形用户界面，用于执行基于三频外差法的投影仪标定过程。
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

# 导入三频外差投影仪标定模块
try:
    import projector_calibration_three_freq as cal_three_freq
except ImportError:
    print("Error: 无法导入三频外差投影仪标定模块。请确保 projector_calibration_three_freq.py 在当前目录中。")
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
    """日志重定向器"""
    def __init__(self, text_widget):
        self.text_widget = text_widget
        
    def write(self, text):
        if text.strip():  # 只处理非空文本
            # 使用QTimer确保在主线程中更新UI
            QTimer.singleShot(0, lambda: self.text_widget.append(text.strip()))
    
    def flush(self):
        pass


class CalibrationThread(QThread):
    """标定线程"""
    
    progress_update = Signal(int)
    status_update = Signal(str)
    calibration_complete = Signal(object, str)
    calibration_error = Signal(str)
    image_update = Signal(str, QPixmap)
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        
    def run(self):
        try:
            # 创建线程安全的print函数
            def thread_safe_print(*args, **kwargs):
                message = " ".join(map(str, args))
                self.status_update.emit(message)
            
            # 发送开始信息
            self.status_update.emit("开始三频外差标定过程...")
            self.progress_update.emit(10)
            
            # 加载相机参数
            camera_params_file = self.params.get("camera_params_file")
            if camera_params_file:
                self.status_update.emit(f"加载相机标定参数: {os.path.basename(camera_params_file)}")
            
            self.progress_update.emit(20)
            
            # 执行标定
            calibration, calibration_file = cal_three_freq.three_freq_projector_calibration(
                projector_width=self.params.get("projector_width"),
                projector_height=self.params.get("projector_height"),
                camera_params_file=camera_params_file,
                phase_images_folder=self.params.get("phase_images_folder"),
                board_type=self.params.get("board_type"),
                chessboard_size=(self.params.get("chessboard_width"), self.params.get("chessboard_height")),
                square_size=self.params.get("square_size"),
                output_folder=self.params.get("output_folder"),
                visualize=self.params.get("visualize", False),
                frequencies=self.params.get("frequencies"),
                phase_step=self.params.get("phase_step"),
                ph0=self.params.get("ph0"),
                quality_threshold=self.params.get("quality_threshold"),
                print_func=thread_safe_print  # 使用线程安全的print函数
            )
            
            self.progress_update.emit(100)
            self.calibration_complete.emit(calibration, calibration_file)
            
        except Exception as e:
            self.calibration_error.emit(str(e))
            import traceback
            self.status_update.emit(f"错误详情: {traceback.format_exc()}")
        finally:
            self.status_update.emit("标定线程结束。")


class ImageViewer(QWidget):
    """图像查看器组件"""
    
    def __init__(self, title="图像"):
        super().__init__()
        self.title = title
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # 标题标签
        self.title_label = QLabel(self.title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px; margin: 5px;")
        layout.addWidget(self.title_label)
        
        # 滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignCenter)
        
        # 图像标签
        self.image_label = QLabel("暂无图像")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        
        scroll_area.setWidget(self.image_label)
        layout.addWidget(scroll_area)
        
        self.setLayout(layout)
    
    def set_image(self, pixmap):
        """设置显示的图像"""
        if pixmap and not pixmap.isNull():
            # 缩放图像以适应显示
            scaled_pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.resize(scaled_pixmap.size())
        else:
            self.image_label.setText("图像加载失败")
    
    def clear_image(self):
        """清除图像"""
        self.image_label.clear()
        self.image_label.setText("暂无图像")


class CalibrationResultWidget(QWidget):
    """标定结果显示组件"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # 结果文本区域
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.result_text)
        
        self.setLayout(layout)
    
    def set_result(self, calibration):
        """设置标定结果"""
        if calibration is None:
            self.result_text.clear()
            return
        
        result_text = "【三频外差投影仪标定结果】\n\n"
        
        if hasattr(calibration, 'reprojection_error') and calibration.reprojection_error is not None:
            result_text += f"重投影误差: {calibration.reprojection_error:.6f} 像素\n\n"
        
        if hasattr(calibration, 'projector_matrix') and calibration.projector_matrix is not None:
            result_text += "投影仪内参矩阵:\n"
            result_text += str(calibration.projector_matrix) + "\n\n"
        
        if hasattr(calibration, 'projector_dist') and calibration.projector_dist is not None:
            result_text += "投影仪畸变系数:\n"
            result_text += str(calibration.projector_dist.flatten()) + "\n\n"
        
        if hasattr(calibration, 'R') and calibration.R is not None:
            result_text += "旋转矩阵 (投影仪到相机):\n"
            result_text += str(calibration.R) + "\n\n"
        
        if hasattr(calibration, 'T') and calibration.T is not None:
            result_text += "平移向量 (投影仪到相机, mm):\n"
            result_text += str(calibration.T.flatten()) + "\n\n"
        
        self.result_text.setPlainText(result_text)


class ThreeFreqProjectorCalibrationGUI(QMainWindow):
    """三频外差投影仪标定主界面"""
    
    def __init__(self):
        super().__init__()
        self.calibration_thread = None
        self.calibration_result = None
        self.setup_ui()
        self.connect_signals_slots()
        self.apply_styles()
        
    def setup_ui(self):
        """设置用户界面"""
        self.setWindowTitle("三频外差投影仪标定系统")
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧参数面板
        self.create_parameter_panel()
        splitter.addWidget(self.params_widget)
        
        # 右侧结果面板
        self.create_result_panel()
        splitter.addWidget(self.results_widget)
        
        # 设置分割器比例
        splitter.setSizes([400, 1000])
        
        # 创建状态栏
        self.statusBar().showMessage("准备就绪")
        
        # 创建进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
    
    def create_parameter_panel(self):
        """创建参数设置面板"""
        self.params_widget = QWidget()
        self.params_layout = QVBoxLayout()
        self.params_widget.setLayout(self.params_layout)
        
        # 标题
        title_label = QLabel("三频外差标定参数")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        self.params_layout.addWidget(title_label)
        
        # 文件路径设置
        self.create_file_path_group()
        
        # 投影仪参数设置
        self.create_projector_params_group()
        
        # 标定板参数设置
        self.create_board_params_group()
        
        # 三频外差参数设置
        self.create_three_freq_params_group()
        
        # 操作按钮
        self.create_action_buttons()
        
        # 添加弹性空间
        self.params_layout.addStretch()
    
    def create_file_path_group(self):
        """创建文件路径设置组"""
        group = QGroupBox("文件路径设置")
        layout = QFormLayout()
        
        # 相机标定参数文件
        self.camera_params_edit = QLineEdit()
        self.camera_params_btn = QPushButton("浏览...")
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(self.camera_params_edit)
        camera_layout.addWidget(self.camera_params_btn)
        layout.addRow("相机标定文件:", camera_layout)
        
        # 相移图像文件夹
        self.phase_images_edit = QLineEdit()
        self.phase_images_btn = QPushButton("浏览...")
        phase_layout = QHBoxLayout()
        phase_layout.addWidget(self.phase_images_edit)
        phase_layout.addWidget(self.phase_images_btn)
        layout.addRow("相移图像文件夹:", phase_layout)
        
        # 输出文件夹
        self.output_folder_edit = QLineEdit()
        self.output_folder_btn = QPushButton("浏览...")
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_folder_edit)
        output_layout.addWidget(self.output_folder_btn)
        layout.addRow("输出文件夹:", output_layout)
        
        group.setLayout(layout)
        self.params_layout.addWidget(group)
    
    def create_projector_params_group(self):
        """创建投影仪参数设置组"""
        group = QGroupBox("投影仪参数")
        layout = QFormLayout()
        
        # 投影仪分辨率
        self.proj_width_spin = QSpinBox()
        self.proj_width_spin.setRange(640, 4096)
        self.proj_width_spin.setValue(1024)
        layout.addRow("投影仪宽度:", self.proj_width_spin)
        
        self.proj_height_spin = QSpinBox()
        self.proj_height_spin.setRange(480, 2160)
        self.proj_height_spin.setValue(768)
        layout.addRow("投影仪高度:", self.proj_height_spin)
        
        group.setLayout(layout)
        self.params_layout.addWidget(group)
    
    def create_board_params_group(self):
        """创建标定板参数设置组"""
        group = QGroupBox("标定板参数")
        layout = QFormLayout()
        
        # 标定板类型
        self.board_type_combo = QComboBox()
        self.board_type_combo.addItems(["chessboard", "circles", "ring_circles"])
        layout.addRow("标定板类型:", self.board_type_combo)
        
        # 标定板尺寸
        self.board_width_spin = QSpinBox()
        self.board_width_spin.setRange(3, 20)
        self.board_width_spin.setValue(9)
        layout.addRow("标定板宽度(内角点):", self.board_width_spin)
        
        self.board_height_spin = QSpinBox()
        self.board_height_spin.setRange(3, 20)
        self.board_height_spin.setValue(6)
        layout.addRow("标定板高度(内角点):", self.board_height_spin)
        
        # 方格尺寸
        self.square_size_spin = QDoubleSpinBox()
        self.square_size_spin.setRange(1.0, 100.0)
        self.square_size_spin.setValue(20.0)
        self.square_size_spin.setDecimals(1)
        self.square_size_spin.setSuffix(" mm")
        layout.addRow("方格尺寸:", self.square_size_spin)
        
        group.setLayout(layout)
        self.params_layout.addWidget(group)
    
    def create_three_freq_params_group(self):
        """创建三频外差参数设置组"""
        group = QGroupBox("三频外差参数")
        layout = QFormLayout()
        
        # 三个频率值
        self.freq_high_spin = QSpinBox()
        self.freq_high_spin.setRange(32, 128)
        self.freq_high_spin.setValue(71)
        layout.addRow("高频:", self.freq_high_spin)
        
        self.freq_mid_spin = QSpinBox()
        self.freq_mid_spin.setRange(32, 128)
        self.freq_mid_spin.setValue(64)
        layout.addRow("中频:", self.freq_mid_spin)
        
        self.freq_low_spin = QSpinBox()
        self.freq_low_spin.setRange(32, 128)
        self.freq_low_spin.setValue(58)
        layout.addRow("低频:", self.freq_low_spin)
        
        # 相移步数
        self.phase_step_spin = QSpinBox()
        self.phase_step_spin.setRange(3, 16)
        self.phase_step_spin.setValue(4)
        layout.addRow("相移步数:", self.phase_step_spin)
        
        # 初始相位偏移
        self.ph0_spin = QDoubleSpinBox()
        self.ph0_spin.setRange(0.0, 1.0)
        self.ph0_spin.setValue(0.5)
        self.ph0_spin.setDecimals(3)
        layout.addRow("初始相位偏移:", self.ph0_spin)
        
        # 质量阈值
        self.quality_threshold_spin = QDoubleSpinBox()
        self.quality_threshold_spin.setRange(0.1, 1.0)
        self.quality_threshold_spin.setValue(0.3)
        self.quality_threshold_spin.setDecimals(3)
        layout.addRow("质量阈值:", self.quality_threshold_spin)
        
        # 可视化选项
        self.visualize_check = QCheckBox("显示过程可视化")
        self.visualize_check.setChecked(False)
        layout.addRow("", self.visualize_check)
        
        group.setLayout(layout)
        self.params_layout.addWidget(group)
    
    def create_action_buttons(self):
        """创建操作按钮"""
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("开始标定")
        self.start_btn.setStyleSheet(f"background-color: {COLORS['success']}; color: white; font-weight: bold; padding: 8px;")
        button_layout.addWidget(self.start_btn)
        
        self.cancel_btn = QPushButton("取消标定")
        self.cancel_btn.setStyleSheet(f"background-color: {COLORS['danger']}; color: white; font-weight: bold; padding: 8px;")
        self.cancel_btn.setEnabled(False)
        button_layout.addWidget(self.cancel_btn)
        
        self.params_layout.addLayout(button_layout)
    
    def create_result_panel(self):
        """创建结果显示面板"""
        self.results_widget = QTabWidget()
        self.create_result_tabs()
    
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
        
        # 质量图
        self.quality_viewer = ImageViewer("相位质量图")
        self.image_tabs.addTab(self.quality_viewer, "质量图")
        
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
    
    def apply_styles(self):
        """应用样式"""
        # 设置主窗口样式
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLORS['background']};
                color: {COLORS['text']};
            }}
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {COLORS['accent']};
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['text']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['secondary']};
                color: {COLORS['accent']};
            }}
            QLineEdit, QComboBox {{
                border: 1px solid {COLORS['accent']};
                border-radius: 3px;
                padding: 4px;
                background-color: white;
            }}
            QSpinBox, QDoubleSpinBox {{
                border: 1px solid {COLORS['accent']};
                border-radius: 3px;
                padding: 4px;
                background-color: white;
            }}
            QSpinBox::up-button, QDoubleSpinBox::up-button {{
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 20px;
                height: 12px;
                border-left: 1px solid {COLORS['accent']};
                border-bottom: 1px solid {COLORS['accent']};
                border-top-right-radius: 3px;
                background-color: {COLORS['primary']};
            }}
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
                background-color: {COLORS['accent']};
            }}
            QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {{
                background-color: {COLORS['text']};
            }}
            QSpinBox::down-button, QDoubleSpinBox::down-button {{
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 20px;
                height: 12px;
                border-left: 1px solid {COLORS['accent']};
                border-top: 1px solid {COLORS['accent']};
                border-bottom-right-radius: 3px;
                background-color: {COLORS['primary']};
            }}
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
                background-color: {COLORS['accent']};
            }}
            QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
                background-color: {COLORS['text']};
            }}
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 6px solid white;
                width: 0px;
                height: 0px;
            }}
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid white;
                width: 0px;
                height: 0px;
            }}
            QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover {{
                border-bottom: 6px solid #f0f0f0;
            }}
            QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {{
                border-top: 6px solid #f0f0f0;
            }}
            QTextEdit {{
                border: 1px solid {COLORS['accent']};
                border-radius: 3px;
                background-color: white;
                font-family: 'Consolas', 'Monaco', monospace;
            }}
            QTabWidget::pane {{
                border: 1px solid {COLORS['accent']};
                background-color: white;
            }}
            QTabBar::tab {{
                background-color: {COLORS['secondary']};
                color: {COLORS['text']};
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: {COLORS['primary']};
                color: white;
            }}
        """)
    
    @Slot()
    def select_camera_params(self):
        """选择相机标定参数文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择相机标定参数文件", "", 
            "标定文件 (*.npz *.json);;所有文件 (*)"
        )
        if file_path:
            self.camera_params_edit.setText(file_path)
    
    @Slot()
    def select_phase_images(self):
        """选择相移图像文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "选择相移图像文件夹"
        )
        if folder_path:
            self.phase_images_edit.setText(folder_path)
    
    @Slot()
    def select_output_folder(self):
        """选择输出文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "选择输出文件夹"
        )
        if folder_path:
            self.output_folder_edit.setText(folder_path)
    
    @Slot()
    def update_board_type_label(self):
        """更新标定板类型标签"""
        board_type = self.board_type_combo.currentText()
        if board_type == "chessboard":
            self.board_width_spin.setSuffix(" (内角点)")
            self.board_height_spin.setSuffix(" (内角点)")
        else:
            self.board_width_spin.setSuffix(" (圆点)")
            self.board_height_spin.setSuffix(" (圆点)")
    
    def validate_inputs(self):
        """验证输入参数"""
        if not self.camera_params_edit.text().strip():
            QMessageBox.warning(self, "输入错误", "请选择相机标定参数文件")
            return False
        
        if not os.path.exists(self.camera_params_edit.text().strip()):
            QMessageBox.warning(self, "文件错误", "相机标定参数文件不存在")
            return False
        
        if not self.phase_images_edit.text().strip():
            QMessageBox.warning(self, "输入错误", "请选择相移图像文件夹")
            return False
        
        if not os.path.exists(self.phase_images_edit.text().strip()):
            QMessageBox.warning(self, "文件夹错误", "相移图像文件夹不存在")
            return False
        
        if not self.output_folder_edit.text().strip():
            QMessageBox.warning(self, "输入错误", "请选择输出文件夹")
            return False
        
        # 检查频率值的合理性
        frequencies = [self.freq_high_spin.value(), self.freq_mid_spin.value(), self.freq_low_spin.value()]
        if frequencies[0] <= frequencies[1] or frequencies[1] <= frequencies[2]:
            QMessageBox.warning(self, "参数错误", "频率值必须按从高到低排序")
            return False
        
        # 检查相移图像文件夹结构
        phase_folder = self.phase_images_edit.text().strip()
        pose_folders = [d for d in os.listdir(phase_folder) 
                       if os.path.isdir(os.path.join(phase_folder, d))]
        
        if len(pose_folders) < 3:
            QMessageBox.warning(self, "数据错误", 
                              f"至少需要3个姿态文件夹，当前只有{len(pose_folders)}个")
            return False
        
        return True
    
    def get_calibration_params(self):
        """获取标定参数"""
        # 获取标定板类型
        board_type = self.board_type_combo.currentText()
        
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
            "frequencies": [self.freq_high_spin.value(), self.freq_mid_spin.value(), self.freq_low_spin.value()],
            "phase_step": self.phase_step_spin.value(),
            "ph0": self.ph0_spin.value(),
            "quality_threshold": self.quality_threshold_spin.value(),
            "visualize": self.visualize_check.isChecked()
        }
        
        return params
    
    @Slot()
    def start_calibration(self):
        """开始标定"""
        # 验证输入参数
        if not self.validate_inputs():
            return
        
        # 获取标定参数
        params = self.get_calibration_params()
        
        # 创建输出文件夹
        output_folder = params["output_folder"]
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法创建输出文件夹: {e}")
                return
        
        # 清空日志
        self.log_text.clear()
        self.log_text.append("开始三频外差投影仪标定...")
        
        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage("正在标定...")
        
        # 创建并启动标定线程
        self.calibration_thread = CalibrationThread(params)
        self.calibration_thread.progress_update.connect(self.progress_bar.setValue)
        self.calibration_thread.status_update.connect(self.log_text.append)
        self.calibration_thread.calibration_complete.connect(self.handle_calibration_complete)
        self.calibration_thread.calibration_error.connect(self.handle_calibration_error)
        self.calibration_thread.image_update.connect(self.handle_image_update)
        self.calibration_thread.start()
    
    @Slot()
    def cancel_calibration(self):
        """取消标定"""
        if self.calibration_thread and self.calibration_thread.isRunning():
            self.calibration_thread.terminate()
            self.calibration_thread.wait()
            
        self.log_text.append("<font color='orange'>标定已取消</font>")
        
        # 更新UI状态
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("标定已取消")
    
    @Slot(object, str)
    def handle_calibration_complete(self, calibration, calibration_file):
        """处理标定完成信号"""
        self.calibration_result = calibration

        # 更新UI状态
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("标定完成")

        # 显示标定结果
        self.result_widget.set_result(calibration)
        self.results_widget.setCurrentIndex(2)  # 切换到结果选项卡

        # 添加完成消息到日志
        self.log_text.append(f"<font color='green'>标定完成！结果已保存至: {calibration_file}</font>")

        # 显示成功消息框
        QMessageBox.information(self, "标定完成",
                              f"投影仪标定成功完成！\n"
                              f"重投影误差: {calibration.reprojection_error:.4f} 像素\n"
                              f"结果已保存至: {calibration_file}")

    @Slot(str)
    def handle_calibration_error(self, error_message):
        """处理标定错误信号"""
        # 更新UI状态
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("标定失败")

        # 添加错误消息到日志
        self.log_text.append(f"<font color='red'>标定失败: {error_message}</font>")

        # 显示错误消息框
        QMessageBox.critical(self, "标定失败", f"标定过程中发生错误:\n{error_message}")

    @Slot(str, QPixmap)
    def handle_image_update(self, image_type, pixmap):
        """处理图像更新信号"""
        if image_type == "combined":
            self.combined_viewer.set_image(pixmap)
        elif image_type == "horizontal":
            self.horizontal_viewer.set_image(pixmap)
        elif image_type == "vertical":
            self.vertical_viewer.set_image(pixmap)
        elif image_type == "quality":
            self.quality_viewer.set_image(pixmap)

        # 切换到图像选项卡
        self.results_widget.setCurrentIndex(1)


if __name__ == "__main__":
    # 检查 PySide6 是否可用
    try:
        from PySide6 import __version__ as pyside_version
        print(f"PySide6 版本: {pyside_version}")
    except ImportError:
        print("错误: PySide6 未安装。请使用 'pip install PySide6' 安装。")
        sys.exit(1)
    
    # 检查三频外差投影仪标定模块是否可用
    if not hasattr(cal_three_freq, "three_freq_projector_calibration"):
        print("错误: 三频外差投影仪标定模块不完整。请确保 projector_calibration_three_freq.py 正确导入。")
        sys.exit(1)
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 使用Fusion风格以获得跨平台一致的外观
    
    # 设置应用程序图标（如果有的话）
    # app.setWindowIcon(QIcon("icon.png"))
    
    window = ThreeFreqProjectorCalibrationGUI()
    window.show()
    
    sys.exit(app.exec())






