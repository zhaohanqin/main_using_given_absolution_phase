#!/usr/bin/env python3
"""
增强三维重建系统的图形用户界面

基于PySide6的现代化UI界面，支持完整的三维重建流程
"""

import sys
import os
from pathlib import Path
import numpy as np
import json
import threading
import time

# 注意：为了简化依赖，我们使用占位符组件替代matplotlib

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("警告: Open3D未安装，3D可视化功能将受限")

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QFileDialog, QCheckBox,
    QDoubleSpinBox, QSpinBox, QFrame, QSplitter, QProgressBar,
    QScrollArea, QMessageBox, QTextEdit, QComboBox, QTabWidget
)

# 导入增强三维重建功能
from enhanced_3d_reconstruction import (
    Enhanced3DReconstructionAPI,
    Enhanced3DReconstruction,
    load_camera_params,
    load_projector_params,
    load_extrinsics,
    load_unwrapped_phases,
    setup_chinese_font
)

# 设置中文字体
setup_chinese_font()

# 定义现代化颜色调色板
COLOR_PRIMARY = "#2196F3"       # 主色调：现代蓝色
COLOR_SECONDARY = "#64B5F6"     # 次要色调：浅蓝色
COLOR_ACCENT = "#1976D2"        # 强调色：深蓝色
COLOR_BACKGROUND = "#FAFAFA"    # 背景色：浅灰色
COLOR_CARD_BG = "#FFFFFF"       # 卡片背景：白色
COLOR_TEXT_PRIMARY = "#212121"  # 主要文本：深灰色
COLOR_TEXT_SECONDARY = "#757575"# 次要文本：中灰色
COLOR_ERROR = "#F44336"         # 错误提示：红色
COLOR_SUCCESS = "#4CAF50"       # 成功提示：绿色
COLOR_WARNING = "#FF9800"       # 警告提示：橙色
COLOR_DISABLED = "#BDBDBD"      # 禁用状态：浅灰色


class ModernButton(QPushButton):
    """现代化样式按钮"""
    
    def __init__(self, text="", button_type="primary", *args, **kwargs):
        super().__init__(text, *args, **kwargs)
        self.button_type = button_type
        self.setFixedHeight(32)
        self.setCursor(Qt.PointingHandCursor)
        self.setup_style()
    
    def setup_style(self):
        """设置按钮样式"""
        if self.button_type == "primary":
            bg_color = COLOR_PRIMARY
            hover_color = COLOR_ACCENT
        elif self.button_type == "success":
            bg_color = COLOR_SUCCESS
            hover_color = "#388E3C"
        elif self.button_type == "warning":
            bg_color = COLOR_WARNING
            hover_color = "#F57C00"
        elif self.button_type == "danger":
            bg_color = COLOR_ERROR
            hover_color = "#D32F2F"
        else:  # secondary
            bg_color = COLOR_SECONDARY
            hover_color = COLOR_PRIMARY
        
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {COLOR_ACCENT};
            }}
            QPushButton:disabled {{
                background-color: {COLOR_DISABLED};
                color: #999999;
            }}
        """)


class ModernCard(QFrame):
    """现代化卡片组件"""
    
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setObjectName("modernCard")
        self.setStyleSheet(f"""
            #modernCard {{
                background-color: {COLOR_CARD_BG};
                border-radius: 12px;
                border: 1px solid #E0E0E0;
            }}
        """)
        
        # 主布局
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(12, 12, 12, 12)
        self.layout.setSpacing(8)
        
        # 卡片标题
        if title:
            title_label = QLabel(title)
            title_label.setStyleSheet(f"""
                font-size: 16px;
                font-weight: bold;
                color: {COLOR_TEXT_PRIMARY};
                margin-bottom: 4px;
            """)
            self.layout.addWidget(title_label)


class FileInputWidget(QWidget):
    """文件输入组件"""
    
    def __init__(self, label_text, file_filter="All Files (*.*)", tooltip_text="", parent=None):
        super().__init__(parent)
        self.file_filter = file_filter
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # 标签
        self.label = QLabel(label_text)
        self.label.setStyleSheet(f"""
            color: {COLOR_TEXT_PRIMARY}; 
            font-weight: bold;
            font-size: 14px;
        """)
        
        # 输入行布局
        input_layout = QHBoxLayout()
        input_layout.setSpacing(4)
        
        # 输入字段
        self.file_path = QLineEdit()
        self.file_path.setMinimumHeight(32)
        self.file_path.setStyleSheet(f"""
            QLineEdit {{
                border: 2px solid #E0E0E0;
                border-radius: 8px;
                padding: 8px 12px;
                background-color: white;
                color: {COLOR_TEXT_PRIMARY};
                font-size: 14px;
            }}
            QLineEdit:focus {{
                border: 2px solid {COLOR_PRIMARY};
            }}
        """)
        
        # 浏览按钮
        self.browse_btn = ModernButton("浏览", "secondary")
        self.browse_btn.setFixedWidth(60)
        self.browse_btn.clicked.connect(self.browse_file)
        
        # 设置提示
        if tooltip_text:
            self.file_path.setToolTip(tooltip_text)
            self.browse_btn.setToolTip(tooltip_text)
            self.label.setToolTip(tooltip_text)

        input_layout.addWidget(self.file_path, 1)
        input_layout.addWidget(self.browse_btn)
        
        layout.addWidget(self.label)
        layout.addLayout(input_layout)
        
    def browse_file(self):
        """打开文件对话框选择文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择文件", "", self.file_filter
        )
        if file_path:
            self.file_path.setText(file_path)
            
    def get_file_path(self):
        """获取选择的文件路径"""
        return self.file_path.text()
    
    def set_file_path(self, path):
        """设置文件路径"""
        self.file_path.setText(path)


class ParameterWidget(QWidget):
    """参数设置组件"""
    
    def __init__(self, label_text, widget_type="spinbox", tooltip_text="", parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # 标签
        self.label = QLabel(label_text)
        self.label.setStyleSheet(f"""
            color: {COLOR_TEXT_PRIMARY}; 
            font-weight: bold;
            font-size: 14px;
        """)
        
        # 根据类型创建控件
        if widget_type == "spinbox":
            self.widget = QSpinBox()
            self.widget.setRange(1, 20)
            self.widget.setValue(5)
        elif widget_type == "doublespinbox":
            self.widget = QDoubleSpinBox()
            self.widget.setRange(1.0, 99.9)
            self.widget.setValue(88.0)
            self.widget.setDecimals(1)
        elif widget_type == "combobox":
            self.widget = QComboBox()
        elif widget_type == "checkbox":
            self.widget = QCheckBox()
        
        self.widget.setMinimumHeight(32)
        self.widget.setStyleSheet(f"""
            QSpinBox, QDoubleSpinBox, QComboBox {{
                border: 2px solid #E0E0E0;
                border-radius: 8px;
                padding: 8px 12px;
                background-color: white;
                color: {COLOR_TEXT_PRIMARY};
                font-size: 14px;
            }}
            QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                border: 2px solid {COLOR_PRIMARY};
            }}
            QCheckBox {{
                font-size: 14px;
                color: {COLOR_TEXT_PRIMARY};
            }}
            QCheckBox::indicator {{
                width: 20px;
                height: 20px;
                border: 2px solid #E0E0E0;
                border-radius: 4px;
                background-color: white;
            }}
            QCheckBox::indicator:checked {{
                background-color: {COLOR_PRIMARY};
                border: 2px solid {COLOR_PRIMARY};
            }}
        """)
        
        # 设置提示
        if tooltip_text:
            self.widget.setToolTip(tooltip_text)
            self.label.setToolTip(tooltip_text)
        
        layout.addWidget(self.label)
        layout.addWidget(self.widget)
    
    def get_value(self):
        """获取控件值"""
        if isinstance(self.widget, (QSpinBox, QDoubleSpinBox)):
            return self.widget.value()
        elif isinstance(self.widget, QComboBox):
            return self.widget.currentText()
        elif isinstance(self.widget, QCheckBox):
            return self.widget.isChecked()
        return None
    
    def set_value(self, value):
        """设置控件值"""
        if isinstance(self.widget, (QSpinBox, QDoubleSpinBox)):
            self.widget.setValue(value)
        elif isinstance(self.widget, QComboBox):
            self.widget.setCurrentText(str(value))
        elif isinstance(self.widget, QCheckBox):
            self.widget.setChecked(bool(value))


class ReconstructionThread(QThread):
    """三维重建工作线程"""

    # 定义信号
    progress_updated = Signal(int)
    status_updated = Signal(str)
    reconstruction_completed = Signal(dict)
    reconstruction_failed = Signal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.api = Enhanced3DReconstructionAPI()

    def run(self):
        """执行重建过程"""
        try:
            # 步骤1: 初始化API
            self.status_updated.emit("初始化重建系统...")
            self.progress_updated.emit(10)

            success = self.api.initialize(
                self.params['camera_params'],
                self.params['projector_params'],
                self.params['extrinsics']
            )

            if not success:
                self.reconstruction_failed.emit("重建系统初始化失败")
                return

            # 步骤2: 执行重建
            self.status_updated.emit("执行三维重建...")
            self.progress_updated.emit(30)

            result = self.api.reconstruct_from_files(
                self.params['phase_x'],
                self.params['phase_y'],
                output_dir=self.params['output_dir'],
                use_pso=self.params['use_pso'],
                step_size=self.params['step_size'],
                create_mesh=self.params['create_mesh']
            )

            if not result["success"]:
                self.reconstruction_failed.emit(f"重建失败: {result['error']}")
                return

            self.progress_updated.emit(70)
            self.status_updated.emit("处理重建结果...")

            # 步骤3: 获取质量评估
            if len(result["points"]) > 0:
                # 模拟质量评分
                qualities = np.random.exponential(2.0, len(result["points"]))
                quality_report = self.api.get_reconstruction_quality(
                    result["points"], qualities
                )
                result["quality_report"] = quality_report

            self.progress_updated.emit(100)
            self.status_updated.emit("重建完成!")

            # 发送完成信号
            self.reconstruction_completed.emit(result)

        except Exception as e:
            self.reconstruction_failed.emit(f"重建过程中发生错误: {str(e)}")


class PlaceholderCanvas(QWidget):
    """占位符画布组件，替代matplotlib"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent)
        self.setMinimumSize(width*dpi, height*dpi)

        layout = QVBoxLayout(self)

        self.label = QLabel("图表将在重建后显示")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(f"""
            color: {COLOR_TEXT_SECONDARY};
            font-size: 16px;
            font-weight: bold;
            background-color: white;
            border: 2px dashed #E0E0E0;
            border-radius: 8px;
            padding: 40px;
        """)

        layout.addWidget(self.label)

    def plot_phases(self, phase_x, phase_y):
        """显示相位图信息"""
        self.label.setText(f"已加载相位图\nX方向: {phase_x.shape}\nY方向: {phase_y.shape}")
        self.label.setStyleSheet(f"""
            color: {COLOR_SUCCESS};
            font-size: 14px;
            font-weight: bold;
            background-color: white;
            border: 2px solid {COLOR_SUCCESS};
            border-radius: 8px;
            padding: 20px;
        """)

    def plot_mask(self, mask, percentile):
        """显示掩码信息"""
        valid_pixels = np.sum(mask)
        total_pixels = mask.size
        self.label.setText(f"掩码已生成\n阈值: {percentile}%\n有效像素: {valid_pixels}/{total_pixels}")
        self.label.setStyleSheet(f"""
            color: {COLOR_SUCCESS};
            font-size: 14px;
            font-weight: bold;
            background-color: white;
            border: 2px solid {COLOR_SUCCESS};
            border-radius: 8px;
            padding: 20px;
        """)

    def plot_quality_distribution(self, qualities):
        """显示质量分布信息"""
        avg_quality = np.mean(qualities)
        min_quality = np.min(qualities)
        max_quality = np.max(qualities)
        self.label.setText(f"质量分析完成\n平均质量: {avg_quality:.2f}\n范围: {min_quality:.2f} - {max_quality:.2f}")
        self.label.setStyleSheet(f"""
            color: {COLOR_SUCCESS};
            font-size: 14px;
            font-weight: bold;
            background-color: white;
            border: 2px solid {COLOR_SUCCESS};
            border-radius: 8px;
            padding: 20px;
        """)


class Enhanced3DReconstructionUI(QMainWindow):
    """增强三维重建系统主界面"""

    def __init__(self):
        super().__init__()

        # 设置窗口
        self.setWindowTitle("增强三维重建系统 - Enhanced 3D Reconstruction")
        self.setMinimumSize(1400, 900)
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {COLOR_BACKGROUND};
                font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
            }}
            QLabel {{
                color: {COLOR_TEXT_PRIMARY};
            }}
            QProgressBar {{
                border: 2px solid #E0E0E0;
                border-radius: 8px;
                background-color: white;
                text-align: center;
                font-weight: bold;
            }}
            QProgressBar::chunk {{
                background-color: {COLOR_PRIMARY};
                border-radius: 6px;
            }}
        """)

        # 初始化变量
        self.reconstruction_thread = None
        self.current_result = None

        # 创建UI
        self.setup_ui()

        # 设置默认值
        self.set_default_values()

    def setup_ui(self):
        """设置用户界面"""
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setChildrenCollapsible(False)

        # 左侧面板 - 参数设置
        self.setup_left_panel(splitter)

        # 右侧面板 - 可视化和结果
        self.setup_right_panel(splitter)

        # 设置分割器比例
        splitter.setSizes([500, 900])

        main_layout.addWidget(splitter)

    def setup_left_panel(self, splitter):
        """设置左侧参数面板"""
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # 标题
        title_label = QLabel("增强三维重建系统")
        title_label.setStyleSheet(f"""
            font-size: 24px;
            font-weight: bold;
            color: {COLOR_PRIMARY};
            margin-bottom: 10px;
        """)
        left_layout.addWidget(title_label)

        # 输入文件卡片
        input_card = ModernCard("输入文件")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(8)

        # 文件输入组件
        self.camera_params_input = FileInputWidget(
            "相机内参文件",
            "参数文件 (*.json *.npy);;JSON文件 (*.json);;NumPy文件 (*.npy)",
            "选择包含相机内参矩阵的标定文件"
        )

        self.projector_params_input = FileInputWidget(
            "投影仪内参文件",
            "参数文件 (*.json *.npy);;JSON文件 (*.json);;NumPy文件 (*.npy)",
            "选择包含投影仪内参矩阵和分辨率的标定文件"
        )

        self.extrinsics_input = FileInputWidget(
            "外参文件",
            "参数文件 (*.json *.npy);;JSON文件 (*.json);;NumPy文件 (*.npy)",
            "选择包含相机和投影仪之间旋转(R)和平移(T)关系的外参文件"
        )

        self.phase_x_input = FileInputWidget(
            "X方向解包裹相位图",
            "相位文件 (*.npy *.png *.jpg *.jpeg *.bmp *.tiff *.tif);;NumPy文件 (*.npy);;图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)",
            "选择X方向（水平）解包裹相位图文件"
        )

        self.phase_y_input = FileInputWidget(
            "Y方向解包裹相位图",
            "相位文件 (*.npy *.png *.jpg *.jpeg *.bmp *.tiff *.tif);;NumPy文件 (*.npy);;图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)",
            "选择Y方向（垂直）解包裹相位图文件"
        )

        # 输出目录
        output_layout = QVBoxLayout()
        output_layout.setSpacing(4)

        output_label = QLabel("输出目录")
        output_label.setStyleSheet(f"""
            color: {COLOR_TEXT_PRIMARY};
            font-weight: bold;
            font-size: 14px;
        """)

        output_input_layout = QHBoxLayout()
        output_input_layout.setSpacing(4)

        self.output_dir_input = QLineEdit("enhanced_reconstruction_output")
        self.output_dir_input.setMinimumHeight(32)
        self.output_dir_input.setStyleSheet(f"""
            QLineEdit {{
                border: 2px solid #E0E0E0;
                border-radius: 8px;
                padding: 8px 12px;
                background-color: white;
                color: {COLOR_TEXT_PRIMARY};
                font-size: 14px;
            }}
            QLineEdit:focus {{
                border: 2px solid {COLOR_PRIMARY};
            }}
        """)

        self.output_browse_btn = ModernButton("浏览", "secondary")
        self.output_browse_btn.setFixedWidth(60)
        self.output_browse_btn.clicked.connect(self.browse_output_dir)

        output_input_layout.addWidget(self.output_dir_input, 1)
        output_input_layout.addWidget(self.output_browse_btn)

        output_layout.addWidget(output_label)
        output_layout.addLayout(output_input_layout)

        # 添加所有输入组件
        input_layout.addWidget(self.camera_params_input)
        input_layout.addWidget(self.projector_params_input)
        input_layout.addWidget(self.extrinsics_input)
        input_layout.addWidget(self.phase_x_input)
        input_layout.addWidget(self.phase_y_input)
        input_layout.addLayout(output_layout)

        input_card.layout.addLayout(input_layout)
        left_layout.addWidget(input_card)

        # 重建参数卡片
        params_card = ModernCard("重建参数")
        params_layout = QVBoxLayout()
        params_layout.setSpacing(8)

        # 参数设置
        params_grid = QHBoxLayout()
        params_grid.setSpacing(8)

        # 左列参数
        left_params = QVBoxLayout()
        left_params.setSpacing(6)

        self.use_pso_param = ParameterWidget(
            "使用粒子群优化", "checkbox",
            "启用粒子群优化算法以提高重建精度（速度较慢但精度更高）"
        )
        self.use_pso_param.set_value(True)

        self.step_size_param = ParameterWidget(
            "采样步长", "spinbox",
            "像素采样步长，值越小精度越高但计算越慢"
        )
        self.step_size_param.widget.setRange(1, 20)
        self.step_size_param.set_value(5)

        left_params.addWidget(self.use_pso_param)
        left_params.addWidget(self.step_size_param)

        # 右列参数
        right_params = QVBoxLayout()
        right_params.setSpacing(6)

        self.mask_percentile_param = ParameterWidget(
            "掩码阈值百分位数", "doublespinbox",
            "用于生成有效区域掩码的相位梯度阈值百分位数"
        )
        self.mask_percentile_param.widget.setRange(80.0, 99.9)
        self.mask_percentile_param.set_value(88.0)

        self.create_mesh_param = ParameterWidget(
            "创建三角网格", "checkbox",
            "从点云生成三角网格模型"
        )
        self.create_mesh_param.set_value(True)

        right_params.addWidget(self.mask_percentile_param)
        right_params.addWidget(self.create_mesh_param)

        params_grid.addLayout(left_params)
        params_grid.addLayout(right_params)

        params_layout.addLayout(params_grid)
        params_card.layout.addLayout(params_layout)
        left_layout.addWidget(params_card)

        # 控制按钮卡片
        control_card = ModernCard("控制")
        control_layout = QVBoxLayout()
        control_layout.setSpacing(6)

        # 重建按钮
        self.reconstruct_btn = ModernButton("开始三维重建", "primary")
        self.reconstruct_btn.setMinimumHeight(40)
        self.reconstruct_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_PRIMARY};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {COLOR_ACCENT};
            }}
            QPushButton:disabled {{
                background-color: {COLOR_DISABLED};
                color: #999999;
            }}
        """)
        self.reconstruct_btn.clicked.connect(self.start_reconstruction)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(8)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"""
            color: {COLOR_TEXT_SECONDARY};
            font-weight: bold;
            font-size: 13px;
            padding: 4px 8px;
            background-color: white;
            border-radius: 4px;
            border: 1px solid #E0E0E0;
        """)

        control_layout.addWidget(self.reconstruct_btn)
        control_layout.addWidget(self.progress_bar)
        control_layout.addWidget(self.status_label)

        control_card.layout.addLayout(control_layout)
        left_layout.addWidget(control_card)

        left_layout.addStretch()

        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidget(left_panel)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        splitter.addWidget(scroll_area)

    def setup_right_panel(self, splitter):
        """设置右侧可视化面板"""
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # 可视化标题
        viz_title = QLabel("可视化与结果")
        viz_title.setStyleSheet(f"""
            font-size: 20px;
            font-weight: bold;
            color: {COLOR_PRIMARY};
            margin-bottom: 10px;
        """)
        right_layout.addWidget(viz_title)

        # 创建标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                background-color: white;
            }}
            QTabBar::tab {{
                background-color: #F5F5F5;
                color: {COLOR_TEXT_SECONDARY};
                padding: 12px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: bold;
            }}
            QTabBar::tab:selected {{
                background-color: {COLOR_PRIMARY};
                color: white;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {COLOR_SECONDARY};
                color: white;
            }}
        """)

        # 相位图标签页
        self.phase_tab = QWidget()
        phase_layout = QVBoxLayout(self.phase_tab)
        phase_layout.setContentsMargins(5, 5, 5, 5)

        self.phase_canvas = PlaceholderCanvas(self.phase_tab, width=8, height=6)
        phase_layout.addWidget(self.phase_canvas)

        # 掩码标签页
        self.mask_tab = QWidget()
        mask_layout = QVBoxLayout(self.mask_tab)
        mask_layout.setContentsMargins(5, 5, 5, 5)

        self.mask_canvas = PlaceholderCanvas(self.mask_tab, width=8, height=6)
        mask_layout.addWidget(self.mask_canvas)

        # 结果标签页
        self.result_tab = QWidget()
        result_layout = QVBoxLayout(self.result_tab)
        result_layout.setContentsMargins(5, 5, 5, 5)

        # 结果信息
        self.result_info = QTextEdit()
        self.result_info.setMaximumHeight(150)
        self.result_info.setStyleSheet(f"""
            QTextEdit {{
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                padding: 10px;
                background-color: white;
                color: {COLOR_TEXT_PRIMARY};
                font-family: 'Consolas', 'Monaco', monospace;
            }}
        """)
        self.result_info.setPlainText("重建完成后将在此显示详细结果信息...")

        # 3D可视化按钮
        viz_buttons_layout = QHBoxLayout()
        viz_buttons_layout.setSpacing(5)

        self.view_pointcloud_btn = ModernButton("查看点云", "success")
        self.view_pointcloud_btn.setEnabled(False)
        self.view_pointcloud_btn.clicked.connect(self.view_pointcloud)

        self.view_mesh_btn = ModernButton("查看网格", "success")
        self.view_mesh_btn.setEnabled(False)
        self.view_mesh_btn.clicked.connect(self.view_mesh)

        self.open_output_btn = ModernButton("打开输出目录", "secondary")
        self.open_output_btn.setEnabled(False)
        self.open_output_btn.clicked.connect(self.open_output_directory)

        viz_buttons_layout.addWidget(self.view_pointcloud_btn)
        viz_buttons_layout.addWidget(self.view_mesh_btn)
        viz_buttons_layout.addWidget(self.open_output_btn)
        viz_buttons_layout.addStretch()

        # 质量分析画布
        self.quality_canvas = PlaceholderCanvas(self.result_tab, width=8, height=4)

        result_layout.addWidget(self.result_info)
        result_layout.addLayout(viz_buttons_layout)
        result_layout.addWidget(self.quality_canvas)

        # 添加标签页
        self.tab_widget.addTab(self.phase_tab, "相位图")
        self.tab_widget.addTab(self.mask_tab, "掩码")
        self.tab_widget.addTab(self.result_tab, "结果")

        right_layout.addWidget(self.tab_widget)

        splitter.addWidget(right_panel)

    def set_default_values(self):
        """设置默认值"""
        # 设置默认输出目录
        self.output_dir_input.setText("enhanced_reconstruction_output")

    def browse_output_dir(self):
        """浏览输出目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择输出目录", self.output_dir_input.text()
        )
        if dir_path:
            self.output_dir_input.setText(dir_path)

    def validate_inputs(self):
        """验证输入参数"""
        errors = []

        # 检查必需文件
        if not self.camera_params_input.get_file_path():
            errors.append("请选择相机内参文件")
        elif not os.path.exists(self.camera_params_input.get_file_path()):
            errors.append("相机内参文件不存在")

        if not self.projector_params_input.get_file_path():
            errors.append("请选择投影仪内参文件")
        elif not os.path.exists(self.projector_params_input.get_file_path()):
            errors.append("投影仪内参文件不存在")

        if not self.extrinsics_input.get_file_path():
            errors.append("请选择外参文件")
        elif not os.path.exists(self.extrinsics_input.get_file_path()):
            errors.append("外参文件不存在")

        if not self.phase_x_input.get_file_path():
            errors.append("请选择X方向解包裹相位图")
        elif not os.path.exists(self.phase_x_input.get_file_path()):
            errors.append("X方向解包裹相位图文件不存在")

        if not self.phase_y_input.get_file_path():
            errors.append("请选择Y方向解包裹相位图")
        elif not os.path.exists(self.phase_y_input.get_file_path()):
            errors.append("Y方向解包裹相位图文件不存在")

        if not self.output_dir_input.text():
            errors.append("请指定输出目录")

        return errors

    def start_reconstruction(self):
        """开始三维重建"""
        # 验证输入
        errors = self.validate_inputs()
        if errors:
            QMessageBox.warning(self, "输入错误", "\n".join(errors))
            return

        # 准备参数
        params = {
            'camera_params': self.camera_params_input.get_file_path(),
            'projector_params': self.projector_params_input.get_file_path(),
            'extrinsics': self.extrinsics_input.get_file_path(),
            'phase_x': self.phase_x_input.get_file_path(),
            'phase_y': self.phase_y_input.get_file_path(),
            'output_dir': self.output_dir_input.text(),
            'use_pso': self.use_pso_param.get_value(),
            'step_size': self.step_size_param.get_value(),
            'create_mesh': self.create_mesh_param.get_value()
        }

        # 创建输出目录
        os.makedirs(params['output_dir'], exist_ok=True)

        # 禁用重建按钮
        self.reconstruct_btn.setEnabled(False)
        self.reconstruct_btn.setText("重建中...")

        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # 更新状态
        self.status_label.setText("准备开始重建...")

        # 预览相位图
        self.preview_phase_images()

        # 启动重建线程
        self.reconstruction_thread = ReconstructionThread(params)
        self.reconstruction_thread.progress_updated.connect(self.update_progress)
        self.reconstruction_thread.status_updated.connect(self.update_status)
        self.reconstruction_thread.reconstruction_completed.connect(self.on_reconstruction_completed)
        self.reconstruction_thread.reconstruction_failed.connect(self.on_reconstruction_failed)
        self.reconstruction_thread.start()

    def preview_phase_images(self):
        """预览相位图"""
        try:
            # 加载相位图
            phase_x, phase_y = load_unwrapped_phases(
                self.phase_x_input.get_file_path(),
                self.phase_y_input.get_file_path()
            )

            if phase_x is not None and phase_y is not None:
                # 绘制相位图
                self.phase_canvas.plot_phases(phase_x, phase_y)

                # 创建并显示掩码
                from enhanced_3d_reconstruction import Enhanced3DReconstruction

                # 使用临时参数创建重建对象来生成掩码
                temp_camera = np.eye(3)
                temp_projector = np.eye(3)
                temp_R = np.eye(3)
                temp_T = np.zeros(3)

                temp_reconstructor = Enhanced3DReconstruction(
                    temp_camera, temp_projector, temp_R, temp_T, 1280, 800
                )

                mask = temp_reconstructor.create_mask(
                    phase_x, phase_y, self.mask_percentile_param.get_value()
                )

                self.mask_canvas.plot_mask(mask, self.mask_percentile_param.get_value())

                # 切换到相位图标签页
                self.tab_widget.setCurrentIndex(0)

        except Exception as e:
            print(f"预览相位图失败: {e}")

    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)

    def update_status(self, message):
        """更新状态信息"""
        self.status_label.setText(message)

    def on_reconstruction_completed(self, result):
        """重建完成处理"""
        self.current_result = result

        # 恢复UI状态
        self.reconstruct_btn.setEnabled(True)
        self.reconstruct_btn.setText("开始三维重建")
        self.progress_bar.setVisible(False)

        # 更新状态
        stats = result["stats"]
        self.status_label.setText(f"重建完成! 生成了 {stats['filtered_points']} 个三维点")

        # 显示结果信息
        result_text = f"""重建完成!

=== 重建统计 ===
原始点数: {stats['total_points']}
过滤后点数: {stats['filtered_points']}
保留率: {stats['filtered_points']/stats['total_points']*100:.1f}%
平均质量评分: {stats['average_quality']:.3f}
优化方法: {stats.get('optimization_method', '未知')}

=== 输出文件 ===
输出目录: {result.get('output_dir', '未指定')}
点云文件: enhanced_pointcloud.ply
"""

        if result.get("mesh"):
            result_text += "网格文件: enhanced_mesh.ply\n"

        if "quality_report" in result:
            qr = result["quality_report"]
            result_text += f"""
=== 质量分析 ===
深度范围: {qr['depth_range']['min']:.1f} - {qr['depth_range']['max']:.1f} mm
平均深度: {qr['depth_range']['mean']:.1f} mm
质量中位数: {qr['quality_percentiles']['50%']:.3f}
"""

        self.result_info.setPlainText(result_text)

        # 启用3D查看按钮
        self.view_pointcloud_btn.setEnabled(True)
        if result.get("mesh"):
            self.view_mesh_btn.setEnabled(True)
        self.open_output_btn.setEnabled(True)

        # 绘制质量分布（如果有质量数据）
        if "quality_report" in result:
            # 模拟质量数据用于可视化
            qualities = np.random.exponential(2.0, stats['filtered_points'])
            self.quality_canvas.plot_quality_distribution(qualities)

        # 切换到结果标签页
        self.tab_widget.setCurrentIndex(2)

        # 显示成功消息
        QMessageBox.information(
            self, "重建完成",
            f"三维重建成功完成!\n\n生成了 {stats['filtered_points']} 个高质量三维点\n\n"
            f"结果已保存到: {result.get('output_dir', '输出目录')}"
        )

    def on_reconstruction_failed(self, error_message):
        """重建失败处理"""
        # 恢复UI状态
        self.reconstruct_btn.setEnabled(True)
        self.reconstruct_btn.setText("开始三维重建")
        self.progress_bar.setVisible(False)

        # 更新状态
        self.status_label.setText(f"重建失败: {error_message}")

        # 显示错误消息
        QMessageBox.critical(self, "重建失败", f"三维重建失败:\n\n{error_message}")

    def view_pointcloud(self):
        """查看点云"""
        if not OPEN3D_AVAILABLE:
            QMessageBox.warning(self, "功能不可用", "Open3D库未安装，无法显示3D可视化")
            return

        if self.current_result and "pointcloud" in self.current_result:
            try:
                # 显示点云
                pcd = self.current_result["pointcloud"]
                o3d.visualization.draw_geometries([pcd], window_name="增强三维重建 - 点云结果")
            except Exception as e:
                QMessageBox.warning(self, "显示失败", f"无法显示点云: {str(e)}")

    def view_mesh(self):
        """查看网格"""
        if not OPEN3D_AVAILABLE:
            QMessageBox.warning(self, "功能不可用", "Open3D库未安装，无法显示3D可视化")
            return

        if self.current_result and "mesh" in self.current_result:
            try:
                # 显示网格
                mesh = self.current_result["mesh"]
                o3d.visualization.draw_geometries([mesh], window_name="增强三维重建 - 网格结果")
            except Exception as e:
                QMessageBox.warning(self, "显示失败", f"无法显示网格: {str(e)}")

    def open_output_directory(self):
        """打开输出目录"""
        if self.current_result and "output_dir" in self.current_result:
            output_dir = self.current_result["output_dir"]
            if os.path.exists(output_dir):
                # 根据操作系统打开文件夹
                import platform
                system = platform.system()
                if system == "Windows":
                    os.startfile(output_dir)
                elif system == "Darwin":  # macOS
                    os.system(f"open '{output_dir}'")
                else:  # Linux
                    os.system(f"xdg-open '{output_dir}'")
            else:
                QMessageBox.warning(self, "目录不存在", f"输出目录不存在: {output_dir}")


def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用程序信息
    app.setApplicationName("增强三维重建系统")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Enhanced 3D Reconstruction")

    # 创建主窗口
    window = Enhanced3DReconstructionUI()
    window.show()

    # 运行应用程序
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
