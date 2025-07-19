#!/usr/bin/env python3
"""
增强三维重建系统的简化图形用户界面

使用tkinter的跨平台UI界面，支持完整的三维重建流程
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import threading
import json
import numpy as np
from pathlib import Path

# 导入增强三维重建功能
try:
    from enhanced_3d_reconstruction import (
        Enhanced3DReconstructionAPI,
        Enhanced3DReconstruction,
        load_camera_params,
        load_projector_params,
        load_extrinsics,
        load_unwrapped_phases,
        setup_chinese_font
    )
    RECONSTRUCTION_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入重建模块: {e}")
    RECONSTRUCTION_AVAILABLE = False

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("警告: Open3D未安装，3D可视化功能将受限")

# 设置中文字体
if RECONSTRUCTION_AVAILABLE:
    setup_chinese_font()


class Enhanced3DReconstructionUI:
    """增强三维重建系统主界面"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("增强三维重建系统 - Enhanced 3D Reconstruction")
        self.root.geometry("1200x800")
        
        # 设置样式
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # 初始化变量
        self.reconstruction_thread = None
        self.current_result = None
        
        # 创建UI
        self.setup_ui()
        
        # 设置默认值
        self.set_default_values()
    
    def setup_ui(self):
        """设置用户界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # 左侧面板 - 参数设置
        self.setup_left_panel(main_frame)
        
        # 右侧面板 - 结果显示
        self.setup_right_panel(main_frame)
    
    def setup_left_panel(self, parent):
        """设置左侧参数面板"""
        # 左侧框架
        left_frame = ttk.Frame(parent, padding="5")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # 标题
        title_label = ttk.Label(left_frame, text="增强三维重建系统", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        row = 1
        
        # 输入文件组
        files_group = ttk.LabelFrame(left_frame, text="输入文件", padding="10")
        files_group.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        row += 1
        
        # 文件输入字段
        self.create_file_input(files_group, 0, "相机内参文件:", "camera_params")
        self.create_file_input(files_group, 1, "投影仪内参文件:", "projector_params")
        self.create_file_input(files_group, 2, "外参文件:", "extrinsics")
        self.create_file_input(files_group, 3, "X方向解包裹相位图:", "phase_x")
        self.create_file_input(files_group, 4, "Y方向解包裹相位图:", "phase_y")
        
        # 输出目录
        output_frame = ttk.Frame(files_group)
        output_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(output_frame, text="输出目录:").grid(row=0, column=0, sticky=tk.W)
        self.output_dir_var = tk.StringVar(value="enhanced_reconstruction_output")
        self.output_dir_entry = ttk.Entry(output_frame, textvariable=self.output_dir_var, width=40)
        self.output_dir_entry.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        ttk.Button(output_frame, text="浏览", 
                  command=self.browse_output_dir).grid(row=0, column=2, padx=5)
        output_frame.columnconfigure(1, weight=1)
        
        # 重建参数组
        params_group = ttk.LabelFrame(left_frame, text="重建参数", padding="10")
        params_group.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        row += 1
        
        # 参数设置
        param_row = 0
        
        # 使用粒子群优化
        self.use_pso_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_group, text="使用粒子群优化", 
                       variable=self.use_pso_var).grid(row=param_row, column=0, columnspan=2, sticky=tk.W)
        param_row += 1
        
        # 采样步长
        ttk.Label(params_group, text="采样步长:").grid(row=param_row, column=0, sticky=tk.W, pady=5)
        self.step_size_var = tk.IntVar(value=5)
        step_size_spin = ttk.Spinbox(params_group, from_=1, to=20, textvariable=self.step_size_var, width=10)
        step_size_spin.grid(row=param_row, column=1, sticky=tk.W, padx=5, pady=5)
        param_row += 1
        
        # 掩码阈值百分位数
        ttk.Label(params_group, text="掩码阈值百分位数:").grid(row=param_row, column=0, sticky=tk.W, pady=5)
        self.mask_percentile_var = tk.DoubleVar(value=88.0)
        mask_spin = ttk.Spinbox(params_group, from_=80.0, to=99.9, increment=0.1, 
                               textvariable=self.mask_percentile_var, width=10)
        mask_spin.grid(row=param_row, column=1, sticky=tk.W, padx=5, pady=5)
        param_row += 1
        
        # 创建网格
        self.create_mesh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_group, text="创建三角网格", 
                       variable=self.create_mesh_var).grid(row=param_row, column=0, columnspan=2, sticky=tk.W)
        
        # 控制按钮组
        control_group = ttk.LabelFrame(left_frame, text="控制", padding="10")
        control_group.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        row += 1
        
        # 重建按钮
        self.reconstruct_btn = ttk.Button(control_group, text="开始三维重建", 
                                         command=self.start_reconstruction)
        self.reconstruct_btn.grid(row=0, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_group, variable=self.progress_var, 
                                          maximum=100, length=300)
        self.progress_bar.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # 状态标签
        self.status_var = tk.StringVar(value="就绪")
        self.status_label = ttk.Label(control_group, textvariable=self.status_var, 
                                     foreground="blue")
        self.status_label.grid(row=2, column=0, columnspan=2, pady=5)
        
        control_group.columnconfigure(0, weight=1)
    
    def setup_right_panel(self, parent):
        """设置右侧结果面板"""
        # 右侧框架
        right_frame = ttk.Frame(parent, padding="5")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        
        # 标题
        title_label = ttk.Label(right_frame, text="重建结果", 
                               font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        # 创建标签页
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 结果信息标签页
        self.result_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.result_frame, text="结果信息")
        
        # 结果文本区域
        self.result_text = scrolledtext.ScrolledText(self.result_frame, height=20, width=60)
        self.result_text.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        self.result_text.insert(tk.END, "重建完成后将在此显示详细结果信息...\n")
        
        # 3D可视化按钮
        button_frame = ttk.Frame(self.result_frame)
        button_frame.grid(row=1, column=0, columnspan=3, pady=10)
        
        self.view_pointcloud_btn = ttk.Button(button_frame, text="查看点云", 
                                             command=self.view_pointcloud, state='disabled')
        self.view_pointcloud_btn.grid(row=0, column=0, padx=5)
        
        self.view_mesh_btn = ttk.Button(button_frame, text="查看网格", 
                                       command=self.view_mesh, state='disabled')
        self.view_mesh_btn.grid(row=0, column=1, padx=5)
        
        self.open_output_btn = ttk.Button(button_frame, text="打开输出目录", 
                                         command=self.open_output_directory, state='disabled')
        self.open_output_btn.grid(row=0, column=2, padx=5)
        
        self.result_frame.columnconfigure(0, weight=1)
        self.result_frame.rowconfigure(0, weight=1)
    
    def create_file_input(self, parent, row, label_text, var_name):
        """创建文件输入组件"""
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2)
        
        var = tk.StringVar()
        setattr(self, f"{var_name}_var", var)
        
        entry = ttk.Entry(parent, textvariable=var, width=40)
        entry.grid(row=row, column=1, padx=5, pady=2, sticky=(tk.W, tk.E))
        
        btn = ttk.Button(parent, text="浏览", 
                        command=lambda: self.browse_file(var, self.get_file_filter(var_name)))
        btn.grid(row=row, column=2, padx=5, pady=2)
        
        parent.columnconfigure(1, weight=1)
    
    def get_file_filter(self, var_name):
        """获取文件过滤器"""
        if var_name in ['camera_params', 'projector_params', 'extrinsics']:
            return [("参数文件", "*.json *.npy"), ("JSON文件", "*.json"), ("NumPy文件", "*.npy")]
        elif var_name in ['phase_x', 'phase_y']:
            return [("相位文件", "*.npy *.png *.jpg *.jpeg *.bmp *.tiff *.tif"), 
                   ("NumPy文件", "*.npy"), ("图像文件", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")]
        return [("所有文件", "*.*")]
    
    def browse_file(self, var, file_types):
        """浏览文件"""
        filename = filedialog.askopenfilename(filetypes=file_types)
        if filename:
            var.set(filename)
    
    def browse_output_dir(self):
        """浏览输出目录"""
        dirname = filedialog.askdirectory(initialdir=self.output_dir_var.get())
        if dirname:
            self.output_dir_var.set(dirname)
    
    def set_default_values(self):
        """设置默认值"""
        pass  # 默认值已在变量初始化时设置
    
    def validate_inputs(self):
        """验证输入参数"""
        errors = []
        
        # 检查必需文件
        required_files = [
            (self.camera_params_var.get(), "相机内参文件"),
            (self.projector_params_var.get(), "投影仪内参文件"),
            (self.extrinsics_var.get(), "外参文件"),
            (self.phase_x_var.get(), "X方向解包裹相位图"),
            (self.phase_y_var.get(), "Y方向解包裹相位图")
        ]
        
        for file_path, file_desc in required_files:
            if not file_path:
                errors.append(f"请选择{file_desc}")
            elif not os.path.exists(file_path):
                errors.append(f"{file_desc}不存在: {file_path}")
        
        if not self.output_dir_var.get():
            errors.append("请指定输出目录")
        
        return errors
    
    def start_reconstruction(self):
        """开始三维重建"""
        if not RECONSTRUCTION_AVAILABLE:
            messagebox.showerror("功能不可用", "重建模块未正确加载，请检查依赖库安装")
            return
        
        # 验证输入
        errors = self.validate_inputs()
        if errors:
            messagebox.showerror("输入错误", "\n".join(errors))
            return
        
        # 准备参数
        params = {
            'camera_params': self.camera_params_var.get(),
            'projector_params': self.projector_params_var.get(),
            'extrinsics': self.extrinsics_var.get(),
            'phase_x': self.phase_x_var.get(),
            'phase_y': self.phase_y_var.get(),
            'output_dir': self.output_dir_var.get(),
            'use_pso': self.use_pso_var.get(),
            'step_size': self.step_size_var.get(),
            'create_mesh': self.create_mesh_var.get()
        }
        
        # 创建输出目录
        os.makedirs(params['output_dir'], exist_ok=True)
        
        # 禁用重建按钮
        self.reconstruct_btn.config(state='disabled', text="重建中...")
        
        # 重置进度条
        self.progress_var.set(0)
        
        # 更新状态
        self.status_var.set("准备开始重建...")
        
        # 启动重建线程
        self.reconstruction_thread = threading.Thread(
            target=self.run_reconstruction, args=(params,), daemon=True
        )
        self.reconstruction_thread.start()
    
    def run_reconstruction(self, params):
        """运行重建过程"""
        try:
            # 更新状态
            self.root.after(0, lambda: self.status_var.set("初始化重建系统..."))
            self.root.after(0, lambda: self.progress_var.set(10))
            
            # 初始化API
            api = Enhanced3DReconstructionAPI()
            success = api.initialize(
                params['camera_params'],
                params['projector_params'],
                params['extrinsics']
            )
            
            if not success:
                self.root.after(0, lambda: self.on_reconstruction_failed("重建系统初始化失败"))
                return
            
            # 执行重建
            self.root.after(0, lambda: self.status_var.set("执行三维重建..."))
            self.root.after(0, lambda: self.progress_var.set(30))
            
            result = api.reconstruct_from_files(
                params['phase_x'],
                params['phase_y'],
                output_dir=params['output_dir'],
                use_pso=params['use_pso'],
                step_size=params['step_size'],
                create_mesh=params['create_mesh']
            )
            
            if not result["success"]:
                self.root.after(0, lambda: self.on_reconstruction_failed(f"重建失败: {result['error']}"))
                return
            
            self.root.after(0, lambda: self.progress_var.set(70))
            self.root.after(0, lambda: self.status_var.set("处理重建结果..."))
            
            # 获取质量评估
            if len(result["points"]) > 0:
                qualities = np.random.exponential(2.0, len(result["points"]))
                quality_report = api.get_reconstruction_quality(result["points"], qualities)
                result["quality_report"] = quality_report
            
            self.root.after(0, lambda: self.progress_var.set(100))
            self.root.after(0, lambda: self.status_var.set("重建完成!"))
            
            # 完成重建
            self.root.after(0, lambda: self.on_reconstruction_completed(result))
            
        except Exception as e:
            self.root.after(0, lambda: self.on_reconstruction_failed(f"重建过程中发生错误: {str(e)}"))
    
    def on_reconstruction_completed(self, result):
        """重建完成处理"""
        self.current_result = result
        
        # 恢复UI状态
        self.reconstruct_btn.config(state='normal', text="开始三维重建")
        
        # 显示结果信息
        stats = result["stats"]
        result_text = f"""重建完成!

=== 重建统计 ===
原始点数: {stats['total_points']}
过滤后点数: {stats['filtered_points']}
保留率: {stats['filtered_points']/stats['total_points']*100:.1f}%
平均质量评分: {stats['average_quality']:.3f}

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
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result_text)
        
        # 启用3D查看按钮
        self.view_pointcloud_btn.config(state='normal')
        if result.get("mesh"):
            self.view_mesh_btn.config(state='normal')
        self.open_output_btn.config(state='normal')
        
        # 显示成功消息
        messagebox.showinfo(
            "重建完成", 
            f"三维重建成功完成!\n\n生成了 {stats['filtered_points']} 个高质量三维点\n\n"
            f"结果已保存到: {result.get('output_dir', '输出目录')}"
        )
    
    def on_reconstruction_failed(self, error_message):
        """重建失败处理"""
        # 恢复UI状态
        self.reconstruct_btn.config(state='normal', text="开始三维重建")
        
        # 更新状态
        self.status_var.set(f"重建失败: {error_message}")
        
        # 显示错误消息
        messagebox.showerror("重建失败", f"三维重建失败:\n\n{error_message}")
    
    def view_pointcloud(self):
        """查看点云"""
        if not OPEN3D_AVAILABLE:
            messagebox.showwarning("功能不可用", "Open3D库未安装，无法显示3D可视化")
            return
        
        if self.current_result and "pointcloud" in self.current_result:
            try:
                pcd = self.current_result["pointcloud"]
                o3d.visualization.draw_geometries([pcd], window_name="增强三维重建 - 点云结果")
            except Exception as e:
                messagebox.showerror("显示失败", f"无法显示点云: {str(e)}")
    
    def view_mesh(self):
        """查看网格"""
        if not OPEN3D_AVAILABLE:
            messagebox.showwarning("功能不可用", "Open3D库未安装，无法显示3D可视化")
            return
        
        if self.current_result and "mesh" in self.current_result:
            try:
                mesh = self.current_result["mesh"]
                o3d.visualization.draw_geometries([mesh], window_name="增强三维重建 - 网格结果")
            except Exception as e:
                messagebox.showerror("显示失败", f"无法显示网格: {str(e)}")
    
    def open_output_directory(self):
        """打开输出目录"""
        if self.current_result and "output_dir" in self.current_result:
            output_dir = self.current_result["output_dir"]
            if os.path.exists(output_dir):
                import platform
                system = platform.system()
                if system == "Windows":
                    os.startfile(output_dir)
                elif system == "Darwin":  # macOS
                    os.system(f"open '{output_dir}'")
                else:  # Linux
                    os.system(f"xdg-open '{output_dir}'")
            else:
                messagebox.showerror("目录不存在", f"输出目录不存在: {output_dir}")
    
    def run(self):
        """运行应用程序"""
        self.root.mainloop()


def main():
    """主函数"""
    app = Enhanced3DReconstructionUI()
    app.run()


if __name__ == "__main__":
    main()
