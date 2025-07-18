#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
投影仪标定程序 (基于三频外差相位解包裹)
独立版本 - 集成三频外差相位解包裹模块
"""

import os
import sys
import cv2
import numpy as np
import argparse
import json
from typing import List, Tuple, Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass
import matplotlib.pyplot as plt
import traceback

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("警告：未安装tqdm，将不显示进度条")
    
def visualize_phase(phase_data: np.ndarray, title: str, save_path: str, show_plots: bool, 
                    is_wrapped: bool, quality_map: Optional[np.ndarray] = None):
    """通用相位可视化函数"""
    plt.figure(figsize=(12, 9 if is_wrapped and quality_map is not None else 8))
    
    if is_wrapped and quality_map is not None:
        plt.subplot(2, 1, 1)

    img = plt.imshow(phase_data, cmap='jet')
    plt.colorbar(img, label='Phase (rad)')
    plt.title(title)

    if is_wrapped and quality_map is not None:
        plt.subplot(2, 1, 2)
        quality_img = plt.imshow(quality_map, cmap='viridis')
        plt.colorbar(quality_img, label='Quality')
        plt.title("Phase Quality Map")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()


def bilinear_interpolate(data: np.ndarray, y: float, x: float, default_value: float = 0) -> float:
    """
    双线性插值函数
    
    参数:
        data: 二维数组
        y, x: 插值坐标
        default_value: 默认值
        
    返回:
        插值结果
    """
    height, width = data.shape
    
    # 边界检查
    if x < 0 or y < 0 or x >= width-1 or y >= height-1:
        return default_value
        
    # 获取整数和小数部分
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    
    # 计算权重
    wx1 = x - x0
    wx0 = 1 - wx1
    wy1 = y - y0
    wy0 = 1 - wy1
    
    # 双线性插值
    result = (data[y0, x0] * wx0 * wy0 + 
              data[y0, x1] * wx1 * wy0 + 
              data[y1, x0] * wx0 * wy1 + 
              data[y1, x1] * wx1 * wy1)
    
    return result


def detect_calibration_board(image: np.ndarray, board_type: str, chessboard_size: Tuple[int, int], 
                           square_size: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    检测标定板角点
    
    参数:
        image: 输入图像
        board_type: 标定板类型
        chessboard_size: 棋盘格尺寸 (宽, 高)
        square_size: 方格大小(mm)
        
    返回:
        obj_points: 3D物体点
        corners: 2D图像点
    """
    # 创建3D物体点
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    if board_type == 'chessboard':
        # 检测棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            # 亚像素精度优化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return objp, corners
    
    elif board_type == 'circles':
        # 检测圆形标定板
        ret, corners = cv2.findCirclesGrid(gray, chessboard_size, None)
        if ret:
            return objp, corners
    
    elif board_type == 'ring_circles':
        # 检测环形圆标定板
        ret, corners = cv2.findCirclesGrid(gray, chessboard_size, 
                                         cv2.CALIB_CB_ASYMMETRIC_GRID)
        if ret:
            return objp, corners
    
    return None, None


def assess_calibration_quality(reprojection_error: float, board_type: str):
    """
    评估标定质量
    
    参数:
        reprojection_error: 重投影误差
        board_type: 标定板类型
    """
    print("\n【标定质量评估】")
    print(f"平均重投影误差: {reprojection_error:.4f} 像素")

    if reprojection_error < 0.5:
        quality = "极佳"
    elif reprojection_error < 1.0:
        quality = "良好"
    elif reprojection_error < 2.0:
        quality = "一般"
    else:
        quality = "较差"

    print(f"标定质量: {quality}")

    if quality == "较差":
        print("\n【改进建议】")
        if board_type == 'chessboard':
            print("- 检查棋盘格是否平整无变形")
            print("- 尝试在更均匀的光照条件下拍摄")
        elif board_type == 'circles':
            print("- 检查圆点是否清晰可见")
            print("- 尝试调整照明减少反光")
        elif board_type == 'ring_circles':
            print("- 确保圆环闭合且形状规则")
            print("- 考虑增加图像对比度")
        
        print("- 尝试增加标定图像数量，覆盖更多角度和位置")
        print("- 确保标定板在图像中清晰可见且无模糊")

def visualize_phase(phase_data: np.ndarray, title: str, save_path: str, show_plots: bool, 
                    is_wrapped: bool, quality_map: Optional[np.ndarray] = None):
    """通用相位可视化函数"""
    plt.figure(figsize=(12, 9 if is_wrapped and quality_map is not None else 8))
    
    if is_wrapped and quality_map is not None:
        plt.subplot(2, 1, 1)

    img = plt.imshow(phase_data, cmap='jet')
    plt.colorbar(img, label='Phase (rad)')
    plt.title(title)

    if is_wrapped and quality_map is not None:
        plt.subplot(2, 1, 2)
        quality_img = plt.imshow(quality_map, cmap='viridis')
        plt.colorbar(quality_img, label='Quality')
        plt.title("Phase Quality Map")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()


class multi_phase:
    """
    三频相位解包裹类
    集成自 get_abs_phase.py
    """
    
    def __init__(self, f: List[int], step: int, images: List[np.ndarray], ph0: float = 0.5):
        """
        初始化三频相位解包裹
        
        参数:
            f: 频率列表 [高频, 中频, 低频]
            step: 相移步数
            images: 图像列表 (24张图像)
            ph0: 初始相位偏移
        """
        self.f = f  # 频率列表
        self.step = step  # 相移步数
        self.images = images  # 图像列表
        self.ph0 = ph0  # 初始相位偏移
        
        # 验证输入参数
        if len(f) != 3:
            raise ValueError("必须提供3个频率值")
        if len(images) != 24:
            raise ValueError(f"需要24张图像，但提供了{len(images)}张")
        if step != 4:
            raise ValueError("当前只支持4步相移")
    
    def get_phase(self):
        """
        执行三频相位解包裹
        
        返回:
            unwrapped_vertical: 垂直方向解包裹相位
            unwrapped_horizontal: 水平方向解包裹相位
            quality_map: 相位质量图
        """
        try:
            # 组织图像数据
            h_images = {
                'high': self.images[0:4],    # 水平高频
                'mid': self.images[4:8],     # 水平中频
                'low': self.images[8:12]     # 水平低频
            }
            
            v_images = {
                'high': self.images[12:16],  # 垂直高频
                'mid': self.images[16:20],   # 垂直中频
                'low': self.images[20:24]    # 垂直低频
            }
            
            # 计算水平方向相位
            h_wrapped_phases = {}
            for freq_name, imgs in h_images.items():
                h_wrapped_phases[freq_name] = self._compute_wrapped_phase(imgs)
            
            # 计算垂直方向相位
            v_wrapped_phases = {}
            for freq_name, imgs in v_images.items():
                v_wrapped_phases[freq_name] = self._compute_wrapped_phase(imgs)
            
            # 执行三频解包裹
            unwrapped_h = self._three_freq_unwrap(
                h_wrapped_phases['high'], 
                h_wrapped_phases['mid'], 
                h_wrapped_phases['low'],
                self.f
            )
            
            unwrapped_v = self._three_freq_unwrap(
                v_wrapped_phases['high'], 
                v_wrapped_phases['mid'], 
                v_wrapped_phases['low'],
                self.f
            )
            
            # 计算质量图
            quality_map = self._compute_quality_map(h_images['high'], v_images['high'])
            
            return unwrapped_v, unwrapped_h, quality_map
            
        except Exception as e:
            raise RuntimeError(f"三频相位解包裹失败: {e}")
    
    def _compute_wrapped_phase(self, images: List[np.ndarray]) -> np.ndarray:
        """
        计算包裹相位 (4步相移算法)
        
        参数:
            images: 4张相移图像
            
        返回:
            wrapped_phase: 包裹相位 (-π到π)
        """
        if len(images) != 4:
            raise ValueError(f"4步相移需要4张图像，但提供了{len(images)}张")
        
        # 转换为浮点数
        I1, I2, I3, I4 = [img.astype(np.float32) for img in images]
        
        # 4步相移算法
        # I1: 0°, I2: 90°, I3: 180°, I4: 270°
        numerator = I4 - I2
        denominator = I1 - I3
        
        # 避免除零
        denominator = np.where(np.abs(denominator) < 1e-6, 1e-6, denominator)
        
        # 计算包裹相位
        wrapped_phase = np.arctan2(numerator, denominator)
        
        return wrapped_phase
    
    def _three_freq_unwrap(self, phase_high: np.ndarray, phase_mid: np.ndarray, 
                          phase_low: np.ndarray, frequencies: List[int]) -> np.ndarray:
        """
        三频外差相位解包裹
        
        参数:
            phase_high: 高频包裹相位
            phase_mid: 中频包裹相位
            phase_low: 低频包裹相位
            frequencies: 频率列表 [高频, 中频, 低频]
            
        返回:
            unwrapped_phase: 解包裹后的相位
        """
        f_high, f_mid, f_low = frequencies
        
        # 计算频率差
        f12 = f_high - f_mid  # 高频与中频之差
        f23 = f_mid - f_low   # 中频与低频之差
        
        # 计算相位差
        phase_12 = phase_high - phase_mid
        phase_23 = phase_mid - phase_low
        
        # 包裹相位差到 [-π, π]
        phase_12 = np.arctan2(np.sin(phase_12), np.cos(phase_12))
        phase_23 = np.arctan2(np.sin(phase_23), np.cos(phase_23))
        
        # 使用低频相位差作为参考解包裹中频相位差
        unwrapped_phase_12 = self._unwrap_with_reference(phase_12, phase_23, f12, f23)
        
        # 使用解包裹的中频相位差解包裹高频相位
        unwrapped_phase = self._unwrap_with_reference(phase_high, unwrapped_phase_12, f_high, f12)
        
        return unwrapped_phase
    
    def _unwrap_with_reference(self, phase: np.ndarray, reference: np.ndarray, 
                              phase_f: int, reference_f: int) -> np.ndarray:
        """
        使用参考相位解包裹目标相位
        
        参数:
            phase: 待解包裹的包裹相位
            reference: 参考相位
            phase_f: 目标相位的频率
            reference_f: 参考相位的频率
            
        返回:
            unwrapped_phase: 解包裹后的相位
        """
        # 根据频率比例缩放参考相位
        temp = phase_f / reference_f * reference
        
        # 计算整数条纹序数k并应用
        k = np.round(temp - phase)
        unwrapped_phase = phase + k * 2 * np.pi
        
        # 高斯滤波去噪，检测错误跳变点
        gauss_size = (3, 3)
        unwrapped_phase_noise = unwrapped_phase - cv2.GaussianBlur(unwrapped_phase, gauss_size, 0)
        unwrapped_reference_noise = temp - cv2.GaussianBlur(temp, gauss_size, 0)

        # 改进异常点检测
        noise_ratio = np.abs(unwrapped_phase_noise) / (np.abs(unwrapped_reference_noise) + 0.001)
        order_flag = (np.abs(unwrapped_phase_noise) - np.abs(unwrapped_reference_noise) > 0.15) & (noise_ratio > 1.5)
        
        # 修正异常点
        unwrapped_phase[order_flag] = temp[order_flag]
        
        return unwrapped_phase
    
    def _compute_quality_map(self, h_images: List[np.ndarray], v_images: List[np.ndarray]) -> np.ndarray:
        """
        计算相位质量图
        
        参数:
            h_images: 水平方向图像列表
            v_images: 垂直方向图像列表
            
        返回:
            quality_map: 相位质量图
        """
        # 计算水平和垂直方向的调制度
        h_modulation = self._compute_modulation(h_images)
        v_modulation = self._compute_modulation(v_images)
        
        # 综合质量图
        quality_map = (h_modulation + v_modulation) / 2.0
        
        return quality_map
    
    def _compute_modulation(self, images: List[np.ndarray]) -> np.ndarray:
        """
        计算调制度
        
        参数:
            images: 4张相移图像
            
        返回:
            modulation: 调制度图
        """
        I1, I2, I3, I4 = [img.astype(np.float32) for img in images]
        
        # 计算平均强度和调制强度
        I_avg = (I1 + I2 + I3 + I4) / 4.0
        I_mod = np.sqrt((I4 - I2)**2 + (I1 - I3)**2) / 2.0
        
        # 计算调制度
        modulation = I_mod / (I_avg + 1e-6)  # 避免除零
        
        # 归一化
        modulation = np.clip(modulation / np.max(modulation), 0, 1)
        
        return modulation


# 异常类定义
class PhaseUnwrappingError(Exception):
    """相位解包裹错误"""
    pass

class BoardDetectionError(Exception):
    """标定板检测错误"""
    pass

class CorrespondenceError(Exception):
    """对应关系建立错误"""
    pass

class CalibrationError(Exception):
    """标定错误"""
    pass

@dataclass
class ThreeFreqCalibrationConfig:
    """三频外差标定配置类"""
    frequencies: List[int]
    phase_step: int
    ph0: float
    projector_width: int
    projector_height: int
    quality_threshold: float = 0.3

class ProjectorCalibration:
    """投影仪标定类"""
    
    def __init__(self):
        self.projector_matrix = None
        self.projector_dist = None
        self.R = None
        self.T = None
        self.reprojection_error = None
    
    def calibrate_projector_with_camera(self, camera_matrix, camera_distortion,
                                      proj_cam_correspondences, board_points):
        """
        使用相机参数标定投影仪
        
        参数:
            camera_matrix: 相机内参矩阵
            camera_distortion: 相机畸变系数
            proj_cam_correspondences: 投影仪-相机对应点列表
            board_points: 标定板角点的世界坐标
            
        返回:
            reprojection_error: 重投影误差
            calibration_data: 标定数据字典
        """
        # 提取投影仪点和相机点
        projector_points = []
        camera_points = []
        object_points = []
        
        for corr in proj_cam_correspondences:
            projector_points.append(corr['projector_point'])
            camera_points.append(corr['camera_point'])
            object_points.append(board_points[corr['board_index']])
        
        # 转换为numpy数组
        projector_points = np.array(projector_points, dtype=np.float32)
        camera_points = np.array(camera_points, dtype=np.float32)
        object_points = np.array(object_points, dtype=np.float32)
        
        # 重新组织数据为OpenCV格式
        object_points_list = []
        projector_points_list = []
        camera_points_list = []
        
        # 按姿态分组数据
        points_per_pose = {}
        for i, corr in enumerate(proj_cam_correspondences):
            pose_id = i // len(set(range(len(proj_cam_correspondences))))  # 简化的姿态分组
            if pose_id not in points_per_pose:
                points_per_pose[pose_id] = {'obj': [], 'proj': [], 'cam': []}
            
            points_per_pose[pose_id]['obj'].append(board_points[corr['board_index']])
            points_per_pose[pose_id]['proj'].append(corr['projector_point'])
            points_per_pose[pose_id]['cam'].append(corr['camera_point'])
        
        # 转换为OpenCV格式
        for pose_id in points_per_pose:
            object_points_list.append(np.array(points_per_pose[pose_id]['obj'], dtype=np.float32))
            projector_points_list.append(np.array(points_per_pose[pose_id]['proj'], dtype=np.float32))
            camera_points_list.append(np.array(points_per_pose[pose_id]['cam'], dtype=np.float32))
        
        # 执行立体标定
        try:
            # 初始化投影仪内参
            projector_matrix_init = camera_matrix.copy()
            projector_dist_init = np.zeros((5,), dtype=np.float32)
            
            # 立体标定
            ret, camera_matrix_new, camera_dist_new, projector_matrix_new, projector_dist_new, R, T, E, F = cv2.stereoCalibrate(
                object_points_list,
                camera_points_list,
                projector_points_list,
                camera_matrix,
                camera_distortion,
                projector_matrix_init,
                projector_dist_init,
                (projector_points[0].shape[0], projector_points[0].shape[1]) if len(projector_points) > 0 else (640, 480),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
                flags=cv2.CALIB_FIX_INTRINSIC
            )
            
            # 保存结果
            self.projector_matrix = projector_matrix_new
            self.projector_dist = projector_dist_new
            self.R = R
            self.T = T
            self.reprojection_error = ret
            
            calibration_data = {
                'projector_matrix': projector_matrix_new,
                'projector_dist': projector_dist_new,
                'R': R,
                'T': T,
                'reprojection_error': ret
            }
            
            return ret, calibration_data
            
        except Exception as e:
            raise CalibrationError(f"立体标定失败: {e}")
    
    def save_calibration(self, filename: str):
        """
        保存标定结果
        
        参数:
            filename: 保存文件名
        """
        if filename.endswith('.npz'):
            # 保存为NPZ格式
            np.savez(filename,
                    projector_matrix=self.projector_matrix,
                    projector_dist=self.projector_dist,
                    R=self.R,
                    T=self.T,
                    reprojection_error=self.reprojection_error)
        elif filename.endswith('.json'):
            # 保存为JSON格式
            data = {
                'projector_matrix': self.projector_matrix.tolist() if self.projector_matrix is not None else None,
                'projector_dist': self.projector_dist.tolist() if self.projector_dist is not None else None,
                'R': self.R.tolist() if self.R is not None else None,
                'T': self.T.tolist() if self.T is not None else None,
                'reprojection_error': float(self.reprojection_error) if self.reprojection_error is not None else None
            }
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError("不支持的文件格式，请使用.npz或.json")

def load_camera_parameters(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载相机标定参数
    
    参数:
        filename: 参数文件路径
        
    返回:
        camera_matrix: 相机内参矩阵
        camera_distortion: 相机畸变系数
    """
    if filename.endswith('.npz'):
        data = np.load(filename)
        camera_matrix = data['camera_matrix']
        camera_distortion = data['dist_coeffs']
    elif filename.endswith('.json'):
        with open(filename, 'r') as f:
            data = json.load(f)
        camera_matrix = np.array(data['camera_matrix'])
        camera_distortion = np.array(data['dist_coeffs'])
    else:
        raise ValueError("不支持的文件格式")
    
    return camera_matrix, camera_distortion

def organize_three_freq_images(image_paths: List[str], config: ThreeFreqCalibrationConfig) -> Dict[str, List[str]]:
    """
    组织三频图像路径
    
    参数:
        image_paths: 24张图像路径列表 (I1-I24)
        config: 三频标定配置
        
    返回:
        organized_paths: 按频率和方向组织的路径字典
    """
    if len(image_paths) != 24:
        raise ValueError(f"需要24张图像，但提供了{len(image_paths)}张")
    
    # 按照固定顺序组织图像路径
    organized_paths = {
        'horizontal_high': image_paths[0:4],    # I1-I4: 水平高频
        'horizontal_mid': image_paths[4:8],     # I5-I8: 水平中频  
        'horizontal_low': image_paths[8:12],    # I9-I12: 水平低频
        'vertical_high': image_paths[12:16],    # I13-I16: 垂直高频
        'vertical_mid': image_paths[16:20],     # I17-I20: 垂直中频
        'vertical_low': image_paths[20:24]      # I21-I24: 垂直低频
    }
    
    return organized_paths

def process_three_freq_phase_unwrapping(organized_paths: Dict[str, List[str]], 
                                       config: ThreeFreqCalibrationConfig,
                                       output_dir: str = None,
                                       visualize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用三频相位解包裹法进行相位解包裹
    
    参数:
        organized_paths: 按频率和方向组织的路径字典
        config: 三频标定配置
        output_dir: 输出目录
        visualize: 是否可视化结果
        
    返回:
        unwrapped_vertical: 垂直方向解包裹相位
        unwrapped_horizontal: 水平方向解包裹相位  
        quality_map: 相位质量图
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 读取所有图像
    all_images = []
    for key in ['horizontal_high', 'horizontal_mid', 'horizontal_low',
                'vertical_high', 'vertical_mid', 'vertical_low']:
        for path in organized_paths[key]:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"无法加载图像: {path}")
            all_images.append(img)
    
    print(f"成功加载 {len(all_images)} 张三频相移图像")
    
    # 设置三频参数
    fx = config.frequencies  # 水平方向频率
    fy = config.frequencies  # 垂直方向频率
    
    try:
        # 创建三频处理对象
        phase_processor = multi_phase(f=fx, step=config.phase_step, images=all_images, ph0=config.ph0)
        
        # 执行三频相位解包裹
        print("正在执行三频相位解包裹...")
        result = phase_processor.get_phase()
        
        # 检查返回值数量
        if len(result) >= 3:
            unwrapped_vertical, unwrapped_horizontal, quality_map = result[0], result[1], result[2]
        else:
            raise ValueError("三频处理返回值数量不足")
            
        print("三频相位解包裹完成")
        
    except Exception as e:
        raise PhaseUnwrappingError(f"三频外差相位解包裹失败: {e}")
    
    # 可视化结果
    if visualize and output_dir:
        try:
            # 保存解包裹相位图
            visualize_phase(unwrapped_vertical, "Three-Freq Unwrapped Phase (Vertical)", 
                           os.path.join(output_dir, "three_freq_unwrapped_vertical.png"), 
                           True, False)
            
            visualize_phase(unwrapped_horizontal, "Three-Freq Unwrapped Phase (Horizontal)", 
                           os.path.join(output_dir, "three_freq_unwrapped_horizontal.png"), 
                           True, False)
            
            # 保存质量图
            plt.figure(figsize=(10, 8))
            plt.imshow(quality_map, cmap='viridis')
            plt.title('Three-Frequency Phase Quality Map')
            plt.colorbar()
            plt.savefig(os.path.join(output_dir, "three_freq_quality_map.png"), dpi=300, bbox_inches='tight')
            if visualize:
                plt.show()
            plt.close()
        except Exception as e:
            print(f"警告：可视化过程出错: {e}")
    
    return unwrapped_vertical, unwrapped_horizontal, quality_map


def load_camera_parameters(camera_params_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载相机标定参数，支持多种格式
    
    参数:
        camera_params_file: 相机参数文件路径
        
    返回:
        camera_matrix: 相机内参矩阵
        camera_distortion: 相机畸变系数
    """
    try:
        if camera_params_file.endswith('.npz'):
            camera_data = np.load(camera_params_file)
            camera_matrix = camera_data['camera_matrix']
            camera_distortion = camera_data['camera_distortion']
        elif camera_params_file.endswith('.json'):
            import json
            with open(camera_params_file, 'r') as f:
                camera_data = json.load(f)
            camera_matrix = np.array(camera_data['camera_matrix'])
            camera_distortion = np.array(camera_data['camera_distortion'])
        else:
            raise ValueError("不支持的文件格式，请使用.npz或.json格式")
        
        # 验证参数格式
        if camera_matrix.shape != (3, 3):
            raise ValueError("相机内参矩阵格式错误")
        if camera_distortion.size < 4:
            raise ValueError("畸变系数数量不足")
            
        return camera_matrix, camera_distortion
        
    except Exception as e:
        raise ValueError(f"加载相机参数失败: {e}")


def three_freq_projector_calibration(projector_width: int, projector_height: int, 
                                    camera_params_file: str,
                                    phase_images_folder: str, 
                                    board_type: str = "chessboard", 
                                    chessboard_size: Tuple[int, int] = (9, 6),
                                    square_size: float = 20.0, 
                                    output_folder: str = None, 
                                    visualize: bool = True,
                                    frequencies: List[int] = [71, 64, 58],
                                    phase_step: int = 4,
                                    ph0: float = 0.5,
                                    quality_threshold: float = 0.3,
                                    print_func=print) -> Tuple[ProjectorCalibration, str]:
    """主标定函数 - 增强错误处理"""
    
    # 输入验证
    if not os.path.exists(camera_params_file):
        raise FileNotFoundError(f"相机标定文件不存在: {camera_params_file}")
    
    if not os.path.exists(phase_images_folder):
        raise FileNotFoundError(f"相移图像文件夹不存在: {phase_images_folder}")
    
    if len(frequencies) != 3:
        raise ValueError("必须提供3个频率值")
    
    if not (0.1 <= quality_threshold <= 1.0):
        raise ValueError("质量阈值必须在0.1-1.0之间")
    
    print_func(f"投影仪标定程序 (基于三频外差相位解包裹)")
    print_func("=" * 60)
    
    # 创建配置对象
    config = ThreeFreqCalibrationConfig(
        frequencies=frequencies,
        phase_step=phase_step,
        ph0=ph0,
        projector_width=projector_width,
        projector_height=projector_height
    )
    
    # 检查输入文件夹
    if not os.path.isdir(phase_images_folder):
        raise FileNotFoundError(f"指定的相位图像文件夹不存在: {phase_images_folder}")
    
    # 设置输出文件夹
    if output_folder is None:
        output_folder = os.path.join(phase_images_folder, "three_freq_calibration_results")
    os.makedirs(output_folder, exist_ok=True)
    
    # 读取相机标定参数
    print_func("读取相机标定参数...")
    try:
        camera_matrix, camera_distortion = load_camera_parameters(camera_params_file)
    except Exception as e:
        raise FileNotFoundError(f"读取相机标定参数失败: {e}")
    
    print_func(f"相机内参矩阵:\n{camera_matrix}")
    
    # 获取子文件夹列表
    pose_folders = [f for f in os.listdir(phase_images_folder) 
                   if os.path.isdir(os.path.join(phase_images_folder, f))]
    
    if not pose_folders:
        raise ValueError(f"在文件夹 '{phase_images_folder}' 中未找到子文件夹")
    
    print_func(f"找到 {len(pose_folders)} 个标定姿态文件夹")
    
    # 存储所有姿态的数据
    all_obj_points = []
    all_proj_points = []
    all_cam_points = []
    valid_poses_count = 0
    
    # 处理每个姿态
    for pose_name in sorted(pose_folders):
        pose_folder = os.path.join(phase_images_folder, pose_name)
        print_func(f"\n处理姿态: {pose_name}")
        
        try:
            # 获取图像文件列表
            image_files = {}
            for i in range(1, 25):  # 1-24
                for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                    img_path = os.path.join(pose_folder, f"I{i}{ext}")
                    if os.path.exists(img_path):
                        image_files[f'I{i}'] = img_path
                        break
            
            if len(image_files) != 24:
                print_func(f"  警告: 姿态 {pose_name} 中图像数量不足 (需要24张，找到{len(image_files)}张)，跳过")
                continue
            
            # 组织图像路径
            image_paths = [image_files[f'I{i}'] for i in range(1, 25)]
            organized_paths = organize_three_freq_images(image_paths, config)
            
            # 执行三频相位解包裹
            pose_output_dir = os.path.join(output_folder, f"phase_results_{pose_name}")
            unwrapped_v, unwrapped_h, quality_map = process_three_freq_phase_unwrapping(
                organized_paths, config, pose_output_dir, visualize
            )
            
            # 使用第一张垂直图像进行角点检测
            cam_img = cv2.imread(organized_paths['vertical_high'][0], cv2.IMREAD_GRAYSCALE)
            
            # 检测标定板角点
            print_func("  - 检测标定板角点...")
            obj_points_pose, cam_points_pose = detect_calibration_board(
                cam_img, board_type, chessboard_size, square_size
            )
            
            if cam_points_pose is None or len(cam_points_pose) == 0:
                raise BoardDetectionError(f"在姿态 {pose_name} 的图像中未能检测到标定板。")
            
            print_func(f"  - 成功检测到 {len(cam_points_pose)} 个角点。")
            
            # 提取投影仪中的对应点
            proj_points_pose = []
            valid_indices = []
            
            # 获取有效相位范围
            valid_mask = quality_map > quality_threshold
            if not np.any(valid_mask):
                print_func(f"  警告: 姿态 {pose_name} 中没有满足质量要求的相位点，跳过")
                continue
                
            v_valid = unwrapped_v[valid_mask]
            h_valid = unwrapped_h[valid_mask]
            v_min, v_max = np.min(v_valid), np.max(v_valid)
            h_min, h_max = np.min(h_valid), np.max(h_valid)
            
            for i, point in enumerate(cam_points_pose):
                x, y = point[0]
                
                # 检查坐标是否在图像范围内
                if 0 <= int(y) < unwrapped_v.shape[0] and 0 <= int(x) < unwrapped_v.shape[1]:
                    # 使用双线性插值获取相位值
                    phi_v = bilinear_interpolate(unwrapped_v, y, x)
                    phi_h = bilinear_interpolate(unwrapped_h, y, x)
                    quality = bilinear_interpolate(quality_map, y, x)
                    
                    # 检查质量
                    if quality > quality_threshold:
                        # 相位值到投影仪像素坐标的映射
                        px = (phi_h - h_min) / (h_max - h_min) * (projector_width - 1)
                        py = (phi_v - v_min) / (v_max - v_min) * (projector_height - 1)
                        
                        # 检查投影仪坐标是否在有效范围内
                        if 0 <= px < projector_width and 0 <= py < projector_height:
                            proj_points_pose.append([px, py])
                            valid_indices.append(i)
            
            if len(proj_points_pose) < 6:
                raise CorrespondenceError(f"姿态 {pose_name} 中有效对应点数量不足 ({len(proj_points_pose)} < 6)")
            
            print_func(f"  - 成功提取 {len(proj_points_pose)} 个有效对应点")
            
            # 添加到总数据集
            valid_obj_points = [obj_points_pose[i] for i in valid_indices]
            valid_cam_points = [cam_points_pose[i] for i in valid_indices]
            
            all_obj_points.extend(valid_obj_points)
            all_proj_points.extend(proj_points_pose)
            all_cam_points.extend(valid_cam_points)
            
            valid_poses_count += 1
            print_func(f"  - 姿态 {pose_name} 处理成功。")
            
        except (FileNotFoundError, BoardDetectionError, PhaseUnwrappingError, CorrespondenceError) as e:
            print_func(f"警告: 跳过姿态 '{pose_name}'，原因: {e}")
            continue
        except Exception as e:
            print_func(f"警告: 处理姿态 '{pose_name}' 时发生未知错误，已跳过。错误: {e}")
            traceback.print_exc()
            continue
    
    if valid_poses_count < 3:
        raise CorrespondenceError(f"未能处理足够数量的有效姿态。至少需要3个有效姿态，但只处理了 {valid_poses_count} 个。")
    
    if len(all_obj_points) < 20:
        raise CorrespondenceError(f"总对应点数量不足 ({len(all_obj_points)} < 20)")
    
    print_func(f"\n总共收集到 {len(all_obj_points)} 个对应点")
    
    # 执行投影仪标定
    print_func("执行投影仪标定...")
    calibration = ProjectorCalibration()
    
    # 准备对应关系数据
    proj_cam_correspondences = []
    for i in range(len(all_obj_points)):
        proj_cam_correspondences.append({
            'projector_point': all_proj_points[i],
            'camera_point': all_cam_points[i],
            'board_index': i
        })
    
    # 执行标定
    try:
        reprojection_error, calibration_data = calibration.calibrate_projector_with_camera(
            camera_matrix=camera_matrix,
            camera_distortion=camera_distortion,
            proj_cam_correspondences=proj_cam_correspondences,
            board_points=all_obj_points
        )
    except Exception as e:
        raise CalibrationError(f"投影仪标定失败: {e}")
    
    # 保存标定结果
    calibration_file = os.path.join(output_folder, "three_freq_projector_calibration.npz")
    try:
        calibration.save_calibration(calibration_file)
    except Exception as e:
        print_func(f"警告: 保存标定结果失败: {e}")
        # 尝试保存为JSON格式
        calibration_file = os.path.join(output_folder, "three_freq_projector_calibration.json")
        calibration.save_calibration(calibration_file)
    
    # 评估标定质量
    assess_calibration_quality(reprojection_error, board_type)
    
    print_func(f"\n三频外差投影仪标定完成！")
    print_func(f"重投影误差: {reprojection_error:.4f} 像素")
    print_func(f"标定结果已保存至: {calibration_file}")
    
    return calibration, calibration_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于三频外差法的投影仪标定程序')
    
    # 基本参数
    parser.add_argument('--projector_width', type=int, default=1024, help='投影仪宽度')
    parser.add_argument('--projector_height', type=int, default=768, help='投影仪高度')
    parser.add_argument('--camera_params', type=str, required=True, help='相机标定参数文件路径')
    parser.add_argument('--phase_images', type=str, required=True, help='相移图像文件夹路径')
    parser.add_argument('--output_folder', type=str, help='输出文件夹路径')
    
    # 标定板参数
    parser.add_argument('--board_type', type=str, default='chessboard', 
                       choices=['chessboard', 'circles', 'ring_circles'], help='标定板类型')
    parser.add_argument('--chessboard_width', type=int, default=9, help='棋盘格宽度（内角点数）')
    parser.add_argument('--chessboard_height', type=int, default=6, help='棋盘格高度（内角点数）')
    parser.add_argument('--square_size', type=float, default=20.0, help='方格尺寸(mm)')
    
    # 三频外差参数
    parser.add_argument('--frequencies', type=int, nargs=3, default=[71, 64, 58], 
                       help='三个频率值（从高到低）')
    parser.add_argument('--phase_step', type=int, default=4, help='相移步数')
    parser.add_argument('--ph0', type=float, default=0.5, help='初始相位偏移')
    parser.add_argument('--quality_threshold', type=float, default=0.3, help='相位质量阈值')
    
    # 其他参数
    parser.add_argument('--no_visualize', action='store_true', help='禁用可视化')
    
    args = parser.parse_args()
    
    try:
        # 执行三频外差投影仪标定
        calibration, calibration_file = three_freq_projector_calibration(
            projector_width=args.projector_width,
            projector_height=args.projector_height,
            camera_params_file=args.camera_params,
            phase_images_folder=args.phase_images,
            board_type=args.board_type,
            chessboard_size=(args.chessboard_width, args.chessboard_height),
            square_size=args.square_size,
            output_folder=args.output_folder,
            visualize=not args.no_visualize,
            frequencies=args.frequencies,
            phase_step=args.phase_step,
            ph0=args.ph0,
            quality_threshold=args.quality_threshold
        )
        
        print(f"\n三频外差投影仪标定完成，结果已保存至: {calibration_file}")
        
        # 显示投影仪与相机之间的位姿关系
        print("\n投影仪与相机的位姿关系:")
        print("旋转矩阵 (从投影仪到相机):")
        print(calibration.R)
        print("\n平移向量 (从投影仪到相机，单位:mm):")
        print(calibration.T)
        
    except Exception as e:
        print(f"\n标定失败: {e}")
        traceback.print_exc()



if __name__ == "__main__":
    main()






