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


def configure_detection_parameters(board_type: str) -> Dict[str, Any]:
    """
    根据标定板类型配置特定的检测参数

    参数:
        board_type: 标定板类型

    返回:
        params: 包含检测参数的字典
    """
    params = {}

    if board_type == 'chessboard':
        params['criteria'] = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        params['flags'] = None
    elif board_type == 'circles':
        # 白底黑圆参数
        blob_params = cv2.SimpleBlobDetector_Params()
        blob_params.filterByArea = True
        blob_params.minArea = 50
        blob_params.maxArea = 5000
        params['detector'] = cv2.SimpleBlobDetector_create(blob_params)
        params['flags'] = cv2.CALIB_CB_SYMMETRIC_GRID
    elif board_type == 'ring_circles':
        # 白底空心圆参数
        blob_params = cv2.SimpleBlobDetector_Params()
        blob_params.filterByArea = True
        blob_params.minArea = 50
        blob_params.maxArea = 5000
        blob_params.filterByCircularity = True
        blob_params.minCircularity = 0.7
        blob_params.filterByConvexity = True
        blob_params.minConvexity = 0.8
        blob_params.filterByInertia = True
        blob_params.minInertiaRatio = 0.7
        params['detector'] = cv2.SimpleBlobDetector_create(blob_params)
        params['flags'] = cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING

    return params

def preprocess_image_for_board(image: np.ndarray, board_type: str) -> np.ndarray:
    """
    根据标定板类型优化图像预处理

    参数:
        image: 输入图像
        board_type: 标定板类型

    返回:
        处理后的灰度图像
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if board_type == 'chessboard':
        # 基本处理，提高对比度
        gray = cv2.equalizeHist(gray)
    elif board_type == 'circles':
        # 增强圆形检测
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    elif board_type == 'ring_circles':
        # 空心圆环特殊处理
        gray = cv2.bitwise_not(gray)  # 反转图像使圆环区域为暗色
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

    return gray

def detect_calibration_board(image: np.ndarray, board_type: str, chessboard_size: Tuple[int, int],
                           square_size: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    检测标定板角点

    参数:
        image: 输入图像
        board_type: 标定板类型 ('chessboard'=棋盘格, 'circles'=圆形标定板, 'ring_circles'=环形标定板)
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

    # 获取检测参数
    detection_params = configure_detection_parameters(board_type)

    # 应用优化的图像预处理
    gray = preprocess_image_for_board(image, board_type)

    if board_type == 'chessboard':
        # 检测棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            # 亚像素精度优化
            criteria = detection_params['criteria']
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return objp, corners

    elif board_type in ['circles', 'ring_circles']:
        # 检测圆形标定板（使用与camera_calibration.py相同的方法）
        print(f"  - 检测{board_type}类型标定板...")

        # 获取圆形检测器和标志
        blob_detector = detection_params['detector']
        flags = detection_params['flags']

        # 使用findCirclesGrid检测圆形网格
        ret, corners = cv2.findCirclesGrid(
            image=gray,
            patternSize=chessboard_size,
            flags=flags,
            blobDetector=blob_detector
        )

        if ret:
            print(f"  - 成功检测到 {len(corners)} 个圆形角点")
            return objp, corners
        else:
            print(f"  - {board_type}标定板检测失败，请检查图像质量和参数设置")

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
            print("- 确保黑白方格对比度足够")
        elif board_type == 'circles':
            print("- 检查圆点是否清晰可见")
            print("- 尝试调整照明减少反光")
            print("- 确保黑色圆形与白色背景对比度足够")
        elif board_type == 'ring_circles':
            print("- 确保白色圆形在白色背景上有足够的边缘对比度")
            print("- 考虑增加图像对比度或调整照明角度")
            print("- 检查圆形是否完整且形状规则")
            print("- 如果检测失败，可能需要调整图像预处理参数")

        print("- 尝试增加标定图像数量，覆盖更多角度和位置")
        print("- 确保标定板在图像中清晰可见且无模糊")




class multi_phase:
    """三频外差相位处理类"""
    
    def __init__(self, f: List[int], step: int, images: List[np.ndarray], ph0: float = 0.5):
        """
        初始化三频相位处理器
        
        参数:
            f: 频率列表 [高频, 中频, 低频]
            step: 相移步数
            images: 24张图像列表
            ph0: 初始相位偏移
        """
        self.f = f
        self.step = step
        self.images = images
        self.ph0 = ph0
        
        if len(images) != 24:
            raise ValueError(f"需要24张图像，但提供了{len(images)}张")
    
    def get_phase(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取解包裹相位
        
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
            quality_map = self._compute_quality_map(
                list(h_images['high']), 
                list(v_images['high'])
            )
            
            return unwrapped_v, unwrapped_h, quality_map
            
        except Exception as e:
            raise PhaseUnwrappingError(f"相位解包裹失败: {e}")
    
    def _compute_wrapped_phase(self, images: List[np.ndarray]) -> np.ndarray:
        """
        计算包裹相位 (4步相移算法)
        
        参数:
            images: 4张相移图像
            
        返回:
            wrapped_phase: 包裹相位
        """
        if len(images) != 4:
            raise ValueError("需要4张相移图像")
        
        I1, I2, I3, I4 = [img.astype(np.float32) for img in images]
        
        # 4步相移算法
        wrapped_phase = np.arctan2(I4 - I2, I1 - I3)
        
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
                                      proj_cam_correspondences, board_points, pose_data=None):
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
        
        # 使用预先组织好的姿态数据（如果提供）
        if pose_data is not None:
            for pose_info in pose_data:
                if len(pose_info['obj_points']) > 0:
                    object_points_list.append(np.array(pose_info['obj_points'], dtype=np.float32))
                    projector_points_list.append(np.array(pose_info['proj_points'], dtype=np.float32))
                    camera_points_list.append(np.array(pose_info['cam_points'], dtype=np.float32))
        else:
            # 回退到原来的方法：将所有点作为一个姿态
            object_points_list.append(np.array(object_points, dtype=np.float32))
            projector_points_list.append(np.array(projector_points, dtype=np.float32))
            camera_points_list.append(np.array(camera_points, dtype=np.float32))
        
        # 执行立体标定
        try:
            # 初始化投影仪内参
            projector_matrix_init = camera_matrix.copy()
            projector_dist_init = np.zeros((5,), dtype=np.float32)
            
            # 立体标定
            # 获取图像尺寸
            if len(projector_points_list) > 0 and len(projector_points_list[0]) > 0:
                image_size = (640, 480)  # 默认尺寸
            else:
                image_size = (640, 480)

            ret, _, _, projector_matrix_new, projector_dist_new, R, T, _, _ = cv2.stereoCalibrate(
                object_points_list,
                camera_points_list,
                projector_points_list,
                camera_matrix,
                camera_distortion,
                projector_matrix_init,
                projector_dist_init,
                image_size,
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
            # 尝试两种可能的键名
            if 'dist_coeffs' in camera_data:
                camera_distortion = camera_data['dist_coeffs']
            elif 'camera_distortion' in camera_data:
                camera_distortion = camera_data['camera_distortion']
            else:
                raise KeyError("未找到畸变系数，请检查文件中是否包含 'dist_coeffs' 或 'camera_distortion'")
        elif camera_params_file.endswith('.json'):
            import json
            with open(camera_params_file, 'r') as f:
                camera_data = json.load(f)
            camera_matrix = np.array(camera_data['camera_matrix'])
            # 尝试两种可能的键名
            if 'dist_coeffs' in camera_data:
                camera_distortion = np.array(camera_data['dist_coeffs'])
            elif 'camera_distortion' in camera_data:
                camera_distortion = np.array(camera_data['camera_distortion'])
            else:
                raise KeyError("未找到畸变系数，请检查文件中是否包含 'dist_coeffs' 或 'camera_distortion'")
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
    """主标定函数 - 完整实现"""
    
    # 输入验证
    if not os.path.exists(camera_params_file):
        raise FileNotFoundError(f"相机标定文件不存在: {camera_params_file}")
    
    if not os.path.exists(phase_images_folder):
        raise FileNotFoundError(f"相移图像文件夹不存在: {phase_images_folder}")
    
    if len(frequencies) != 3:
        raise ValueError("必须提供3个频率值")
    
    print_func(f"投影仪标定程序 (基于三频外差相位解包裹)")
    print_func("=" * 60)
    
    # 创建配置对象
    config = ThreeFreqCalibrationConfig(
        frequencies=frequencies,
        phase_step=phase_step,
        ph0=ph0,
        projector_width=projector_width,
        projector_height=projector_height,
        quality_threshold=quality_threshold
    )
    
    # 设置输出文件夹
    if output_folder is None:
        output_folder = os.path.join(phase_images_folder, "three_freq_calibration_results")
    os.makedirs(output_folder, exist_ok=True)
    
    # 加载相机标定参数
    print_func("加载相机标定参数...")
    try:
        if camera_params_file.endswith('.npz'):
            camera_data = np.load(camera_params_file)
            camera_matrix = camera_data['camera_matrix']
            camera_distortion = camera_data['dist_coeffs']
        elif camera_params_file.endswith('.json'):
            with open(camera_params_file, 'r') as f:
                camera_data = json.load(f)
            camera_matrix = np.array(camera_data['camera_matrix'])
            camera_distortion = np.array(camera_data['dist_coeffs'])
        else:
            raise ValueError("不支持的相机标定文件格式")
    except Exception as e:
        raise FileNotFoundError(f"无法加载相机标定参数: {e}")
    
    print_func(f"相机内参矩阵:\n{camera_matrix}")
    
    # 验证文件夹结构
    print_func("验证图像文件夹结构...")
    valid_pose_folders = validate_image_folder_structure(phase_images_folder)
    print_func(f"找到 {len(valid_pose_folders)} 个有效姿态文件夹")
    
    # 处理每个姿态
    all_obj_points = []
    all_proj_points = []
    all_cam_points = []
    pose_data = []  # 存储每个姿态的数据
    
    for i, pose_folder in enumerate(valid_pose_folders):
        pose_name = os.path.basename(pose_folder)
        print_func(f"\n处理姿态 {i+1}/{len(valid_pose_folders)}: {pose_name}")
        
        try:
            # 加载图像路径
            image_paths = load_pose_images(pose_folder)
            organized_paths = organize_three_freq_images(image_paths, config)
            
            # 执行三频外差相位解包裹
            print_func("  - 执行三频外差相位解包裹...")
            unwrapped_v, unwrapped_h, quality_map = process_three_freq_phase_unwrapping(
                organized_paths, config, output_folder if visualize else None, visualize
            )
            
            # 使用第一张垂直图像进行角点检测
            cam_img = cv2.imread(organized_paths['vertical_high'][0], cv2.IMREAD_GRAYSCALE)
            
            # 检测标定板角点
            print_func("  - 检测标定板角点...")
            obj_points_pose, cam_points_pose = detect_calibration_board(
                cam_img, board_type, chessboard_size, square_size
            )
            
            if cam_points_pose is None or len(cam_points_pose) == 0:
                print_func(f"  警告: 在姿态 {pose_name} 中未能检测到标定板，跳过")
                continue
            
            print_func(f"  - 成功检测到 {len(cam_points_pose)} 个角点")
            
            # 提取投影仪中的对应点
            proj_points_pose = []
            valid_indices = []
            
            # 获取有效相位范围
            valid_mask = quality_map > quality_threshold
            if not np.any(valid_mask):
                print_func(f"  警告: 姿态 {pose_name} 中没有满足质量要求的相位点，跳过")
                continue
            
            # 为每个相机角点找到对应的投影仪点
            for j, cam_point in enumerate(cam_points_pose.reshape(-1, 2)):
                x, y = cam_point[0], cam_point[1]
                
                # 检查坐标是否在图像范围内
                if 0 <= int(y) < unwrapped_v.shape[0] and 0 <= int(x) < unwrapped_v.shape[1]:
                    # 使用双线性插值获取相位值
                    phi_v = bilinear_interpolate(unwrapped_v, y, x)
                    phi_h = bilinear_interpolate(unwrapped_h, y, x)
                    quality = bilinear_interpolate(quality_map, y, x)
                    
                    # 检查质量是否满足要求
                    if quality > quality_threshold:
                        # 将相位转换为投影仪坐标
                        proj_x = phi_h / (2 * np.pi) * projector_width
                        proj_y = phi_v / (2 * np.pi) * projector_height
                        
                        # 检查投影仪坐标是否在有效范围内
                        if 0 <= proj_x < projector_width and 0 <= proj_y < projector_height:
                            proj_points_pose.append([proj_x, proj_y])
                            valid_indices.append(j)
            
            if len(proj_points_pose) < 4:
                print_func(f"  警告: 姿态 {pose_name} 有效对应点太少 ({len(proj_points_pose)} < 4)，跳过")
                continue
            
            # 添加到总列表
            valid_obj_points = obj_points_pose[valid_indices]
            valid_cam_points = cam_points_pose.reshape(-1, 2)[valid_indices]

            # 记录这个姿态的数据
            pose_data.append({
                'obj_points': valid_obj_points,
                'proj_points': proj_points_pose,
                'cam_points': valid_cam_points,
                'pose_name': pose_name
            })

            all_obj_points.extend(valid_obj_points)
            all_proj_points.extend(proj_points_pose)
            all_cam_points.extend(valid_cam_points)

            print_func(f"  - 成功提取 {len(proj_points_pose)} 个有效对应点")
            
        except Exception as e:
            print_func(f"  错误: 处理姿态 {pose_name} 时出错: {e}")
            continue
    
    if len(all_obj_points) < 20:
        raise CorrespondenceError(f"总对应点数量不足 ({len(all_obj_points)} < 20)")
    
    print_func(f"\n总共收集到 {len(all_obj_points)} 个对应点")
    
    # 执行投影仪标定
    print_func("执行投影仪标定...")
    calibration = ProjectorCalibration()
    
    # 准备对应关系数据 - 按姿态组织
    proj_cam_correspondences = []
    pose_point_counts = []  # 记录每个姿态的点数

    # 重新组织数据，记录每个姿态的点数
    current_index = 0
    for i, pose_folder in enumerate(valid_pose_folders):
        pose_name = os.path.basename(pose_folder)

        # 计算这个姿态有多少个有效点
        pose_points = 0
        for j in range(current_index, len(all_obj_points)):
            # 这里需要根据实际情况判断点属于哪个姿态
            # 简化处理：假设点是按姿态顺序添加的
            pose_points += 1
            if j == len(all_obj_points) - 1:
                break

        pose_point_counts.append(pose_points)
        current_index += pose_points

    # 创建对应关系
    for i in range(len(all_obj_points)):
        proj_cam_correspondences.append({
            'projector_point': all_proj_points[i],
            'camera_point': all_cam_points[i],
            'board_index': i
        })
    
    # 执行标定
    try:
        reprojection_error, _ = calibration.calibrate_projector_with_camera(
            camera_matrix=camera_matrix,
            camera_distortion=camera_distortion,
            proj_cam_correspondences=proj_cam_correspondences,
            board_points=all_obj_points,
            pose_data=pose_data
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
                       choices=['chessboard', 'circles', 'ring_circles'],
                       help='标定板类型: chessboard=棋盘格, circles=圆形标定板(黑色圆形), ring_circles=环形标定板(白色圆形在白色背景)')
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

# 添加文件扫描和加载函数
def scan_pose_images(pose_folder: str) -> List[str]:
    """
    扫描姿态文件夹中的图像文件
    
    参数:
        pose_folder: 姿态文件夹路径
        
    返回:
        image_files: 图像文件路径列表
    """
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    image_files = []
    
    if not os.path.exists(pose_folder):
        return image_files
    
    for file in os.listdir(pose_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(pose_folder, file))
    
    # 按文件名排序确保顺序正确
    image_files.sort()
    return image_files

def load_pose_images(pose_folder: str) -> List[str]:
    """
    加载姿态文件夹中的图像路径
    
    参数:
        pose_folder: 姿态文件夹路径
        
    返回:
        image_paths: 24张图像的路径列表
    """
    image_files = scan_pose_images(pose_folder)
    
    if len(image_files) != 24:
        raise ValueError(f"姿态文件夹 {pose_folder} 中应包含24张图像，实际找到{len(image_files)}张")
    
    return image_files

def validate_image_folder_structure(phase_images_folder: str) -> List[str]:
    """
    验证图像文件夹结构
    
    参数:
        phase_images_folder: 相移图像根文件夹
        
    返回:
        valid_pose_folders: 有效的姿态文件夹列表
    """
    if not os.path.exists(phase_images_folder):
        raise FileNotFoundError(f"相移图像文件夹不存在: {phase_images_folder}")
    
    pose_folders = [d for d in os.listdir(phase_images_folder) 
                   if os.path.isdir(os.path.join(phase_images_folder, d))]
    
    valid_pose_folders = []
    for pose_folder in pose_folders:
        pose_path = os.path.join(phase_images_folder, pose_folder)
        image_files = scan_pose_images(pose_path)
        
        if len(image_files) == 24:
            valid_pose_folders.append(pose_path)
        else:
            print(f"警告: 姿态文件夹 {pose_folder} 图像数量不正确 ({len(image_files)}/24)")
    
    if len(valid_pose_folders) < 3:
        raise ValueError(f"至少需要3个有效姿态文件夹，当前只有{len(valid_pose_folders)}个")
    
    return valid_pose_folders

if __name__ == "__main__":
    main()



