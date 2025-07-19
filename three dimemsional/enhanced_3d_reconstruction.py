import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
import open3d as o3d
from typing import List, Dict, Tuple, Optional
import os
import json
import argparse
from pathlib import Path
import random
from mpl_toolkits.mplot3d import Axes3D

# 配置matplotlib支持中文显示
def setup_chinese_font():
    """设置matplotlib中文字体"""
    try:
        # 尝试使用系统中文字体
        import platform
        system = platform.system()

        if system == "Windows":
            # Windows系统
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        elif system == "Darwin":  # macOS
            matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans GB', 'STHeiti']
        else:  # Linux
            matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'SimHei']

        matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print("✓ 中文字体配置成功")

    except Exception as e:
        print(f"⚠ 中文字体配置失败: {e}")
        print("  将使用默认字体，中文可能显示为方块")

# 初始化中文字体
setup_chinese_font()


class ParticleSwarmOptimizer:
    """
    粒子群优化算法类，用于三维重建中的深度优化
    
    该类实现了粒子群优化算法，通过最小化相机投影误差来估计三维点的深度值。
    在结构光三维重建中，利用双目几何约束和已知的相位图来优化三维点的深度值。
    """
    
    def __init__(self, camera_matrix, projector_matrix, R, T, 
                 point_camera, point_screen, min_depth=400, max_depth=500,
                 max_iterations=40, num_particles=20, w_ini=0.5, w_end=0.1, c1=2, c2=2):
        """
        初始化粒子群优化器
        
        参数:
            camera_matrix: 相机内参矩阵
            projector_matrix: 投影仪内参矩阵
            R, T: 从投影仪到相机的旋转矩阵和平移向量
            point_camera: 待优化的三维点在相机坐标系下的单位方向向量
            point_screen: 投影仪上的对应点
            min_depth, max_depth: 深度搜索范围
            max_iterations: 最大迭代次数
            num_particles: 粒子数量
            w_ini, w_end: 惯性权重的初始值和终止值
            c1, c2: 加速常数
        """
        self.camera_matrix = camera_matrix
        self.projector_matrix = projector_matrix
        self.R = R
        self.T = T
        self.point_camera = point_camera
        self.point_screen = point_screen
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.num_particles = num_particles
        self.w_ini = w_ini
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        
        # 初始化粒子群
        self.particles = np.random.uniform(min_depth, max_depth, num_particles)
        self.velocities = np.zeros(num_particles)
        self.personal_best = self.particles.copy()
        self.personal_best_scores = np.full(num_particles, float('inf'))
        self.global_best = 0
        self.global_best_score = float('inf')
        
        # 计算初始适应度
        for i in range(num_particles):
            score = self.calculate_fitness(self.particles[i])
            self.personal_best_scores[i] = score
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best = self.particles[i]
    
    def calculate_fitness(self, depth):
        """
        计算适应度函数值(投影误差)
        
        通过计算三维点投影到投影仪上的位置与实际观测位置之间的误差来评估深度值的准确性
        
        参数:
            depth: 当前深度值
            
        返回:
            error: 投影误差值
        """
        # 根据深度值计算三维点在相机坐标系下的坐标
        point_3d = self.point_camera * depth
        
        # 转换到投影仪坐标系
        point_proj = np.dot(self.R, point_3d.reshape(-1, 1)) + self.T.reshape(-1, 1)
        
        # 投影到投影仪成像平面
        point_proj_2d = np.dot(self.projector_matrix, point_proj)
        point_proj_2d = point_proj_2d[:2] / point_proj_2d[2]
        
        # 计算与实际观测点的误差
        error = np.linalg.norm(point_proj_2d.flatten() - self.point_screen[:2])
        
        return error
    
    def optimize(self):
        """
        运行粒子群优化算法
        
        返回:
            best_depth: 最优深度值
            best_score: 最优解对应的误差值
        """
        for iteration in range(self.max_iterations):
            # 计算当前迭代的惯性权重
            w = (self.w_ini - self.w_end) * (self.max_iterations - iteration) / self.max_iterations + self.w_end
            
            for i in range(self.num_particles):
                # 更新速度
                r1, r2 = random.random(), random.random()
                self.velocities[i] = (w * self.velocities[i] + 
                                    self.c1 * r1 * (self.personal_best[i] - self.particles[i]) +
                                    self.c2 * r2 * (self.global_best - self.particles[i]))
                
                # 更新位置
                self.particles[i] += self.velocities[i]
                
                # 边界处理
                self.particles[i] = np.clip(self.particles[i], self.min_depth, self.max_depth)
                
                # 计算适应度
                score = self.calculate_fitness(self.particles[i])
                
                # 更新个体最优
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best[i] = self.particles[i]
                
                # 更新全局最优
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = self.particles[i]
        
        return self.global_best, self.global_best_score


class Enhanced3DReconstruction:
    """
    增强的三维重建类
    
    集成了相位图处理、粒子群优化和点云生成功能
    """
    
    def __init__(self, camera_matrix, projector_matrix, R, T, 
                 projector_width=1280, projector_height=800):
        """
        初始化三维重建对象
        
        参数:
            camera_matrix: 相机内参矩阵
            projector_matrix: 投影仪内参矩阵
            R, T: 从投影仪到相机的旋转矩阵和平移向量
            projector_width, projector_height: 投影仪分辨率
        """
        self.camera_matrix = camera_matrix
        self.projector_matrix = projector_matrix
        self.R = R
        self.T = T
        self.projector_width = projector_width
        self.projector_height = projector_height
    
    def phase_to_pointcloud_optimized(self, unwrapped_phase_x, unwrapped_phase_y,
                                    mask=None, use_pso=True, step_size=5,
                                    quality_threshold=100.0):
        """
        使用粒子群优化的相位图到点云转换（改进版）

        参数:
            unwrapped_phase_x: X方向解包裹相位
            unwrapped_phase_y: Y方向解包裹相位
            mask: 有效区域掩码
            use_pso: 是否使用粒子群优化
            step_size: 采样步长，用于减少计算量
            quality_threshold: 质量阈值

        返回:
            points: 点云坐标数组 (N, 3)
            colors: 点云颜色数组 (N, 3)
            qualities: 重建质量评分数组 (N,)
        """
        height, width = unwrapped_phase_x.shape

        if mask is None:
            mask = np.ones_like(unwrapped_phase_x, dtype=bool)

        # 计算投影仪坐标
        proj_x = unwrapped_phase_x * self.projector_width / (2 * np.pi)
        proj_y = unwrapped_phase_y * self.projector_height / (2 * np.pi)

        points = []
        colors = []
        qualities = []

        print(f"开始三维重建，图像尺寸: {height}x{width}")
        print(f"使用方法: {'粒子群优化' if use_pso else '标准三角测量'}")
        print(f"质量阈值: {quality_threshold}")

        total_pixels = ((height // step_size) * (width // step_size))
        processed = 0
        valid_processed = 0

        for v in range(0, height, step_size):
            for u in range(0, width, step_size):
                if not mask[v, u]:
                    continue

                # 检查投影仪坐标是否在有效范围内
                if (proj_x[v, u] < 0 or proj_x[v, u] >= self.projector_width or
                    proj_y[v, u] < 0 or proj_y[v, u] >= self.projector_height):
                    continue

                # 相机像素坐标转换为归一化坐标
                pixel_coords = np.array([u, v, 1.0])
                camera_ray = np.linalg.inv(self.camera_matrix) @ pixel_coords
                camera_ray = camera_ray / camera_ray[2]  # 归一化

                # 投影仪对应点
                proj_point = np.array([proj_x[v, u], proj_y[v, u], 1.0])

                if use_pso:
                    # 使用粒子群优化估计深度（改进参数）
                    pso = ParticleSwarmOptimizer(
                        self.camera_matrix, self.projector_matrix, self.R, self.T,
                        camera_ray, proj_point,
                        min_depth=100, max_depth=1500,  # 扩大深度搜索范围
                        max_iterations=30,  # 减少迭代次数以提高速度
                        num_particles=15    # 减少粒子数量
                    )
                    depth, quality = pso.optimize()
                else:
                    # 标准三角测量方法
                    depth, quality = self.standard_triangulation(camera_ray, proj_point)

                if depth > 0 and quality < quality_threshold:
                    # 计算三维点坐标
                    point_3d = camera_ray * depth
                    points.append(point_3d)

                    # 基于深度的伪彩色
                    color_val = np.clip((depth - 100) / (1500 - 100), 0, 1)
                    color = plt.cm.jet(color_val)[:3]
                    colors.append(color)
                    qualities.append(quality)
                    valid_processed += 1

                processed += 1
                if processed % 500 == 0:
                    print(f"处理进度: {processed}/{total_pixels} ({100*processed/total_pixels:.1f}%), "
                          f"有效点: {valid_processed}")

        if len(points) == 0:
            print("警告: 没有生成有效的三维点")
            print("建议:")
            print("1. 检查相位图质量")
            print("2. 增大质量阈值")
            print("3. 检查标定参数")
            return np.array([]), np.array([]), np.array([])

        points = np.array(points)
        colors = np.array(colors)
        qualities = np.array(qualities)

        print(f"成功重建 {len(points)} 个三维点")
        print(f"质量统计: 最小={np.min(qualities):.3f}, 最大={np.max(qualities):.3f}, "
              f"平均={np.mean(qualities):.3f}")

        return points, colors, qualities

    def calculate_surface_normal(self, camera_ray, proj_point, depth):
        """
        计算表面法线（借鉴原始版本的方法）

        参数:
            camera_ray: 相机射线方向
            proj_point: 投影仪对应点
            depth: 深度值

        返回:
            normal: 归一化的表面法线向量
        """
        try:
            # 计算三维点
            point_3d = camera_ray * depth

            # 转换到投影仪坐标系
            point_proj_3d = np.dot(self.R, point_3d.reshape(-1, 1)) + self.T.reshape(-1, 1)
            point_proj_3d = point_proj_3d.flatten()

            # 相机方向（归一化）
            c1 = -camera_ray / np.linalg.norm(camera_ray)

            # 从点到投影仪的方向
            s1 = point_proj_3d - point_3d
            s1 = s1 / np.linalg.norm(s1)

            # 法线为相机方向和投影方向的和（原始版本的方法）
            normal = c1 + s1
            normal = normal / np.linalg.norm(normal)

            return normal

        except Exception as e:
            # 如果计算失败，返回默认法线
            return np.array([0, 0, 1])

    def calculate_common_view_region(self, phase_x1, phase_y1, phase_x2, phase_y2):
        """
        计算两个相机的共视区域（借鉴原始版本）

        参数:
            phase_x1, phase_y1: 第一个相机的相位图
            phase_x2, phase_y2: 第二个相机的相位图

        返回:
            mask1, mask2: 两个相机的共视区域掩码
        """
        # 处理水平方向共视区域
        min_x1, max_x1 = np.min(phase_x1), np.max(phase_x1)
        min_x2, max_x2 = np.min(phase_x2), np.max(phase_x2)

        # 取共同的水平相位范围
        max_x = min(max_x1, max_x2)
        min_x = max(min_x1, min_x2)

        # 创建水平方向掩码
        mask1_x = (phase_x1 >= min_x) & (phase_x1 <= max_x)
        mask2_x = (phase_x2 >= min_x) & (phase_x2 <= max_x)

        # 处理垂直方向共视区域
        min_y1, max_y1 = np.min(phase_y1), np.max(phase_y1)
        min_y2, max_y2 = np.min(phase_y2), np.max(phase_y2)

        # 取共同的垂直相位范围
        max_y = min(max_y1, max_y2)
        min_y = max(min_y1, min_y2)

        # 创建垂直方向掩码
        mask1_y = (phase_y1 >= min_y) & (phase_y1 <= max_y)
        mask2_y = (phase_y2 >= min_y) & (phase_y2 <= max_y)

        # 水平和垂直方向都有效的区域才是最终有效区域
        mask1 = mask1_x & mask1_y
        mask2 = mask2_x & mask2_y

        return mask1, mask2

    def standard_triangulation(self, camera_ray, proj_point):
        """
        标准三角测量方法（基于参考文件的SVD方法）

        参数:
            camera_ray: 相机射线方向 (归一化齐次坐标)
            proj_point: 投影仪对应点 (像素坐标)

        返回:
            depth: 估计的深度值
            quality: 重建质量评分
        """
        try:
            # 构建投影矩阵（参考标准方法）
            P_cam = self.camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
            P_proj = self.projector_matrix @ np.hstack((self.R, self.T.reshape(-1, 1)))

            # 相机像素坐标（从射线反推）
            cam_pixel = self.camera_matrix @ camera_ray
            cam_pixel = cam_pixel / cam_pixel[2]  # 归一化

            # 构建线性方程组 AX = 0
            A = np.zeros((4, 4))

            # 相机投影方程
            A[0, :] = cam_pixel[0] * P_cam[2, :] - P_cam[0, :]
            A[1, :] = cam_pixel[1] * P_cam[2, :] - P_cam[1, :]

            # 投影仪投影方程
            A[2, :] = proj_point[0] * P_proj[2, :] - P_proj[0, :]
            A[3, :] = proj_point[1] * P_proj[2, :] - P_proj[1, :]

            # 使用SVD求解齐次线性方程组 AX = 0
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1, :]

            # 齐次坐标转为3D点
            if abs(X[3]) < 1e-10:  # 避免除零
                return 0, float('inf')

            X = X / X[3]
            point_3d = X[:3]

            # 计算深度（相机坐标系下的Z值）
            depth = point_3d[2]

            # 计算重投影误差作为质量评分
            # 重投影到相机
            cam_reproj = self.camera_matrix @ point_3d
            cam_reproj = cam_reproj / cam_reproj[2]
            cam_error = np.linalg.norm(cam_reproj[:2] - cam_pixel[:2])

            # 重投影到投影仪
            proj_3d_in_proj = self.R @ point_3d + self.T.reshape(-1, 1)
            proj_reproj = self.projector_matrix @ proj_3d_in_proj
            proj_reproj = proj_reproj / proj_reproj[2]
            proj_error = np.linalg.norm(proj_reproj[:2] - proj_point[:2])

            # 综合误差
            quality = cam_error + proj_error

            # 确保深度值合理
            if depth < 50 or depth > 2000:
                return 0, float('inf')

            return depth, quality

        except Exception as e:
            return 0, float('inf')

    def simple_triangulation(self, camera_ray, proj_point):
        """
        简化的三角测量方法（保持向后兼容）
        """
        return self.standard_triangulation(camera_ray, proj_point)

    def vectorized_triangulation(self, unwrapped_phase_x, unwrapped_phase_y, mask, step_size=1):
        """
        向量化的三角测量方法，提高处理速度

        参数:
            unwrapped_phase_x, unwrapped_phase_y: 解包裹相位
            mask: 有效区域掩码
            step_size: 采样步长

        返回:
            points: 点云坐标数组
            colors: 点云颜色数组
            qualities: 质量评分数组
        """
        height, width = unwrapped_phase_x.shape

        # 计算投影仪坐标
        proj_x = unwrapped_phase_x * self.projector_width / (2 * np.pi)
        proj_y = unwrapped_phase_y * self.projector_height / (2 * np.pi)

        # 构建投影矩阵
        P_cam = self.camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P_proj = self.projector_matrix @ np.hstack((self.R, self.T.reshape(-1, 1)))

        points = []
        colors = []
        qualities = []

        # 批量处理以提高效率
        batch_size = 1000
        valid_pixels = []

        # 收集所有有效像素
        for v in range(0, height, step_size):
            for u in range(0, width, step_size):
                if mask[v, u]:
                    valid_pixels.append((u, v, proj_x[v, u], proj_y[v, u]))

        print(f"开始向量化三角测量，有效像素数: {len(valid_pixels)}")

        # 分批处理
        for i in range(0, len(valid_pixels), batch_size):
            batch = valid_pixels[i:i+batch_size]
            batch_points, batch_colors, batch_qualities = self._process_pixel_batch(
                batch, P_cam, P_proj
            )

            points.extend(batch_points)
            colors.extend(batch_colors)
            qualities.extend(batch_qualities)

            if (i // batch_size + 1) % 10 == 0:
                progress = min(100, (i + batch_size) * 100 // len(valid_pixels))
                print(f"处理进度: {progress}%")

        return np.array(points), np.array(colors), np.array(qualities)

    def _process_pixel_batch(self, pixel_batch, P_cam, P_proj):
        """
        处理一批像素的三角测量
        """
        batch_points = []
        batch_colors = []
        batch_qualities = []

        for u, v, proj_u, proj_v in pixel_batch:
            # 检查投影仪坐标是否有效
            if proj_u < 0 or proj_u >= self.projector_width or proj_v < 0 or proj_v >= self.projector_height:
                continue

            # 相机和投影仪的像素坐标
            cam_pixel = np.array([u, v, 1.0])
            proj_pixel = np.array([proj_u, proj_v, 1.0])

            # 构建线性方程组
            A = np.zeros((4, 4))

            # 相机投影方程
            A[0, :] = cam_pixel[0] * P_cam[2, :] - P_cam[0, :]
            A[1, :] = cam_pixel[1] * P_cam[2, :] - P_cam[1, :]

            # 投影仪投影方程
            A[2, :] = proj_pixel[0] * P_proj[2, :] - P_proj[0, :]
            A[3, :] = proj_pixel[1] * P_proj[2, :] - P_proj[1, :]

            try:
                # 使用SVD求解
                _, _, Vt = np.linalg.svd(A)
                X = Vt[-1, :]

                if abs(X[3]) < 1e-10:
                    continue

                X = X / X[3]
                point_3d = X[:3]

                # 更宽松的深度检查
                if point_3d[2] < 10 or point_3d[2] > 5000:
                    continue

                # 计算重投影误差
                cam_reproj = P_cam @ np.append(point_3d, 1)
                if abs(cam_reproj[2]) < 1e-10:
                    continue
                cam_reproj = cam_reproj / cam_reproj[2]
                cam_error = np.linalg.norm(cam_reproj[:2] - cam_pixel[:2])

                proj_reproj = P_proj @ np.append(point_3d, 1)
                if abs(proj_reproj[2]) < 1e-10:
                    continue
                proj_reproj = proj_reproj / proj_reproj[2]
                proj_error = np.linalg.norm(proj_reproj[:2] - proj_pixel[:2])

                quality = cam_error + proj_error

                # 极其宽松的质量阈值以保留更多点
                if quality < 500.0:  # 进一步增大质量阈值
                    batch_points.append(point_3d)

                    # 基于深度的伪彩色
                    color_val = np.clip((point_3d[2] - 100) / (1000 - 100), 0, 1)
                    color = plt.cm.jet(color_val)[:3]
                    batch_colors.append(color)
                    batch_qualities.append(quality)

            except Exception as e:
                continue

        return batch_points, batch_colors, batch_qualities

    def enhanced_reconstruction_with_normals(self, unwrapped_phase_x, unwrapped_phase_y,
                                           mask=None, use_pso=True, step_size=5,
                                           quality_threshold=100.0, include_normals=True):
        """
        增强的三维重建方法，包含法线计算和详细结果存储

        参数:
            unwrapped_phase_x, unwrapped_phase_y: 解包裹相位
            mask: 有效区域掩码
            use_pso: 是否使用粒子群优化
            step_size: 采样步长
            quality_threshold: 质量阈值
            include_normals: 是否计算表面法线

        返回:
            detailed_results: 详细结果数组，包含图像坐标、三维坐标、法线、质量等信息
        """
        height, width = unwrapped_phase_x.shape

        if mask is None:
            mask = np.ones_like(unwrapped_phase_x, dtype=bool)

        # 计算投影仪坐标
        proj_x = unwrapped_phase_x * self.projector_width / (2 * np.pi)
        proj_y = unwrapped_phase_y * self.projector_height / (2 * np.pi)

        detailed_results = []

        print(f"开始增强三维重建，图像尺寸: {height}x{width}")
        print(f"使用方法: {'粒子群优化' if use_pso else '标准三角测量'}")
        print(f"包含法线计算: {include_normals}")

        total_pixels = ((height // step_size) * (width // step_size))
        processed = 0
        valid_processed = 0

        for v in range(0, height, step_size):
            for u in range(0, width, step_size):
                if not mask[v, u]:
                    continue

                # 检查投影仪坐标是否在有效范围内
                if (proj_x[v, u] < 0 or proj_x[v, u] >= self.projector_width or
                    proj_y[v, u] < 0 or proj_y[v, u] >= self.projector_height):
                    continue

                # 相机像素坐标转换为归一化坐标
                pixel_coords = np.array([u, v, 1.0])
                camera_ray = np.linalg.inv(self.camera_matrix) @ pixel_coords
                camera_ray = camera_ray / camera_ray[2]  # 归一化

                # 投影仪对应点
                proj_point = np.array([proj_x[v, u], proj_y[v, u], 1.0])

                if use_pso:
                    # 使用粒子群优化估计深度
                    pso = ParticleSwarmOptimizer(
                        self.camera_matrix, self.projector_matrix, self.R, self.T,
                        camera_ray, proj_point,
                        min_depth=100, max_depth=1500,
                        max_iterations=30,
                        num_particles=15
                    )
                    depth, quality = pso.optimize()
                else:
                    # 标准三角测量方法
                    depth, quality = self.standard_triangulation(camera_ray, proj_point)

                if depth > 0 and quality < quality_threshold:
                    # 计算三维点坐标
                    point_3d = camera_ray * depth

                    # 计算表面法线（如果需要）
                    if include_normals:
                        normal = self.calculate_surface_normal(camera_ray, proj_point, depth)
                    else:
                        normal = np.array([0, 0, 1])  # 默认法线

                    # 创建详细结果数组（类似原始版本的9维信息）
                    result_entry = {
                        'image_coords': [v, u],           # 图像坐标 (y, x)
                        'point_3d': point_3d,             # 三维坐标
                        'normal': normal,                 # 表面法线
                        'quality': quality,               # 重建质量
                        'depth': depth,                   # 深度值
                        'proj_coords': [proj_x[v, u], proj_y[v, u]]  # 投影仪坐标
                    }

                    detailed_results.append(result_entry)
                    valid_processed += 1

                processed += 1
                if processed % 500 == 0:
                    print(f"处理进度: {processed}/{total_pixels} ({100*processed/total_pixels:.1f}%), "
                          f"有效点: {valid_processed}")

        print(f"增强重建完成，生成 {len(detailed_results)} 个详细结果")

        return detailed_results

    def detailed_results_to_arrays(self, detailed_results):
        """
        将详细结果转换为标准数组格式

        参数:
            detailed_results: 详细结果列表

        返回:
            points, colors, qualities, normals: 标准数组格式
        """
        if not detailed_results:
            return np.array([]), np.array([]), np.array([]), np.array([])

        points = []
        colors = []
        qualities = []
        normals = []

        for result in detailed_results:
            point_3d = result['point_3d']
            quality = result['quality']
            normal = result['normal']
            depth = result['depth']

            points.append(point_3d)
            qualities.append(quality)
            normals.append(normal)

            # 基于深度的伪彩色
            color_val = np.clip((depth - 100) / (1500 - 100), 0, 1)
            color = plt.cm.jet(color_val)[:3]
            colors.append(color)

        return np.array(points), np.array(colors), np.array(qualities), np.array(normals)

    def create_mask(self, unwrapped_phase_x, unwrapped_phase_y, percentile_threshold=98.0):
        """
        根据解包裹相位创建有效区域掩码

        参数:
            unwrapped_phase_x: X方向解包裹相位
            unwrapped_phase_y: Y方向解包裹相位
            percentile_threshold: 相位梯度阈值的百分位数

        返回:
            mask: 有效区域掩码
        """
        # 计算相位梯度
        phase_gradient_x = np.gradient(unwrapped_phase_x)
        phase_gradient_y = np.gradient(unwrapped_phase_y)
        gradient_magnitude = np.sqrt(phase_gradient_x[0]**2 + phase_gradient_x[1]**2 +
                                   phase_gradient_y[0]**2 + phase_gradient_y[1]**2)

        # 使用更宽松的阈值以保留更多区域（特别是球形物体的边缘）
        threshold = np.percentile(gradient_magnitude, percentile_threshold)
        mask = gradient_magnitude < threshold

        # 形态学操作以平滑掩码
        import cv2 as cv
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        mask_uint8 = mask.astype(np.uint8) * 255

        # 闭运算：填充小洞
        mask_uint8 = cv.morphologyEx(mask_uint8, cv.MORPH_CLOSE, kernel)
        # 开运算：去除小噪声
        mask_uint8 = cv.morphologyEx(mask_uint8, cv.MORPH_OPEN, kernel)

        mask = mask_uint8 > 0

        return mask

    def filter_pointcloud(self, points, colors, qualities, quality_threshold=50.0,
                         enable_statistical_filter=True, preserve_density=True):
        """
        改进的点云过滤，保留更多有效点

        参数:
            points: 点云坐标
            colors: 点云颜色
            qualities: 重建质量评分
            quality_threshold: 质量阈值
            enable_statistical_filter: 是否启用统计过滤
            preserve_density: 是否保持点云密度

        返回:
            filtered_points, filtered_colors: 过滤后的点云和颜色
        """
        if len(points) == 0:
            return points, colors

        print(f"开始过滤，原始点数: {len(points)}")

        # 1. 基于质量的渐进过滤
        if preserve_density and len(points) > 1000:
            # 如果点数较多，使用更严格的质量阈值
            quality_threshold = min(quality_threshold, np.percentile(qualities, 70))
        elif len(points) < 500:
            # 如果点数较少，使用更宽松的质量阈值
            quality_threshold = max(quality_threshold, np.percentile(qualities, 90))

        quality_mask = qualities < quality_threshold
        print(f"质量过滤 (阈值={quality_threshold:.1f}): 保留 {np.sum(quality_mask)} 个点")

        # 2. 基于深度的智能过滤
        z_values = points[:, 2]

        # 使用更鲁棒的深度范围估计
        z_median = np.median(z_values)
        z_mad = np.median(np.abs(z_values - z_median))  # 中位数绝对偏差

        # 动态调整深度范围
        if z_mad > 0:
            depth_range = 6 * z_mad  # 使用6倍MAD作为范围
        else:
            depth_range = 3 * np.std(z_values)  # 回退到标准差

        depth_mask = np.abs(z_values - z_median) < depth_range
        print(f"深度过滤 (中位数={z_median:.1f}, 范围=±{depth_range:.1f}): 保留 {np.sum(depth_mask)} 个点")

        # 3. 移除明显异常的深度值
        reasonable_depth_mask = (z_values > 10) & (z_values < 5000)  # 更宽松的范围
        print(f"合理深度过滤: 保留 {np.sum(reasonable_depth_mask)} 个点")

        # 4. 组合所有过滤条件
        if preserve_density:
            # 保持密度模式：优先保留更多点
            final_mask = quality_mask & reasonable_depth_mask
            if np.sum(final_mask) > len(points) * 0.8:  # 如果保留点数过多，再加上深度过滤
                final_mask = final_mask & depth_mask
        else:
            # 标准模式：使用所有过滤条件
            final_mask = quality_mask & depth_mask & reasonable_depth_mask

        filtered_points = points[final_mask]
        filtered_colors = colors[final_mask]

        print(f"初步过滤后: {len(filtered_points)} 个点")

        # 5. 可选的统计噪声过滤（更温和）
        if enable_statistical_filter and len(filtered_points) > 100:
            # 使用更温和的统计过滤参数
            try:
                import open3d as o3d
                pcd_temp = o3d.geometry.PointCloud()
                pcd_temp.points = o3d.utility.Vector3dVector(filtered_points)

                # 更宽松的统计过滤参数
                pcd_filtered, outlier_mask = pcd_temp.remove_statistical_outlier(
                    nb_neighbors=10,  # 减少邻居数量
                    std_ratio=3.0     # 增大标准差比例
                )

                if len(pcd_filtered.points) > len(filtered_points) * 0.5:  # 至少保留50%的点
                    filtered_points = np.asarray(pcd_filtered.points)
                    filtered_colors = filtered_colors[outlier_mask]
                    print(f"统计噪声过滤后: {len(filtered_points)} 个点")
                else:
                    print("统计过滤过于严格，跳过此步骤")
            except:
                print("统计过滤失败，跳过此步骤")

        print(f"最终过滤结果: {len(filtered_points)} 个点 (保留率: {100*len(filtered_points)/len(points):.1f}%)")

        return filtered_points, filtered_colors

    def save_pointcloud_multiple_formats(self, points, colors=None, normals=None,
                                        output_dir="output", filename_base="pointcloud",
                                        formats=['ply', 'pcd', 'xyz', 'txt']):
        """
        保存点云到多种格式

        参数:
            points: 点云坐标数组 (N, 3)
            colors: 点云颜色数组 (N, 3)，可选
            normals: 点云法线数组 (N, 3)，可选
            output_dir: 输出目录
            filename_base: 文件名基础部分
            formats: 要保存的格式列表

        返回:
            saved_files: 保存的文件路径列表
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []

        print(f"保存点云到多种格式: {formats}")

        # 创建Open3D点云对象
        pcd = self.create_open3d_pointcloud(points, colors)
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)

        for fmt in formats:
            try:
                if fmt.lower() == 'ply':
                    # PLY格式 - 最常用的点云格式
                    filepath = os.path.join(output_dir, f"{filename_base}.ply")
                    o3d.io.write_point_cloud(filepath, pcd)
                    saved_files.append(filepath)
                    print(f"✅ PLY格式保存成功: {filepath}")

                elif fmt.lower() == 'pcd':
                    # PCD格式 - PCL库常用格式
                    filepath = os.path.join(output_dir, f"{filename_base}.pcd")
                    o3d.io.write_point_cloud(filepath, pcd)
                    saved_files.append(filepath)
                    print(f"✅ PCD格式保存成功: {filepath}")

                elif fmt.lower() == 'xyz':
                    # XYZ格式 - 简单的ASCII格式
                    filepath = os.path.join(output_dir, f"{filename_base}.xyz")
                    self._save_xyz_format(points, colors, normals, filepath)
                    saved_files.append(filepath)
                    print(f"✅ XYZ格式保存成功: {filepath}")

                elif fmt.lower() == 'txt':
                    # TXT格式 - 纯文本格式
                    filepath = os.path.join(output_dir, f"{filename_base}.txt")
                    self._save_txt_format(points, colors, normals, filepath)
                    saved_files.append(filepath)
                    print(f"✅ TXT格式保存成功: {filepath}")

                elif fmt.lower() == 'obj':
                    # OBJ格式 - 3D模型格式
                    filepath = os.path.join(output_dir, f"{filename_base}.obj")
                    self._save_obj_format(points, colors, normals, filepath)
                    saved_files.append(filepath)
                    print(f"✅ OBJ格式保存成功: {filepath}")

                elif fmt.lower() == 'csv':
                    # CSV格式 - 表格格式
                    filepath = os.path.join(output_dir, f"{filename_base}.csv")
                    self._save_csv_format(points, colors, normals, filepath)
                    saved_files.append(filepath)
                    print(f"✅ CSV格式保存成功: {filepath}")

                else:
                    print(f"⚠️ 不支持的格式: {fmt}")

            except Exception as e:
                print(f"❌ 保存{fmt}格式失败: {e}")

        return saved_files

    def _save_xyz_format(self, points, colors, normals, filepath):
        """保存XYZ格式"""
        with open(filepath, 'w') as f:
            for i in range(len(points)):
                line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
                if colors is not None:
                    line += f" {colors[i, 0]:.6f} {colors[i, 1]:.6f} {colors[i, 2]:.6f}"
                if normals is not None:
                    line += f" {normals[i, 0]:.6f} {normals[i, 1]:.6f} {normals[i, 2]:.6f}"
                f.write(line + "\n")

    def _save_txt_format(self, points, colors, normals, filepath):
        """保存TXT格式（兼容原始版本格式）"""
        # 创建完整的数据矩阵
        data_matrix = []
        for i in range(len(points)):
            row = [i, i]  # 图像坐标占位符
            row.extend(points[i])  # 三维坐标
            if normals is not None:
                row.extend(normals[i])  # 法线
            else:
                row.extend([0, 0, 1])  # 默认法线
            row.append(1.0)  # 质量评分占位符
            data_matrix.append(row)

        np.savetxt(filepath, np.array(data_matrix), fmt='%.6f',
                  header='ImageY ImageX X Y Z NormalX NormalY NormalZ Quality')

    def _save_obj_format(self, points, colors, normals, filepath):
        """保存OBJ格式"""
        with open(filepath, 'w') as f:
            f.write("# OBJ file generated by Enhanced 3D Reconstruction\n")
            f.write(f"# {len(points)} vertices\n\n")

            # 写入顶点
            for i in range(len(points)):
                if colors is not None:
                    f.write(f"v {points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                           f"{colors[i, 0]:.6f} {colors[i, 1]:.6f} {colors[i, 2]:.6f}\n")
                else:
                    f.write(f"v {points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}\n")

            # 写入法线
            if normals is not None:
                f.write("\n")
                for i in range(len(normals)):
                    f.write(f"vn {normals[i, 0]:.6f} {normals[i, 1]:.6f} {normals[i, 2]:.6f}\n")

    def _save_csv_format(self, points, colors, normals, filepath):
        """保存CSV格式"""
        import pandas as pd

        # 创建数据字典
        data = {
            'X': points[:, 0],
            'Y': points[:, 1],
            'Z': points[:, 2]
        }

        if colors is not None:
            data.update({
                'R': colors[:, 0],
                'G': colors[:, 1],
                'B': colors[:, 2]
            })

        if normals is not None:
            data.update({
                'NormalX': normals[:, 0],
                'NormalY': normals[:, 1],
                'NormalZ': normals[:, 2]
            })

        # 创建DataFrame并保存
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, float_format='%.6f')

    def create_open3d_pointcloud(self, points, colors=None):
        """
        创建Open3D点云对象

        参数:
            points: 点云坐标数组 (N, 3)
            colors: 点云颜色数组 (N, 3)

        返回:
            pcd: Open3D点云对象
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def visualize_pointcloud(self, pcd, window_name="增强三维重建结果"):
        """
        可视化点云

        参数:
            pcd: Open3D点云对象
            window_name: 窗口名称
        """
        # 添加坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=50, origin=[0, 0, 0]
        )

        # 显示点云
        o3d.visualization.draw_geometries([pcd, coordinate_frame], window_name=window_name)

    def create_mesh_from_pointcloud(self, pcd, voxel_size=2.0, depth=9, method='poisson'):
        """
        从点云创建网格

        参数:
            pcd: Open3D点云对象
            voxel_size: 体素大小
            depth: 泊松重建深度
            method: 重建方法 ('poisson' 或 'alpha_shape')

        返回:
            mesh: 三角网格
        """
        print("估计点云法线...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )

        print("确保法线方向一致...")
        pcd.orient_normals_consistent_tangent_plane(k=20)

        if method == 'poisson':
            print("使用泊松重建生成网格...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth
            )

            # 根据密度裁剪网格
            vertices_to_remove = densities < np.quantile(densities, 0.1)
            mesh.remove_vertices_by_mask(vertices_to_remove)

        elif method == 'alpha_shape':
            print("使用Alpha Shape重建生成网格...")
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, alpha=voxel_size * 5
            )

        else:
            raise ValueError(f"不支持的网格重建方法: {method}")

        # 简化网格
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=50000)

        # 平滑网格
        mesh = mesh.filter_smooth_simple(number_of_iterations=3)

        # 计算法线
        mesh.compute_vertex_normals()

        return mesh


class Enhanced3DReconstructionAPI:
    """
    增强三维重建API类

    提供简化的API接口供其他程序调用
    """

    def __init__(self):
        self.reconstructor = None
        self.is_initialized = False

    def initialize(self, camera_params_file, projector_params_file, extrinsics_file):
        """
        初始化重建系统

        参数:
            camera_params_file: 相机参数文件路径
            projector_params_file: 投影仪参数文件路径
            extrinsics_file: 外参文件路径

        返回:
            success: 是否初始化成功
        """
        try:
            # 加载参数
            camera_matrix = load_camera_params(camera_params_file)
            projector_matrix, proj_width, proj_height = load_projector_params(projector_params_file)
            R, T = load_extrinsics(extrinsics_file)

            if (camera_matrix is None or projector_matrix is None or
                R is None or T is None):
                return False

            # 创建重建对象
            self.reconstructor = Enhanced3DReconstruction(
                camera_matrix, projector_matrix, R, T, proj_width, proj_height
            )

            self.is_initialized = True
            return True

        except Exception as e:
            print(f"初始化失败: {e}")
            return False

    def reconstruct_from_files(self, phase_x_file, phase_y_file, output_dir=None,
                             use_pso=True, step_size=5, create_mesh=True):
        """
        从文件进行三维重建

        参数:
            phase_x_file: X方向相位文件路径
            phase_y_file: Y方向相位文件路径
            output_dir: 输出目录（可选）
            use_pso: 是否使用粒子群优化
            step_size: 采样步长
            create_mesh: 是否创建网格

        返回:
            result: 重建结果字典
        """
        if not self.is_initialized:
            return {"success": False, "error": "系统未初始化"}

        try:
            # 加载相位图
            phase_x, phase_y = load_unwrapped_phases(phase_x_file, phase_y_file)
            if phase_x is None or phase_y is None:
                return {"success": False, "error": "相位图加载失败"}

            return self.reconstruct_from_arrays(
                phase_x, phase_y, output_dir, use_pso, step_size, create_mesh
            )

        except Exception as e:
            return {"success": False, "error": str(e)}

    def reconstruct_from_arrays(self, phase_x, phase_y, output_dir=None,
                              use_pso=True, step_size=5, create_mesh=True):
        """
        从数组进行三维重建

        参数:
            phase_x, phase_y: 相位数组
            output_dir: 输出目录（可选）
            use_pso: 是否使用粒子群优化
            step_size: 采样步长
            create_mesh: 是否创建网格

        返回:
            result: 重建结果字典
        """
        if not self.is_initialized:
            return {"success": False, "error": "系统未初始化"}

        try:
            # 创建掩码
            mask = self.reconstructor.create_mask(phase_x, phase_y)

            # 选择重建方法
            if use_pso:
                points, colors, qualities = self.reconstructor.phase_to_pointcloud_optimized(
                    phase_x, phase_y, mask, use_pso=True, step_size=step_size
                )
            else:
                points, colors, qualities = self.reconstructor.vectorized_triangulation(
                    phase_x, phase_y, mask, step_size=step_size
                )

            if len(points) == 0:
                return {"success": False, "error": "未生成有效点云"}

            # 过滤点云
            filtered_points, filtered_colors = self.reconstructor.filter_pointcloud(
                points, colors, qualities
            )

            # 创建点云对象
            pcd = self.reconstructor.create_open3d_pointcloud(filtered_points, filtered_colors)

            result = {
                "success": True,
                "points": filtered_points,
                "colors": filtered_colors,
                "pointcloud": pcd,
                "stats": {
                    "total_points": len(points),
                    "filtered_points": len(filtered_points),
                    "average_quality": float(np.mean(qualities)) if len(qualities) > 0 else 0
                }
            }

            # 创建网格（如果需要）
            if create_mesh and len(filtered_points) > 100:
                try:
                    mesh = self.reconstructor.create_mesh_from_pointcloud(pcd)
                    result["mesh"] = mesh
                except Exception as e:
                    result["mesh_error"] = str(e)

            # 保存结果（如果指定了输出目录）
            if output_dir:
                result["saved_files"] = self.save_reconstruction_results(
                    filtered_points, filtered_colors, result.get("mesh"),
                    result["stats"], output_dir
                )
                result["output_dir"] = output_dir

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def save_reconstruction_results(self, points, colors, mesh=None, stats=None,
                                  output_dir="output", save_formats=['ply', 'txt', 'csv']):
        """
        保存重建结果到多种格式

        参数:
            points: 点云坐标数组
            colors: 点云颜色数组
            mesh: 网格对象（可选）
            stats: 统计信息字典（可选）
            output_dir: 输出目录
            save_formats: 要保存的格式列表

        返回:
            saved_files: 保存的文件路径字典
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}

        print(f"保存重建结果到目录: {output_dir}")

        # 保存点云到多种格式
        if len(points) > 0:
            pointcloud_files = self.reconstructor.save_pointcloud_multiple_formats(
                points, colors, output_dir=output_dir,
                filename_base="enhanced_pointcloud", formats=save_formats
            )
            saved_files["pointcloud"] = pointcloud_files

        # 保存网格
        if mesh is not None:
            try:
                import open3d as o3d
                mesh_file = os.path.join(output_dir, "enhanced_mesh.ply")
                o3d.io.write_triangle_mesh(mesh_file, mesh)
                saved_files["mesh"] = mesh_file
                print(f"✅ 网格保存成功: {mesh_file}")
            except Exception as e:
                print(f"❌ 网格保存失败: {e}")

        # 保存统计信息
        if stats is not None:
            try:
                import json
                stats_file = os.path.join(output_dir, "reconstruction_stats.json")
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)
                saved_files["stats"] = stats_file
                print(f"✅ 统计信息保存成功: {stats_file}")
            except Exception as e:
                print(f"❌ 统计信息保存失败: {e}")

        # 创建保存报告
        self._create_save_report(saved_files, output_dir)

        return saved_files

    def _create_save_report(self, saved_files, output_dir):
        """创建保存报告"""
        import time
        report_file = os.path.join(output_dir, "save_report.txt")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("增强三维重建系统 - 保存报告\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"保存时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输出目录: {output_dir}\n\n")

            f.write("保存的文件:\n")
            f.write("-" * 30 + "\n")

            for category, files in saved_files.items():
                f.write(f"\n{category.upper()}:\n")
                if isinstance(files, list):
                    for file_path in files:
                        f.write(f"  - {os.path.basename(file_path)}\n")
                else:
                    f.write(f"  - {os.path.basename(files)}\n")

            f.write(f"\n总计保存文件数: {sum(len(files) if isinstance(files, list) else 1 for files in saved_files.values())}\n")

        print(f"✅ 保存报告创建成功: {report_file}")

    def get_reconstruction_quality(self, points, qualities):
        """
        获取重建质量评估

        参数:
            points: 点云坐标
            qualities: 质量评分

        返回:
            quality_report: 质量报告字典
        """
        if len(points) == 0:
            return {"error": "点云为空"}

        return {
            "point_count": len(points),
            "average_quality": float(np.mean(qualities)),
            "quality_std": float(np.std(qualities)),
            "depth_range": {
                "min": float(np.min(points[:, 2])),
                "max": float(np.max(points[:, 2])),
                "mean": float(np.mean(points[:, 2]))
            },
            "quality_percentiles": {
                "25%": float(np.percentile(qualities, 25)),
                "50%": float(np.percentile(qualities, 50)),
                "75%": float(np.percentile(qualities, 75)),
                "95%": float(np.percentile(qualities, 95))
            }
        }


def load_camera_params(file_path):
    """
    加载相机参数

    参数:
        file_path: 相机参数文件路径

    返回:
        camera_matrix: 相机内参矩阵
    """
    try:
        if file_path.endswith('.npy'):
            camera_data = np.load(file_path, allow_pickle=True).item()
            camera_matrix = np.array(camera_data['camera_matrix'])
        else:
            with open(file_path, 'r') as f:
                camera_data = json.load(f)
            camera_matrix = np.array(camera_data['camera_matrix'])

        print(f"已加载相机内参矩阵:\n{camera_matrix}")
        return camera_matrix
    except Exception as e:
        print(f"加载相机参数失败: {e}")
        return None


def load_projector_params(file_path):
    """
    加载投影仪参数

    参数:
        file_path: 投影仪参数文件路径

    返回:
        projector_matrix: 投影仪内参矩阵
        projector_width: 投影仪宽度
        projector_height: 投影仪高度
    """
    try:
        if file_path.endswith('.npy'):
            projector_data = np.load(file_path, allow_pickle=True).item()
        else:
            with open(file_path, 'r') as f:
                projector_data = json.load(f)

        projector_matrix = np.array(projector_data['projector_matrix'])
        projector_width = projector_data.get('projector_width', 1280)
        projector_height = projector_data.get('projector_height', 800)

        print(f"已加载投影仪内参矩阵:\n{projector_matrix}")
        print(f"投影仪分辨率: {projector_width}x{projector_height}")

        return projector_matrix, projector_width, projector_height
    except Exception as e:
        print(f"加载投影仪参数失败: {e}")
        return None, 1280, 800


def load_extrinsics(file_path):
    """
    加载外参数据

    参数:
        file_path: 外参文件路径

    返回:
        R: 旋转矩阵
        T: 平移向量
    """
    try:
        if file_path.endswith('.npy'):
            extrinsics_data = np.load(file_path, allow_pickle=True).item()
        else:
            with open(file_path, 'r') as f:
                extrinsics_data = json.load(f)

        R = np.array(extrinsics_data['R'])
        T = np.array(extrinsics_data['T'])

        print(f"已加载外参数据:")
        print(f"旋转矩阵R:\n{R}")
        print(f"平移向量T:\n{T}")

        return R, T
    except Exception as e:
        print(f"加载外参数据失败: {e}")
        return None, None


def load_unwrapped_phases(phase_x_path, phase_y_path):
    """
    加载解包裹相位数据

    参数:
        phase_x_path: X方向解包裹相位文件路径
        phase_y_path: Y方向解包裹相位文件路径

    返回:
        unwrapped_phase_x: X方向解包裹相位
        unwrapped_phase_y: Y方向解包裹相位
    """
    def read_phase_file(file_path):
        """读取单个相位文件，支持.npy和图像格式"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")

        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.npy':
            return np.load(file_path)
        elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            img = cv.imread(file_path, cv.IMREAD_UNCHANGED)
            if img is None:
                raise IOError(f"无法读取图像文件: {file_path}")

            if len(img.shape) == 3 and img.shape[2] > 1:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            return img.astype(np.float32)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")

    try:
        unwrapped_phase_x = read_phase_file(phase_x_path)
        unwrapped_phase_y = read_phase_file(phase_y_path)

        print(f"已加载解包裹相位数据:")
        print(f"X方向相位形状: {unwrapped_phase_x.shape}")
        print(f"Y方向相位形状: {unwrapped_phase_y.shape}")

        return unwrapped_phase_x, unwrapped_phase_y
    except Exception as e:
        print(f"加载解包裹相位数据失败: {e}")
        return None, None


def reconstruct_3d_scene_enhanced(unwrapped_phase_x, unwrapped_phase_y,
                                camera_matrix, projector_matrix, R, T,
                                projector_width, projector_height,
                                output_dir="enhanced_output", use_pso=True,
                                step_size=5, create_mesh=True, mask_percentile=98.0):
    """
    增强的三维重建主函数

    参数:
        unwrapped_phase_x, unwrapped_phase_y: 解包裹相位
        camera_matrix, projector_matrix: 相机和投影仪内参
        R, T: 外参
        projector_width, projector_height: 投影仪分辨率
        output_dir: 输出目录
        use_pso: 是否使用粒子群优化
        step_size: 采样步长
        create_mesh: 是否创建网格
        mask_percentile: 掩码阈值百分位数
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 创建重建对象
    reconstructor = Enhanced3DReconstruction(
        camera_matrix, projector_matrix, R, T, projector_width, projector_height
    )

    # 可视化解包裹相位
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(unwrapped_phase_x, cmap='jet')
    plt.colorbar(label='相位 (弧度)')
    plt.title("X方向解包裹相位")

    plt.subplot(122)
    plt.imshow(unwrapped_phase_y, cmap='jet')
    plt.colorbar(label='相位 (弧度)')
    plt.title("Y方向解包裹相位")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'unwrapped_phases.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 创建有效区域掩码
    print(f"创建有效区域掩码 (梯度阈值百分位数: {mask_percentile})...")
    mask = reconstructor.create_mask(unwrapped_phase_x, unwrapped_phase_y, mask_percentile)

    # 可视化掩码
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray')
    plt.title(f"有效区域掩码 (阈值百分位数: {mask_percentile}%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mask.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 从相位生成点云（使用改进的参数）
    if use_pso:
        optimization_method = "粒子群优化"
        print(f"使用{optimization_method}方法生成点云...")
        points, colors, qualities = reconstructor.phase_to_pointcloud_optimized(
            unwrapped_phase_x, unwrapped_phase_y, mask,
            use_pso=True, step_size=step_size, quality_threshold=150.0
        )
    else:
        optimization_method = "向量化三角测量"
        print(f"使用{optimization_method}方法生成点云...")
        points, colors, qualities = reconstructor.vectorized_triangulation(
            unwrapped_phase_x, unwrapped_phase_y, mask, step_size=step_size
        )

    if len(points) == 0:
        print("警告: 生成的点云为空")
        return

    # 过滤点云（使用改进的参数）
    print("过滤点云...")
    filtered_points, filtered_colors = reconstructor.filter_pointcloud(
        points, colors, qualities,
        quality_threshold=100.0,  # 更宽松的质量阈值
        enable_statistical_filter=False,  # 暂时禁用统计过滤
        preserve_density=True  # 保持点云密度
    )

    if len(filtered_points) == 0:
        print("警告: 过滤后的点云为空")
        return

    # 创建Open3D点云
    pcd = reconstructor.create_open3d_pointcloud(filtered_points, filtered_colors)

    # 可选的统计噪声去除（使用更温和的参数）
    if len(pcd.points) > 200:  # 只有足够多的点才进行统计过滤
        print("移除统计噪声点...")
        pcd_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=3.0)
        if len(pcd_filtered.points) > len(pcd.points) * 0.7:  # 至少保留70%的点
            pcd = pcd_filtered
            print(f"统计过滤后保留 {len(pcd.points)} 个点")
        else:
            print("统计过滤过于严格，跳过此步骤")
    else:
        print("点数较少，跳过统计噪声过滤")

    # 保存点云到多种格式
    print("保存重建结果...")
    saved_files = reconstructor.save_pointcloud_multiple_formats(
        filtered_points, filtered_colors,
        output_dir=output_dir,
        filename_base="enhanced_pointcloud",
        formats=['ply', 'pcd', 'xyz', 'txt', 'csv']
    )

    # 可视化点云
    print("可视化点云...")
    reconstructor.visualize_pointcloud(pcd, "增强三维重建结果")

    # 如果需要，创建并保存网格
    if create_mesh and len(pcd.points) > 100:
        try:
            print("从点云创建网格...")
            mesh = reconstructor.create_mesh_from_pointcloud(pcd, voxel_size=2.0, depth=8)

            # 保存网格到多种格式
            mesh_files = []

            # PLY格式
            ply_file = os.path.join(output_dir, 'enhanced_mesh.ply')
            o3d.io.write_triangle_mesh(ply_file, mesh)
            mesh_files.append(ply_file)
            print(f"✅ 网格PLY格式保存成功: {ply_file}")

            # OBJ格式
            try:
                obj_file = os.path.join(output_dir, 'enhanced_mesh.obj')
                o3d.io.write_triangle_mesh(obj_file, mesh)
                mesh_files.append(obj_file)
                print(f"✅ 网格OBJ格式保存成功: {obj_file}")
            except Exception as e:
                print(f"⚠️ OBJ格式保存失败: {e}")

            # STL格式
            try:
                stl_file = os.path.join(output_dir, 'enhanced_mesh.stl')
                o3d.io.write_triangle_mesh(stl_file, mesh)
                mesh_files.append(stl_file)
                print(f"✅ 网格STL格式保存成功: {stl_file}")
            except Exception as e:
                print(f"⚠️ STL格式保存失败: {e}")

            saved_files.extend(mesh_files)

            # 可视化网格
            print("可视化网格...")
            o3d.visualization.draw_geometries([mesh], window_name="增强三维重建网格")
        except Exception as e:
            print(f"网格创建失败: {e}")

    # 保存重建统计信息
    stats = {
        'total_points': len(points),
        'filtered_points': len(filtered_points),
        'optimization_method': optimization_method,
        'step_size': step_size,
        'mask_percentile': mask_percentile,
        'average_quality': float(np.mean(qualities)) if len(qualities) > 0 else 0,
        'depth_range': {
            'min': float(np.min(filtered_points[:, 2])) if len(filtered_points) > 0 else 0,
            'max': float(np.max(filtered_points[:, 2])) if len(filtered_points) > 0 else 0,
            'mean': float(np.mean(filtered_points[:, 2])) if len(filtered_points) > 0 else 0
        },
        'quality_statistics': {
            'min': float(np.min(qualities)) if len(qualities) > 0 else 0,
            'max': float(np.max(qualities)) if len(qualities) > 0 else 0,
            'median': float(np.median(qualities)) if len(qualities) > 0 else 0,
            'std': float(np.std(qualities)) if len(qualities) > 0 else 0
        },
        'saved_files': saved_files
    }

    stats_file = os.path.join(output_dir, 'reconstruction_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"✅ 统计信息保存成功: {stats_file}")

    print(f"\n重建完成! 统计信息:")
    print(f"- 原始点数: {stats['total_points']}")
    print(f"- 过滤后点数: {stats['filtered_points']}")
    print(f"- 优化方法: {stats['optimization_method']}")
    print(f"- 平均质量评分: {stats['average_quality']:.3f}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='增强的三维重建系统 - 集成粒子群优化')

    parser.add_argument('--camera-params', type=str, required=True,
                        help='相机参数文件路径 (.npy 或 .json)')

    parser.add_argument('--projector-params', type=str, required=True,
                        help='投影仪参数文件路径 (.json 或 .npy)')

    parser.add_argument('--extrinsics', type=str, required=True,
                        help='相机和投影仪之间的外参文件路径 (.npy 或 .json)')

    parser.add_argument('--phase-x', type=str, required=True,
                        help='X方向解包裹相位文件路径 (.npy 或 图像文件)')

    parser.add_argument('--phase-y', type=str, required=True,
                        help='Y方向解包裹相位文件路径 (.npy 或 图像文件)')

    parser.add_argument('--output-dir', type=str, default='enhanced_reconstruction_output',
                        help='输出目录路径')

    parser.add_argument('--use-pso', action='store_true', default=True,
                        help='使用粒子群优化 (默认启用)')

    parser.add_argument('--no-pso', action='store_true',
                        help='禁用粒子群优化，使用简化三角测量')

    parser.add_argument('--step-size', type=int, default=5,
                        help='采样步长 (默认: 5)')

    parser.add_argument('--create-mesh', action='store_true', default=True,
                        help='从点云创建网格 (默认启用)')

    parser.add_argument('--mask-percentile', type=float, default=98.0,
                        help='掩码阈值的百分位数 (默认: 98.0)')

    return parser.parse_args()


def main():
    """主函数"""
    # 确认Open3D是否可用
    try:
        import open3d as o3d
    except ImportError:
        print("警告: Open3D库未安装，将无法进行3D可视化和网格重建")
        print("可以使用 'pip install open3d' 安装此库")
        return

    print("=" * 60)
    print("增强的三维重建系统 - 集成粒子群优化")
    print("=" * 60)

    # 解析命令行参数
    try:
        args = parse_arguments()
        use_interactive = False
    except SystemExit:
        # 如果没有提供命令行参数，则使用交互模式
        print("\n未提供足够的命令行参数，切换到交互模式...\n")
        use_interactive = True
        args = None

    # 交互模式
    if use_interactive:
        print("==== 增强三维重建系统 ====")
        print("请提供以下参数以进行三维重建:\n")

        camera_params_path = input("1. 输入相机内参文件路径 (.npy 或 .json): ").strip()
        projector_params_path = input("2. 输入投影仪内参文件路径 (.json 或 .npy): ").strip()
        extrinsics_path = input("3. 输入相机和投影仪之间的外参文件路径 (.npy 或 .json): ").strip()
        phase_x_path = input("4. 输入X方向解包裹相位文件路径 (.npy 或 图像文件): ").strip()
        phase_y_path = input("5. 输入Y方向解包裹相位文件路径 (.npy 或 图像文件): ").strip()
        output_dir = input("6. 输入输出目录路径 (默认: enhanced_reconstruction_output): ").strip() or "enhanced_reconstruction_output"

        # 优化方法选择
        pso_choice = input("7. 是否使用粒子群优化? (y/n, 默认: y): ").strip().lower()
        use_pso = pso_choice != 'n'

        # 采样步长
        step_input = input("8. 输入采样步长 (默认: 5, 值越小精度越高但计算越慢): ").strip()
        step_size = int(step_input) if step_input.isdigit() else 5

        # 掩码阈值
        mask_input = input("9. 输入掩码阈值的百分位数 (默认: 98.0): ").strip()
        mask_percentile = float(mask_input) if mask_input else 98.0

        # 是否创建网格
        mesh_choice = input("10. 是否从点云创建网格? (y/n, 默认: y): ").strip().lower()
        create_mesh = mesh_choice != 'n'

    else:
        camera_params_path = args.camera_params
        projector_params_path = args.projector_params
        extrinsics_path = args.extrinsics
        phase_x_path = args.phase_x
        phase_y_path = args.phase_y
        output_dir = args.output_dir
        use_pso = args.use_pso and not args.no_pso
        step_size = args.step_size
        mask_percentile = args.mask_percentile
        create_mesh = args.create_mesh

    # 检查文件是否存在
    required_files = [camera_params_path, projector_params_path, extrinsics_path,
                     phase_x_path, phase_y_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"错误: 文件 '{file_path}' 不存在!")
            return

    # 加载参数
    print("\n" + "=" * 40)
    print("加载参数文件...")
    print("=" * 40)

    camera_matrix = load_camera_params(camera_params_path)
    projector_matrix, projector_width, projector_height = load_projector_params(projector_params_path)
    R, T = load_extrinsics(extrinsics_path)
    unwrapped_phase_x, unwrapped_phase_y = load_unwrapped_phases(phase_x_path, phase_y_path)

    # 检查是否所有参数都成功加载
    if None in [camera_matrix, projector_matrix, R, T, unwrapped_phase_x, unwrapped_phase_y]:
        print("错误: 未能成功加载所有必要参数!")
        return

    # 显示重建配置
    print("\n" + "=" * 40)
    print("重建配置:")
    print("=" * 40)
    print(f"优化方法: {'粒子群优化' if use_pso else '简化三角测量'}")
    print(f"采样步长: {step_size}")
    print(f"掩码阈值百分位数: {mask_percentile}")
    print(f"创建网格: {'是' if create_mesh else '否'}")
    print(f"输出目录: {output_dir}")

    # 执行三维重建
    print("\n" + "=" * 40)
    print("开始增强三维重建过程...")
    print("=" * 40)

    try:
        reconstruct_3d_scene_enhanced(
            unwrapped_phase_x, unwrapped_phase_y,
            camera_matrix, projector_matrix, R, T,
            projector_width, projector_height,
            output_dir, use_pso, step_size, create_mesh, mask_percentile
        )

        print("\n" + "=" * 60)
        print("增强三维重建完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n错误: 三维重建过程中发生异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
