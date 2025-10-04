import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
from typing import List, Optional, Tuple
from enum import Enum
from skimage import morphology
import skimage.filters as filters
import os

# 导入 Mask_generation.py 模块（权威的掩膜生成实现）
try:
    # 尝试直接导入（如果在同一目录或已在sys.path中）
    from Mask_generation import (
        generate_projection_mask,
        save_mask_visualization,
        PhaseShiftingAlgorithm as MG_PhaseShiftingAlgorithm,
    )
except ImportError:
    # 若直接导入失败，尝试从父目录导入
    import sys as _sys
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _parent_dir = os.path.dirname(_current_dir)
    if _parent_dir not in _sys.path:
        _sys.path.insert(0, _parent_dir)
    try:
        from Mask_generation import (
            generate_projection_mask,
            save_mask_visualization,
            PhaseShiftingAlgorithm as MG_PhaseShiftingAlgorithm,
        )
    except ImportError:
        # 最后尝试：从当前目录的父目录中导入
        if _current_dir not in _sys.path:
            _sys.path.insert(0, _current_dir)
        from Mask_generation import (
            generate_projection_mask,
            save_mask_visualization,
            PhaseShiftingAlgorithm as MG_PhaseShiftingAlgorithm,
        )

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


class PhaseShiftingAlgorithm(Enum):
    """相移算法类型枚举"""
    three_step = 0      # 三步相移
    four_step = 1       # 四步相移
    n_step = 2          # N步相移


# =============================================================================
# 掩膜生成函数（直接使用 Mask_generation.py 模块）
# =============================================================================

def generate_projection_mask_three_freq(images: List[np.ndarray], 
                                       algorithm: PhaseShiftingAlgorithm = PhaseShiftingAlgorithm.four_step,
                                       method: str = 'otsu', 
                                       thresh_rel: Optional[float] = None, 
                                       min_area: int = 500,
                                       confidence: float = 0.5,
                                       border_trim_px: int = 10,
                                       save_debug_images: bool = True,
                                       output_dir: Optional[str] = None) -> np.ndarray:
    """
    兼容接口适配器：直接调用 Mask_generation.py 的 generate_projection_mask 函数
    
    参数:
        images: 相移图像列表
        algorithm: 相移算法类型（本模块的枚举）
        method: 阈值化方法（固定为'otsu'）
        thresh_rel: 未使用（保留参数兼容性）
        min_area: 最小连通区域面积
        confidence: 掩膜置信度阈值
        border_trim_px: 边界收缩像素数
        save_debug_images: 是否保存调试图像
        output_dir: 输出目录
    
    返回:
        mask: 二值掩膜，True表示投影区域
    """
    if len(images) < 3:
        raise ValueError(f"至少需要3张图像，但只提供了{len(images)}张")
    
    # 简单的图像有效性检查
    all_zero = all(np.sum(img) == 0 for img in images)
    if all_zero:
        print(f"⚠ 警告: 传入的所有图像都是全零图像！")
        print(f"   这通常是因为图像加载错误或使用了占位的空白图像。")
    
    # 将本模块的 PhaseShiftingAlgorithm 映射到 Mask_generation 模块的枚举
    try:
        mg_alg = MG_PhaseShiftingAlgorithm[algorithm.name]
    except (KeyError, AttributeError):
        # 如果名称匹配失败，使用值进行判断
        if algorithm == PhaseShiftingAlgorithm.three_step:
            mg_alg = MG_PhaseShiftingAlgorithm.three_step
        elif algorithm == PhaseShiftingAlgorithm.four_step:
            mg_alg = MG_PhaseShiftingAlgorithm.four_step
        else:
            mg_alg = MG_PhaseShiftingAlgorithm.n_step
    
    
    # 直接调用 Mask_generation 模块的函数生成掩膜
    mask = generate_projection_mask(
        images=images,
        algorithm=mg_alg,
        method=method,
        thresh_rel=thresh_rel,
        min_area=min_area,
        confidence=confidence,
        border_trim_px=border_trim_px
    )
    
    # 保存可视化图像（调用 Mask_generation 模块的函数）
    if save_debug_images and output_dir is not None:
        mask_dir = os.path.join(output_dir, 'mask')
        os.makedirs(mask_dir, exist_ok=True)
        save_mask_visualization(images, mask, mask_dir)
    
    return mask.astype(bool)



class multi_phase():
    """
    多频外差法相位解包裹类
    
    该类实现了基于多频率条纹图像的相位解包裹算法，
    可同时处理水平和垂直方向的相位图，使用外差法逐级展开相位
    """
    def __init__(self, f, step, images, ph0, output_dir=None, save_intermediate=True, 
                 use_mask=False, mask_confidence=0.5):
        """
        初始化多频相位解包裹对象
        
        参数:
            f: 列表，包含多个频率值，按从高到低排序，例如[64,8,1]
            step: 整数，相移步数(通常为3或4)
            images: ndarray，所有条纹图像组成的数组
            ph0: 浮点数，相移初始相位偏移量
            output_dir: 字符串，输出目录路径（可选）
            save_intermediate: 布尔值，是否保存中间结果（默认True）
            use_mask: 布尔值，是否使用掩膜（默认False）
            mask_confidence: 浮点数，掩膜置信度 (0.1-0.9)，越高越严格
        """
        self.f = f                # 频率列表
        
        # 确保images是列表（如果是numpy数组，转换为列表以便索引）
        # 或者保持原样（UI传递的是列表）
        if isinstance(images, np.ndarray):
            # 如果已经是numpy数组，检查维度
            if images.ndim == 3:
                # 如果是3D数组 (n_images, height, width)，转换为列表
                self.images = [images[i] for i in range(images.shape[0])]
            else:
                self.images = images
        else:
            # 如果是列表，直接使用
            self.images = images
        
        self.step = step          # 相移步数
        self.ph0 = ph0            # 相移初始相位
        self.f12 = f[0] - f[1]    # 第1和第2个频率的差值(高频-中频)
        self.f23 = f[1] - f[2]    # 第2和第3个频率的差值(中频-低频)
        self.output_dir = output_dir  # 输出目录
        self.save_intermediate = save_intermediate  # 是否保存中间结果
        self.use_mask = use_mask  # 是否使用掩膜
        self.mask_confidence = mask_confidence  # 掩膜置信度
        self.mask = None          # 掩膜数组（将在get_phase中生成）

    

    def decode_phase(self,image):
        """
        N步相移算法解码相位
        
        使用正弦和余弦项计算相移图像的包裹相位，并计算幅值和偏移量
        
        参数:
            image: ndarray或列表，相移图像组，形状为[step, height, width]
            
        返回:
            result: ndarray，归一化的包裹相位图
            amp: ndarray，调制幅值
            offset: ndarray，亮度偏移
        """
        # 确保image是numpy数组
        # 如果是列表，需要堆叠成3D数组 [step, height, width]
        if isinstance(image, list):
            image = np.stack(image, axis=0)
        elif isinstance(image, np.ndarray):
            # 如果已经是数组，确保形状正确
            if image.ndim != 3:
                raise ValueError(f"图像数组维度不正确，期望3维 [step, height, width]，实际为 {image.ndim} 维")
        
        # 验证图像数量是否与相移步数匹配
        if image.shape[0] != self.step:
            raise ValueError(
                f"图像数量 ({image.shape[0]}) 与相移步数 ({self.step}) 不匹配。"
                f"请检查图像切片范围和相移步数设置。"
            )
        
        # 生成相移角度数组(0,2π/N,4π/N...)
        temp = 2*np.pi*np.arange(self.step,dtype=np.float32)/self.step
        temp.shape=-1,1,1  # 调整形状以便于广播运算
        
        # 计算正弦项(分子)和余弦项(分母)
        molecule = np.sum(image*np.sin(temp),axis=0)      # 正弦项
        denominator=np.sum(image*np.cos(temp),axis=0)     # 余弦项

        # 使用arctan2计算相位，保证相位值在[-π,π]范围内
        result = -np.arctan2(molecule,denominator)
        
        # 计算调制幅值和亮度偏移
        amp = 2/self.step*molecule        # 调制幅值
        offset = 2/self.step*denominator  # 亮度偏移

        # 归一化相位至[0,1]区间并减去初始相位
        result = (result+np.pi)/(2*np.pi)-self.ph0

        return result,amp,offset

    def phase_diff(self,image1,image2):
        """
        计算两个相位图之间的差值
        
        实现了外差法的核心操作，确保相位差在[0,1]范围内
        
        参数:
            image1: 高频相位图
            image2: 低频相位图
            
        返回:
            result: 两相位图的归一化差值
        """
        result = image1-image2       # 计算相位差
        result[result<0]+=1          # 处理负值，保证结果在[0,1]区间
        return result

    def unwarpphase(self,reference,phase,reference_f,phase_f):
        """
        基于低频参考相位展开高频相位
        
        参数:
            reference: 参考(低频)相位图
            phase: 需展开的(高频)包裹相位图
            reference_f: 参考相位的频率
            phase_f: 需展开相位的频率
            
        返回:
            unwarp_phase: 展开后的相位图
        """
        # 根据频率比例缩放参考相位
        # 低频相位乘以频率比得到高频相位的估计值
        temp = phase_f/reference_f*reference
        
        # 计算整数条纹序数k并应用
        # 用缩放后的低频相位减去高频包裹相位，四舍五入得到整数条纹序数
        k = np.round(temp-phase)
        unwarp_phase = phase + k
        
        # 高斯滤波去噪，检测错误跳变点
        # 使用更小的高斯核以保留更多细节
        gauss_size = (3, 3)
        unwarp_phase_noise = unwarp_phase - cv.GaussianBlur(unwarp_phase, gauss_size, 0)
        unwarp_reference_noise = temp - cv.GaussianBlur(temp, gauss_size, 0)

        # 改进异常点检测：降低阈值，增加相对比例判断
        noise_ratio = np.abs(unwarp_phase_noise) / (np.abs(unwarp_reference_noise) + 0.001)  # 避免除零
        order_flag = (np.abs(unwarp_phase_noise) - np.abs(unwarp_reference_noise) > 0.15) & (noise_ratio > 1.5)
        
        if np.sum(order_flag) > 0:  # 只在有异常点时进行修复
            # 修复异常跳变点
            unwarp_error = unwarp_phase[order_flag]
            unwarp_error_direct = unwarp_phase_noise[order_flag]
            
            # 根据噪声方向调整条纹序数
            unwarp_error[unwarp_error_direct > 0] -= 1  # 正向噪声减少一个周期
            unwarp_error[unwarp_error_direct < 0] += 1  # 负向噪声增加一个周期
            
            # 应用修复结果
            unwarp_phase[order_flag] = unwarp_error
            
            # 第二次高斯滤波去噪，进一步检测剩余的错误跳变点
            unwarp_phase_noise = unwarp_phase - cv.GaussianBlur(unwarp_phase, gauss_size, 0)
            order_flag2 = np.abs(unwarp_phase_noise) > 0.2
            
            if np.sum(order_flag2) > 0:
                unwarp_error2 = unwarp_phase[order_flag2]
                unwarp_error_direct2 = unwarp_phase_noise[order_flag2]
                
                # 根据噪声方向调整条纹序数
                unwarp_error2[unwarp_error_direct2 > 0] -= 1  # 正向噪声减少一个周期
                unwarp_error2[unwarp_error_direct2 < 0] += 1  # 负向噪声增加一个周期
                
                # 应用修复结果
                unwarp_phase[order_flag2] = unwarp_error2

        return unwarp_phase

    def post_process_phase(self, unwrap_phase, quality_map=None):
        """
        增强的后处理步骤：改善相位连续性和去除异常值，特别针对断层问题
        
        参数:
            unwrap_phase: 解包裹后的相位图
            quality_map: 相位质量图（可选）
            
        返回:
            processed_phase: 处理后的相位图
        """
        if not self.use_mask:
            return unwrap_phase
            
        processed_phase = unwrap_phase.copy()
        
        # 1. 检测和修复断层问题
        processed_phase = self._repair_phase_discontinuities(processed_phase, quality_map)
        
        # 2. 基于质量图的自适应平滑
        if quality_map is not None:
            processed_phase = self._adaptive_quality_smoothing(processed_phase, quality_map)
        else:
            # 没有质量图时使用保边平滑
            processed_phase = self._edge_preserving_smoothing(processed_phase)
        
        # 3. 检测和修复孤立异常点
        processed_phase = self._repair_outliers(processed_phase)
        
        # 4. 边界平滑处理
        processed_phase = self._smooth_boundaries(processed_phase)
        
        # 5. 最终确保掩膜外区域为0
        processed_phase[~self.mask] = 0
        
        return processed_phase

    def _repair_phase_discontinuities(self, phase, quality_map=None):
        """
        检测和修复相位断层问题
        
        参数:
            phase: 相位图
            quality_map: 质量图（可选）
            
        返回:
            repaired_phase: 修复后的相位图
        """
        repaired_phase = phase.copy()
        
        # 1. 检测大的相位跳跃（可能的断层）
        # 计算相位梯度
        grad_y, grad_x = np.gradient(repaired_phase)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 在掩膜区域内计算梯度统计
        if np.sum(self.mask) > 0:
            valid_gradients = gradient_magnitude[self.mask]
            grad_mean = np.mean(valid_gradients)
            grad_std = np.std(valid_gradients)
            
            # 使用更严格的阈值检测断层
            discontinuity_threshold = grad_mean + 4.0 * grad_std  # 4-sigma规则
            discontinuity_mask = (gradient_magnitude > discontinuity_threshold) & self.mask
            
            if np.sum(discontinuity_mask) > 0:
                print(f"检测到 {np.sum(discontinuity_mask)} 个可能的断层点，正在修复...")
                
                # 2. 使用形态学方法修复断层
                # 膨胀断层检测区域
                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
                expanded_discontinuity = cv.dilate(discontinuity_mask.astype(np.uint8), kernel)
                expanded_discontinuity = expanded_discontinuity.astype(bool) & self.mask
                
                # 3. 使用双边滤波修复断层区域
                # 双边滤波能保持边缘的同时平滑噪声
                bilateral_filtered = cv.bilateralFilter(
                    repaired_phase.astype(np.float32), 
                    d=9,           # 像素邻域直径
                    sigmaColor=0.1,  # 颜色空间滤波器的sigma值
                    sigmaSpace=2.0   # 坐标空间滤波器的sigma值
                )
                
                # 4. 在断层区域应用修复
                if quality_map is not None:
                    # 使用质量图加权混合
                    quality_weights = quality_map / (np.max(quality_map) + 1e-8)
                    quality_weights = np.clip(quality_weights, 0.1, 1.0)
                    
                    # 低质量区域更多地使用滤波结果
                    blend_factor = np.clip(1 - quality_weights, 0.3, 0.9)
                    repaired_phase[expanded_discontinuity] = (
                        repaired_phase[expanded_discontinuity] * (1 - blend_factor[expanded_discontinuity]) +
                        bilateral_filtered[expanded_discontinuity] * blend_factor[expanded_discontinuity]
                    )
                else:
                    # 在断层区域直接使用双边滤波结果
                    repaired_phase[expanded_discontinuity] = bilateral_filtered[expanded_discontinuity]
                
                # 5. 使用inpainting进一步修复严重断层
                severe_discontinuity = gradient_magnitude > (grad_mean + 6.0 * grad_std)
                severe_discontinuity = severe_discontinuity & self.mask
                
                if np.sum(severe_discontinuity) > 0:
                    # 使用OpenCV的inpainting算法修复严重断层
                    inpaint_mask = severe_discontinuity.astype(np.uint8) * 255
                    inpainted = cv.inpaint(
                        (repaired_phase * 255).astype(np.uint8),
                        inpaint_mask,
                        inpaintRadius=3,
                        flags=cv.INPAINT_TELEA
                    )
                    inpainted = inpainted.astype(np.float32) / 255.0
                    
                    # 只在严重断层区域应用inpainting结果
                    repaired_phase[severe_discontinuity] = inpainted[severe_discontinuity]
        
        return repaired_phase

    def _adaptive_quality_smoothing(self, phase, quality_map):
        """
        基于质量图的自适应平滑
        """
        # 使用质量图作为权重进行加权平滑
        weights = quality_map / (np.max(quality_map) + 1e-8)
        weights[~self.mask] = 0
        
        # 多尺度平滑
        smoothed_phase = phase.copy()
        
        for scale in [3, 5, 7]:  # 不同尺度的平滑
            kernel_size = (scale, scale)
            sigma = scale / 3.0
            
            smoothed = cv.GaussianBlur(smoothed_phase, kernel_size, sigma)
            
            # 根据质量权重和尺度调整混合因子
            scale_factor = scale / 7.0  # 归一化尺度因子
            blend_factor = np.clip((1 - weights) * scale_factor, 0.05, 0.3)
            
            smoothed_phase = smoothed_phase * (1 - blend_factor) + smoothed * blend_factor
            smoothed_phase[~self.mask] = 0
        
        return smoothed_phase

    def _edge_preserving_smoothing(self, phase):
        """
        保边平滑算法
        """
        # 使用双边滤波进行保边平滑
        smoothed = cv.bilateralFilter(
            phase.astype(np.float32),
            d=7,
            sigmaColor=0.05,
            sigmaSpace=1.5
        )
        
        # 轻微混合以保持原始细节
        result = phase * 0.7 + smoothed * 0.3
        result[~self.mask] = 0
        
        return result

    def _repair_outliers(self, phase):
        """
        检测和修复孤立异常点
        """
        # 使用多种滤波器检测异常点
        median_filtered = cv.medianBlur(phase.astype(np.float32), 5)
        gaussian_filtered = cv.GaussianBlur(phase, (5, 5), 1.0)
        
        # 计算与两种滤波结果的差异
        median_diff = np.abs(phase - median_filtered)
        gaussian_diff = np.abs(phase - gaussian_filtered)
        
        # 在掩膜区域内计算异常点阈值
        if np.sum(self.mask) > 0:
            median_threshold = np.percentile(median_diff[self.mask], 98)  # 更严格的阈值
            gaussian_threshold = np.percentile(gaussian_diff[self.mask], 98)
            
            # 同时满足两个条件才认为是异常点
            outlier_mask = (median_diff > median_threshold) & (gaussian_diff > gaussian_threshold) & self.mask
            
            if np.sum(outlier_mask) > 0:
                print(f"检测到 {np.sum(outlier_mask)} 个异常点，正在修复...")
                # 使用中值滤波结果替换异常点
                phase[outlier_mask] = median_filtered[outlier_mask]
        
        return phase

    def _smooth_boundaries(self, phase):
        """
        边界平滑处理
        """
        # 对掩膜边界附近的区域进行额外平滑
        boundary_mask = self._get_boundary_mask(boundary_width=8)  # 增加边界宽度
        
        if np.sum(boundary_mask) > 0:
            # 使用更强的平滑
            boundary_smoothed = cv.GaussianBlur(phase, (9, 9), 2.0)
            
            # 在边界区域逐渐混合
            # 计算到边界的距离，用于渐变混合
            distance_transform = cv.distanceTransform(
                (~self.mask).astype(np.uint8), 
                cv.DIST_L2, 
                5
            )
            
            # 归一化距离，用于计算混合权重
            max_dist = np.max(distance_transform[boundary_mask])
            if max_dist > 0:
                normalized_dist = distance_transform / max_dist
                # 距离边界越近，使用越多的平滑结果
                blend_factor = np.clip(1 - normalized_dist, 0.1, 0.6)
                
                phase[boundary_mask] = (
                    phase[boundary_mask] * (1 - blend_factor[boundary_mask]) + 
                    boundary_smoothed[boundary_mask] * blend_factor[boundary_mask]
                )
        
        return phase
    
    def _get_boundary_mask(self, boundary_width=5):
        """
        获取掩膜边界区域
        
        参数:
            boundary_width: 边界宽度（像素）
            
        返回:
            boundary_mask: 边界区域掩膜
        """
        if not self.use_mask:
            return np.zeros_like(self.mask, dtype=bool)
            
        # 膨胀和腐蚀操作来获取边界
        kernel = np.ones((boundary_width*2+1, boundary_width*2+1), np.uint8)
        dilated = cv.dilate(self.mask.astype(np.uint8), kernel, iterations=1)
        eroded = cv.erode(self.mask.astype(np.uint8), kernel, iterations=1)
        
        # 边界是膨胀后减去腐蚀后的区域
        boundary = (dilated - eroded) > 0
        
        # 只保留在原始掩膜内的边界
        boundary = boundary & self.mask
        
        return boundary
    
    def _save_phase_image(self, phase_data, filename, folder_name=None):
        """
        保存相位图像（2D可视化图）
        
        参数:
            phase_data: 相位数据
            filename: 文件名（不含路径）
            folder_name: 子文件夹名称（可选）
        """
        if not self.save_intermediate or self.output_dir is None:
            return
            
        # 确定保存路径
        if folder_name:
            save_dir = os.path.join(self.output_dir, folder_name)
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = self.output_dir
            
        # 归一化相位数据到0-255范围用于显示
        phase_normalized = cv.normalize(phase_data, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        
        # 应用jet色图
        phase_colored = cv.applyColorMap(phase_normalized, cv.COLORMAP_JET)
        
        # 保存图像
        save_path = os.path.join(save_dir, filename)
        cv.imwrite(save_path, phase_colored)
        
    def _save_phase_tiff(self, phase_data, filename, folder_name=None):
        """
        保存相位数据为TIFF格式（原始浮点数据）
        
        参数:
            phase_data: 相位数据
            filename: 文件名（不含路径）
            folder_name: 子文件夹名称（可选）
        """
        if self.output_dir is None:
            return
            
        # 确定保存路径
        if folder_name:
            save_dir = os.path.join(self.output_dir, folder_name)
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = self.output_dir
            
        # 保存TIFF文件
        save_path = os.path.join(save_dir, filename)
        cv.imwrite(save_path, phase_data.astype(np.float32))
    
    def get_phase(self):
        """
        多频相移解包裹主流程
        
        处理所有频率的相位图，分别对水平和垂直方向进行解包裹
        
        返回:
            unwarp_phase_y: 垂直方向展开的相位图
            unwarp_phase_x: 水平方向展开的相位图
            ratio: 相位质量图(基于调制度与偏移比)
            phase_2y: 垂直方向中频包裹相位
            phase_2x: 水平方向中频包裹相位
        """
        # 打印处理参数信息
        print(f"=" * 60)
        print(f"三频相位解包裹处理开始")
        print(f"相移步数: {self.step}")
        print(f"频率设置: {self.f}")
        print(f"总图像数: {len(self.images)}")
        print(f"期望图像数: {6 * self.step} (3个频率 × 2个方向 × {self.step}步)")
        print(f"=" * 60)
        
        # 重要说明：
        # 相移步数（3步或4步）只影响第1步"解码包裹相位"的计算
        # 一旦得到包裹相位后，后续的三频外差解包裹流程完全相同，
        # 与包裹相位是通过3步还是4步相移得到的无关
        
        # 0. 生成掩膜（如果启用）
        if self.use_mask:
            print(f"=" * 60)
            print(f"正在生成投影区域掩膜 (使用Otsu自适应阈值方法, 置信度: {self.mask_confidence})")
            print(f"=" * 60)
            
            step = self.step
            
            # 智能选择用于掩膜生成的图像：
            # 优先使用垂直高频图像（索引0:step），如果全零则使用水平高频图像（索引3*step:4*step）
            # 这样可以避免使用占位的全零图像
            
            # 先尝试垂直高频图像
            vertical_high_freq_images = [self.images[i] for i in range(step)]
            
            # 检查垂直高频图像是否有效（不是全零）
            vertical_valid = any(np.sum(img) > 0 for img in vertical_high_freq_images)
            
            if vertical_valid:
                # 使用垂直高频图像
                mask_images = vertical_high_freq_images
                print(f"使用垂直方向高频图像生成掩膜")
                print(f"图像索引范围: [0:{step}]")
            else:
                # 垂直图像全零，使用水平高频图像
                horizontal_high_freq_images = [self.images[i] for i in range(3*step, 4*step)]
                horizontal_valid = any(np.sum(img) > 0 for img in horizontal_high_freq_images)
                
                if horizontal_valid:
                    mask_images = horizontal_high_freq_images
                    print(f"垂直图像为占位图像（全零），改用水平方向高频图像生成掩膜")
                    print(f"图像索引范围: [{3*step}:{4*step}]")
                else:
                    # 两个方向都是全零，这是错误的
                    raise ValueError("垂直和水平方向的高频图像都是全零，无法生成掩膜！请检查图像加载是否正确。")
            
            print(f"相移步数: {step}")
            print(f"使用 {len(mask_images)} 张高频图像生成掩膜")
            print(f"相移算法: {step}步相移")
            
            # 打印图像信息用于调试
            for i, img in enumerate(mask_images):
                print(f"  图像{i+1}: 形状={img.shape}, 类型={img.dtype}, 最小值={np.min(img)}, 最大值={np.max(img)}")
            
            # 确定掩膜保存目录
            mask_dir = os.path.join(self.output_dir, 'mask')
            os.makedirs(mask_dir, exist_ok=True)
            print(f"掩膜保存目录: {mask_dir}")
            
            # 生成掩膜（直接使用 Mask_generation.py 模块）
            try:
                # 调用 generate_projection_mask_three_freq，它会内部调用 Mask_generation.py
                self.mask = generate_projection_mask_three_freq(
                    images=mask_images,
                    algorithm=PhaseShiftingAlgorithm.four_step if self.step == 4 else PhaseShiftingAlgorithm.three_step,
                    method='otsu',
                    confidence=self.mask_confidence,
                    min_area=500,
                    border_trim_px=10,
                    save_debug_images=True,  # 自动保存可视化
                    output_dir=self.output_dir  # 函数内部会创建 mask 子目录
                )
                
                # 统计掩膜覆盖率
                if self.mask is not None:
                    mask_coverage = np.sum(self.mask) / self.mask.size
                    print(f"✓ 掩膜生成完成")
                    print(f"✓ 有效区域覆盖率: {mask_coverage:.1%}")
                    print(f"✓ 有效像素数: {np.sum(self.mask)}")
                    
                    if mask_coverage < 0.1:
                        print(f"⚠ 警告: 掩膜覆盖率过低 ({mask_coverage:.1%})")
                        print(f"⚠ 建议: 调整掩膜参数或检查图像质量")
                else:
                    print(f"✗ 错误: 掩膜生成失败")
                    self.mask = None
            except Exception as e:
                print(f"✗ 掩膜生成异常: {str(e)}")
                import traceback
                traceback.print_exc()
                self.mask = None
            
            print(f"=" * 60)
        else:
            print("未启用掩膜，将处理整个图像")
            self.mask = None
        
        # ========================================================================
        # 第 1 步：解码各个频率的包裹相位（唯一与相移步数相关的步骤）
        # ========================================================================
        # 
        # 这是整个流程中唯一需要区分3步/4步相移的地方！
        # - 3步相移：每个频率使用3张图像计算包裹相位
        # - 4步相移：每个频率使用4张图像计算包裹相位
        #
        # 根据相移步数动态计算图像切片范围
        step = self.step  # 相移步数（3或4）
        
        print(f"\n第1步：使用 {step} 步相移算法解码包裹相位...")
        
        # 解码垂直方向的三个频率的相位
        phase_1y,amp1_y,offset1_y = self.decode_phase(image=self.images[0:step])           # 高频
        phase_2y,amp2_y,offset2_y = self.decode_phase(image=self.images[step:2*step])     # 中频
        phase_3y,amp3_y,offset3_y = self.decode_phase(image=self.images[2*step:3*step])   # 低频

        # 解码水平方向的三个频率的相位
        phase_1x,amp1_x,offset1_x = self.decode_phase(image=self.images[3*step:4*step])   # 高频
        phase_2x,amp2_x,offset2_x = self.decode_phase(image=self.images[4*step:5*step])   # 中频
        phase_3x,amp3_x,offset3_x = self.decode_phase(image=self.images[5*step:6*step])   # 低频
        
        print(f"✓ 包裹相位解码完成 (使用了 {step} 步相移算法)")
        print(f"\n" + "=" * 60)
        print(f"注意：后续的三频外差解包裹流程与相移步数无关")
        print(f"所有步骤只依赖于频率关系和已得到的包裹相位数值")
        print(f"=" * 60 + "\n")
        
        # 应用掩膜到包裹相位（如果启用）
        # 注意：掩膜只需在包裹相位阶段应用一次
        # 后续的解包裹过程是基于包裹相位的数学运算，会自动保持掩膜外区域的无效性
        if self.use_mask and self.mask is not None:
            print("应用掩膜到包裹相位（掩膜外区域将设为0）...")
            # 将掩膜外的区域设置为0
            phase_1y[~self.mask] = 0
            phase_2y[~self.mask] = 0
            phase_3y[~self.mask] = 0
            phase_1x[~self.mask] = 0
            phase_2x[~self.mask] = 0
            phase_3x[~self.mask] = 0
            print(f"✓ 已将掩膜外区域设为0，后续解包裹将基于这些包裹相位进行")
        
        # 保存包裹相位（三个频率的水平和垂直方向）
        self._save_phase_image(phase_1y, 'phase_1y_wrapped.png', '1_wrapped_phases')
        self._save_phase_image(phase_2y, 'phase_2y_wrapped.png', '1_wrapped_phases')
        self._save_phase_image(phase_3y, 'phase_3y_wrapped.png', '1_wrapped_phases')
        self._save_phase_image(phase_1x, 'phase_1x_wrapped.png', '1_wrapped_phases')
        self._save_phase_image(phase_2x, 'phase_2x_wrapped.png', '1_wrapped_phases')
        self._save_phase_image(phase_3x, 'phase_3x_wrapped.png', '1_wrapped_phases')

        # 注释掉所有中间过程的可视化代码
        """
        plt.figure()
        plt.imshow(phase_1x)
        plt.title('原始包裹相位: 水平高频 (1x)', fontproperties=font)
        plt.colorbar()
        plt.figure()
        plt.imshow(phase_2x)
        plt.title('原始包裹相位: 水平中频 (2x)', fontproperties=font)
        plt.colorbar()
        plt.figure()
        plt.imshow(phase_3x)
        plt.title('原始包裹相位: 水平低频 (3x)', fontproperties=font)
        plt.colorbar()
        """
        #plt.show()

        # ========================================================================
        # 第 2 步：外差法计算相位差（与相移步数无关）
        # ========================================================================
        #
        # 从这里开始，所有操作都与相移步数无关！
        # 无论包裹相位是用3步还是4步得到的，外差法的数学原理完全相同
        #
        print(f"第2步：外差法计算相位差...")
        
        # 计算垂直方向相位差
        phase_12y = self.phase_diff(phase_1y,phase_2y)  # 频率1和2的差异
        phase_23y = self.phase_diff(phase_2y,phase_3y)  # 频率2和3的差异
        phase_123y = self.phase_diff(phase_12y,phase_23y) # 差异的差异(等效最低频)

        # 计算水平方向相位差
        phase_12x = self.phase_diff(phase_1x,phase_2x)  # 频率1和2的差异
        phase_23x = self.phase_diff(phase_2x,phase_3x)  # 频率2和3的差异
        phase_123x = self.phase_diff(phase_12x,phase_23x) # 差异的差异(等效最低频)
        
        print(f"✓ 相位差计算完成")
        
        # 保存第一次多频外差结果
        self._save_phase_image(phase_12y, 'phase_12y.png', '2_first_heterodyne')
        self._save_phase_image(phase_23y, 'phase_23y.png', '2_first_heterodyne')
        self._save_phase_image(phase_12x, 'phase_12x.png', '2_first_heterodyne')
        self._save_phase_image(phase_23x, 'phase_23x.png', '2_first_heterodyne')

        # 注释掉所有中间过程的可视化代码
        """
        # 显示相位差结果
        plt.figure()
        plt.subplot(131)
        plt.imshow(phase_12x, cmap='jet')
        plt.title('相位差: 水平(高-中)', fontproperties=font)
        plt.colorbar()
        plt.subplot(132)
        plt.imshow(phase_23x, cmap='jet')
        plt.title('相位差: 水平(中-低)', fontproperties=font)
        plt.colorbar()
        plt.subplot(133)
        plt.imshow(phase_123x, cmap='jet')
        plt.title('相位差: 水平((高-中)-(中-低))', fontproperties=font)
        plt.colorbar()
        plt.tight_layout()

        plt.figure()
        plt.subplot(131)
        plt.imshow(phase_12y, cmap='jet')
        plt.title('相位差: 垂直(高-中)', fontproperties=font)
        plt.colorbar()
        plt.subplot(132)
        plt.imshow(phase_23y, cmap='jet')
        plt.title('相位差: 垂直(中-低)', fontproperties=font)
        plt.colorbar()
        plt.subplot(133)
        plt.imshow(phase_123y, cmap='jet')
        plt.title('相位差: 垂直((高-中)-(中-低))', fontproperties=font)
        plt.colorbar()
        plt.tight_layout()
        """
        #plt.show()

        # ========================================================================
        # 第 3 步：平滑最低等效频率相位（与相移步数无关）
        # ========================================================================
        print(f"\n第3步：平滑最低等效频率相位...")
        phase_123y = cv.GaussianBlur(phase_123y,(3,3),0)
        phase_123x = cv.GaussianBlur(phase_123x,(3,3),0)
        print(f"✓ 平滑完成")
        
        # 保存第二次多频外差结果
        self._save_phase_image(phase_123y, 'phase_123y.png', '3_second_heterodyne')
        self._save_phase_image(phase_123x, 'phase_123x.png', '3_second_heterodyne')

        # ========================================================================
        # 第 4 步：第一级相位展开（与相移步数无关）
        # ========================================================================
        print(f"\n第4步：第一级相位展开...")
        # 使用最低等效频率相位(phase_123y/x)展开中等频率相位差(phase_12y/x和phase_23y/x)
        unwarp_phase_12_y = self.unwarpphase(phase_123y,phase_12y,1,self.f12)
        unwarp_phase_23_y = self.unwarpphase(phase_123y,phase_23y,1,self.f23)

        unwarp_phase_12_x = self.unwarpphase(phase_123x,phase_12x,1,self.f12)
        unwarp_phase_23_x = self.unwarpphase(phase_123x,phase_23x,1,self.f23)
        print(f"✓ 第一级展开完成")
        
        # 保存相位展开流程结果
        self._save_phase_image(unwarp_phase_12_y, 'unwarp_phase_12_y.png', '4_phase_unwrapping')
        self._save_phase_image(unwarp_phase_23_y, 'unwarp_phase_23_y.png', '4_phase_unwrapping')
        self._save_phase_image(unwarp_phase_12_x, 'unwarp_phase_12_x.png', '4_phase_unwrapping')
        self._save_phase_image(unwarp_phase_23_x, 'unwarp_phase_23_x.png', '4_phase_unwrapping')
        
        # 注释掉所有中间过程的可视化代码
        """
        # 显示一级解包裹结果
        plt.figure()
        plt.subplot(121)
        plt.imshow(unwarp_phase_12_x, cmap='jet')
        plt.title('解包裹相位: 水平(高-中)', fontproperties=font)
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(unwarp_phase_23_x, cmap='jet')
        plt.title('解包裹相位: 水平(中-低)', fontproperties=font)
        plt.colorbar()
        plt.tight_layout()

        plt.figure()
        plt.subplot(121)
        plt.imshow(unwarp_phase_12_y, cmap='jet')
        plt.title('解包裹相位: 垂直(高-中)', fontproperties=font)
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(unwarp_phase_23_y, cmap='jet')
        plt.title('解包裹相位: 垂直(中-低)', fontproperties=font)
        plt.colorbar()
        plt.tight_layout()
        """
        
        # ========================================================================
        # 第 5 步：第二级相位展开（与相移步数无关）
        # ========================================================================
        print(f"\n第5步：第二级相位展开...")
        # 使用展开后的中等频率相位差(unwarp_phase_12_y/x和unwarp_phase_23_y/x)
        # 展开中频相位(phase_2y/x)
        unwarp_phase2_y_12 = self.unwarpphase(unwarp_phase_12_y,phase_2y,self.f12,self.f[1])
        unwarp_phase2_y_23 = self.unwarpphase(unwarp_phase_23_y,phase_2y,self.f23,self.f[1])

        unwarp_phase2_x_12 = self.unwarpphase(unwarp_phase_12_x,phase_2x,self.f12,self.f[1])
        unwarp_phase2_x_23 = self.unwarpphase(unwarp_phase_23_x,phase_2x,self.f23,self.f[1])
        print(f"✓ 第二级展开完成")
        
        # 保存最终结果（2D图像和TIFF）
        self._save_phase_image(unwarp_phase2_y_12/self.f[1], 'unwrap_phase2_y_12_2d.png', '5_final_results')
        self._save_phase_image(unwarp_phase2_y_23/self.f[1], 'unwrap_phase2_y_23_2d.png', '5_final_results')
        self._save_phase_image(unwarp_phase2_x_12/self.f[1], 'unwrap_phase2_x_12_2d.png', '5_final_results')
        self._save_phase_image(unwarp_phase2_x_23/self.f[1], 'unwrap_phase2_x_23_2d.png', '5_final_results')
        
        self._save_phase_tiff(unwarp_phase2_y_12/self.f[1], 'unwrap_phase2_y_12.tiff', '5_final_results')
        self._save_phase_tiff(unwarp_phase2_y_23/self.f[1], 'unwrap_phase2_y_23.tiff', '5_final_results')
        self._save_phase_tiff(unwarp_phase2_x_12/self.f[1], 'unwrap_phase2_x_12.tiff', '5_final_results')
        self._save_phase_tiff(unwarp_phase2_x_23/self.f[1], 'unwrap_phase2_x_23.tiff', '5_final_results')

        # 注释掉所有中间过程的可视化代码
        """
        # 显示二级解包裹结果（通过两个路径）
        plt.figure()
        plt.subplot(121)
        plt.imshow(unwarp_phase2_x_12/self.f[1], cmap='jet')
        plt.title('二级解包裹: 水平((高-中)路径)', fontproperties=font)
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(unwarp_phase2_x_23/self.f[1], cmap='jet')
        plt.title('二级解包裹: 水平((中-低)路径)', fontproperties=font)
        plt.colorbar()
        plt.tight_layout()

        plt.figure()
        plt.subplot(121)
        plt.imshow(unwarp_phase2_y_12/self.f[1], cmap='jet')
        plt.title('二级解包裹: 垂直((高-中)路径)', fontproperties=font)
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(unwarp_phase2_y_23/self.f[1], cmap='jet')
        plt.title('二级解包裹: 垂直((中-低)路径)', fontproperties=font)
        plt.colorbar()
        plt.tight_layout()
        """

        # ========================================================================
        # 第 6-7 步：两路径平均与归一化（与相移步数无关）
        # ========================================================================
        print(f"\n第6步：两路径平均...")
        # 6. 取两个展开路径的平均值以提高鲁棒性
        unwarp_phase_y = (unwarp_phase2_y_12+unwarp_phase2_y_23)/2
        unwarp_phase_x = (unwarp_phase2_x_12+unwarp_phase2_x_23)/2

        print(f"第7步：归一化相位结果...")
        # 7. 归一化相位结果
        unwarp_phase_y/=self.f[1]  # 以中频为基准归一化
        unwarp_phase_x/=self.f[1]  # 以中频为基准归一化
        print(f"✓ 三频外差解包裹完成")
        
        # 注意：不需要在这里再次应用掩膜
        # 因为包裹相位已经应用了掩膜，解包裹过程会保持掩膜外区域的无效性
        
        # 保存最终的水平和垂直方向解包裹相位（TIFF格式，保存在主输出目录）
        self._save_phase_tiff(unwarp_phase_y, 'unwrapped_phase_vertical.tiff')
        self._save_phase_tiff(unwarp_phase_x, 'unwrapped_phase_horizontal.tiff')

        # 注释掉所有中间过程的可视化代码
        """
        # 显示最终结果
        plt.figure()
        plt.subplot(121)
        plt.imshow(unwarp_phase_x, cmap='jet')
        plt.title('最终解包裹相位: 水平方向', fontproperties=font)
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(unwarp_phase_y, cmap='jet')
        plt.title('最终解包裹相位: 垂直方向', fontproperties=font)
        plt.colorbar()
        plt.tight_layout()
        """

        # ========================================================================
        # 第 8 步：计算相位质量图（与相移步数无关）
        # ========================================================================
        print(f"\n第8步：计算相位质量图...")
        # 8. 计算相位质量，使用调制度/偏移比值的最小值
        ratio_x = np.min([amp1_x/offset1_x,amp2_x/offset2_x,amp3_x/offset3_x],axis=0)
        ratio_y = np.min([amp1_y/offset1_y,amp2_y/offset2_y,amp3_y/offset3_y],axis=0)

        ratio = np.min([ratio_x,ratio_y],axis=0)  # 取水平和垂直方向的最小值作为最终质量图
        print(f"✓ 质量图计算完成")
        
        print(f"\n" + "=" * 60)
        print(f"三频相位解包裹处理全部完成！")
        print(f"=" * 60 + "\n")
        
        # 不再保存相位质量图文件
        
        # 注释掉所有中间过程的可视化代码
        """
        # 显示相位质量图
        plt.figure()
        plt.imshow(ratio, cmap='viridis')
        plt.title('相位质量图', fontproperties=font)
        plt.colorbar()
        """
        
        return unwarp_phase_y,unwarp_phase_x,ratio, phase_2y, phase_2x

"""
以下是被注释掉的旧版解码相位函数，仅供参考
def decode_phase(fore,phase_step,phase):
    
    #旧版相位解码函数，未被使用
    if len(phase) !=4:
        print("the image numbers of phase is {}".format(len(phase)))
        raise("please check out the phase")

    temp = 2*np.pi*np.arange(phase_step,dtype=np.float32)/phase_step
    temp.shape = -1,1,1
    molecule = np.sum(phase*np.sin(temp),axis=0)
    denominator=np.sum(phase*np.cos(temp),axis=0)

    result = -np.arctan2(molecule,denominator)

    #归一化
    result = (result+np.pi)/(2*np.pi)*fore

    return result
"""
