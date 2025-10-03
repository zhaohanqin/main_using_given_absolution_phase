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


def generate_projection_mask_three_freq(images: List[np.ndarray], 
                                       algorithm: PhaseShiftingAlgorithm = PhaseShiftingAlgorithm.four_step,
                                       method: str = 'adaptive', 
                                       thresh_rel: Optional[float] = None, 
                                       min_area: int = 500,
                                       confidence: float = 0.5,
                                       border_trim_px: int = 10,
                                       save_debug_images: bool = True,
                                       output_dir: Optional[str] = None) -> np.ndarray:
    """
    改进的三频相移图像投影区域掩膜生成
    专门针对条纹图像的特点进行优化，避免条纹明暗变化对掩膜的干扰
    
    参数:
        images: 相移图像列表
        algorithm: 相移算法类型
        method: 阈值化方法 ('otsu', 'relative', 'adaptive', 'modulation')
        thresh_rel: 相对阈值（仅用于relative方法）
        min_area: 最小连通区域面积
        confidence: 掩膜置信度阈值 (0.1-0.9)，控制掩膜的严格程度
        border_trim_px: 边界收缩像素数
        save_debug_images: 是否保存调试用的特征图像
        output_dir: 输出目录，如果提供则在其中创建mask文件夹保存图像
    
    返回:
        mask: 二值掩膜，True表示投影区域
    """
    if len(images) < 3:
        raise ValueError(f"至少需要3张图像，但只提供了{len(images)}张")
    
    # 计算基础特征
    imgs = np.stack([img.astype(np.float32) for img in images], axis=2)
    I_max = np.max(imgs, axis=2)
    I_min = np.min(imgs, axis=2)
    I_mean = np.mean(imgs, axis=2)
    I_std = np.std(imgs, axis=2)

    # 关键改进：使用调制度作为主要特征，而不是亮度
    # 调制度能更好地区分投影区域和背景，不受条纹明暗变化影响
    modulation = (I_max - I_min) / (I_max + I_min + 1e-9)  # 调制度
    amplitude = (I_max - I_min) / 2.0  # 振幅
    
    # 计算相位稳定性 - 投影区域的相位变化应该是规律的
    phase_stability = I_std / (I_mean + 1e-9)
    
    # 计算亮度对比度 - 用于辅助判断
    brightness_contrast = I_std / (I_mean + 1e-9)
    
    # 应用噪声抑制
    modulation, amplitude, phase_stability = _apply_noise_reduction_improved(
        modulation, amplitude, phase_stability, I_mean)
    
    # 归一化特征到0-255范围
    mod_norm = cv.normalize(modulation, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    amp_norm = cv.normalize(amplitude, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    stab_norm = cv.normalize(phase_stability, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    mean_norm = cv.normalize(I_mean, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    
    # 保存调试图像到mask文件夹
    if save_debug_images and output_dir is not None:
        mask_dir = os.path.join(output_dir, 'mask')
        os.makedirs(mask_dir, exist_ok=True)
        
        # 保存特征图像
        cv.imwrite(os.path.join(mask_dir, 'Modulation.png'), mod_norm)
        cv.imwrite(os.path.join(mask_dir, 'Amplitude.png'), amp_norm)
        cv.imwrite(os.path.join(mask_dir, 'Phase_Stability.png'), stab_norm)
        cv.imwrite(os.path.join(mask_dir, 'Mean_Intensity.png'), mean_norm)
        
        print(f"特征图像已保存到: {mask_dir}")
        
        # 保存原始特征的浮点数版本（用于分析）
        np.save(os.path.join(mask_dir, 'modulation_raw.npy'), modulation)
        np.save(os.path.join(mask_dir, 'amplitude_raw.npy'), amplitude)
        np.save(os.path.join(mask_dir, 'phase_stability_raw.npy'), phase_stability)
        np.save(os.path.join(mask_dir, 'mean_intensity_raw.npy'), I_mean)
    
    # 选择改进的阈值化方法
    if method == 'modulation':
        mask = _modulation_based_thresholding(mod_norm, amp_norm, mean_norm, confidence)
    elif method == 'adaptive':
        mask = _adaptive_thresholding_improved(mod_norm, amp_norm, stab_norm, mean_norm, confidence)
    elif method == 'otsu':
        mask = _otsu_thresholding_improved(mod_norm, amp_norm, mean_norm)
    elif method == 'relative':
        mask = _relative_thresholding_improved(mod_norm, amp_norm, mean_norm, thresh_rel, confidence)
    else:
        raise ValueError(f"不支持的阈值化方法: {method}")
    
    print(f"掩膜生成参数: 方法={method}, 置信度={confidence:.2f}")
    
    # 改进的形态学处理
    mask = _improved_morphological_processing(mask, min_area)
    
    # 改进的边界处理
    mask = _improved_border_processing(mask, border_trim_px)
    
    # 改进的连通性优化
    mask = _improved_connectivity_optimization(mask)
    
    # 最终的投影区域完整性处理
    mask = _ensure_projection_integrity(mask)
    
    # 保存最终掩膜图像
    if save_debug_images and output_dir is not None:
        mask_dir = os.path.join(output_dir, 'mask')
        os.makedirs(mask_dir, exist_ok=True)
        
        # 保存最终掩膜
        final_mask_img = mask.astype(np.uint8) * 255
        cv.imwrite(os.path.join(mask_dir, 'final_mask.png'), final_mask_img)
        cv.imwrite(os.path.join(mask_dir, 'Final Mask.png'), final_mask_img)  # 与unwrap_phase.py保持一致
        
        # 保存掩膜的numpy数组
        np.save(os.path.join(mask_dir, 'final_mask.npy'), mask)
        
        # 计算并保存掩膜统计信息
        total_pixels = mask.size
        valid_pixels = np.sum(mask)
        coverage_ratio = valid_pixels / total_pixels
        
        # 计算连通分量信息
        num_labels, labels = cv.connectedComponents(mask.astype(np.uint8))
        num_components = num_labels - 1  # 减去背景
        
        # 计算最大连通分量的面积
        if num_components > 0:
            areas = [(labels == i).sum() for i in range(1, num_labels)]
            max_component_area = max(areas)
            max_component_ratio = max_component_area / total_pixels
        else:
            max_component_area = 0
            max_component_ratio = 0
        
        stats_file = os.path.join(mask_dir, 'mask_stats.txt')
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(f"掩膜生成统计信息\n")
            f.write(f"================\n")
            f.write(f"方法: {method}\n")
            f.write(f"置信度: {confidence:.2f}\n")
            f.write(f"总像素数: {total_pixels}\n")
            f.write(f"有效像素数: {valid_pixels}\n")
            f.write(f"覆盖率: {coverage_ratio:.1%}\n")
            f.write(f"连通分量数: {num_components}\n")
            f.write(f"最大分量面积: {max_component_area}\n")
            f.write(f"最大分量占比: {max_component_ratio:.1%}\n")
            f.write(f"最小连通区域面积: {min_area}\n")
            f.write(f"边界收缩像素数: {border_trim_px}\n")
            f.write(f"空洞填充: 已启用 (投影区域完整性保证)\n")
        
        print(f"掩膜图像和统计信息已保存到: {mask_dir}")
        print(f"最终掩膜覆盖率: {coverage_ratio:.1%}")
    
    return mask.astype(bool)


def _apply_noise_reduction_improved(modulation, amplitude, phase_stability, I_mean):
    """改进的噪声抑制方法"""
    # 使用更温和的高斯滤波
    mod_filtered = cv.GaussianBlur(modulation, (3, 3), 0.8)
    amp_filtered = cv.GaussianBlur(amplitude, (3, 3), 0.8)
    stab_filtered = cv.GaussianBlur(phase_stability, (3, 3), 0.8)
    
    # 检测异常区域 - 低亮度且高噪声的区域
    noise_mask = (phase_stability > np.percentile(phase_stability, 90)) & \
                 (I_mean < np.percentile(I_mean, 25))
    
    # 在噪声区域使用滤波后的值
    modulation = np.where(noise_mask, mod_filtered, modulation)
    amplitude = np.where(noise_mask, amp_filtered, amplitude)
    phase_stability = np.where(noise_mask, stab_filtered, phase_stability)
    
    return modulation, amplitude, phase_stability

def _modulation_based_thresholding(mod_norm, amp_norm, mean_norm, confidence):
    """基于调制度的阈值化方法 - 最适合条纹图像"""
    # 分析调制度分布
    valid_mod = mod_norm[mod_norm > 0]
    valid_amp = amp_norm[amp_norm > 0]
    
    if len(valid_mod) == 0 or len(valid_amp) == 0:
        return np.zeros_like(mod_norm, dtype=bool)
    
    # 动态调制度阈值 - 根据置信度调整
    # 低置信度使用更低的阈值，高置信度使用更高的阈值
    mod_percentile = max(10, 50 - confidence * 30)  # 从20-50%范围调整
    mod_threshold = max(10, np.percentile(valid_mod, mod_percentile))
    
    # 动态振幅阈值
    amp_percentile = max(15, 40 - confidence * 20)  # 从20-40%范围调整
    amp_threshold = max(10, np.percentile(valid_amp, amp_percentile))
    
    # 更宽松的亮度阈值 - 不要过度排除暗区域
    brightness_threshold = max(5, np.percentile(mean_norm, 15))
    
    # 基础掩膜
    basic_mask = (mod_norm >= mod_threshold) & \
                 (amp_norm >= amp_threshold) & \
                 (mean_norm > brightness_threshold)
    
    # 如果基础掩膜覆盖率太低，进一步降低阈值
    coverage = np.sum(basic_mask) / basic_mask.size
    
    if coverage < 0.6:  # 如果覆盖率低于60%
        # 进一步降低阈值
        mod_threshold = max(5, np.percentile(valid_mod, max(5, mod_percentile - 15)))
        amp_threshold = max(5, np.percentile(valid_amp, max(10, amp_percentile - 10)))
        brightness_threshold = max(3, np.percentile(mean_norm, 10))
        
        basic_mask = (mod_norm >= mod_threshold) & \
                     (amp_norm >= amp_threshold) & \
                     (mean_norm > brightness_threshold)
    
    return basic_mask

def _adaptive_thresholding_improved(mod_norm, amp_norm, stab_norm, mean_norm, confidence):
    """改进的自适应阈值化方法"""
    # 基于调制度的多级阈值
    mod_high = np.percentile(mod_norm[mod_norm > 0], 70)
    mod_medium = np.percentile(mod_norm[mod_norm > 0], 50)
    mod_low = np.percentile(mod_norm[mod_norm > 0], 30)
    
    # 基于振幅的阈值
    amp_threshold = np.percentile(amp_norm[amp_norm > 0], 40)
    
    # 亮度阈值
    brightness_threshold = np.percentile(mean_norm, 25)
    
    # 根据置信度选择阈值策略
    if confidence >= 0.7:
        # 高置信度：严格标准
        mask = (mod_norm >= mod_high) & (amp_norm >= amp_threshold) & (mean_norm > brightness_threshold)
    elif confidence >= 0.4:
        # 中等置信度：平衡标准
        mask = (mod_norm >= mod_medium) & (amp_norm >= amp_threshold * 0.8) & (mean_norm > brightness_threshold)
    else:
        # 低置信度：宽松标准
        mask = (mod_norm >= mod_low) & (amp_norm >= amp_threshold * 0.6) & (mean_norm > brightness_threshold * 0.8)
    
    return mask

def _otsu_thresholding_improved(mod_norm, amp_norm, mean_norm):
    """改进的Otsu阈值化方法"""
    # 对调制度使用Otsu阈值
    _, mask_mod = cv.threshold(mod_norm, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    # 对振幅使用较低的阈值
    amp_threshold = max(20, np.percentile(amp_norm[amp_norm > 0], 25))
    mask_amp = amp_norm >= amp_threshold
    
    # 亮度阈值
    brightness_threshold = np.percentile(mean_norm, 20)
    mask_brightness = mean_norm > brightness_threshold
    
    # 组合掩膜：调制度为主，振幅和亮度为辅
    mask = (mask_mod > 0) & mask_amp & mask_brightness
    
    return mask

def _relative_thresholding_improved(mod_norm, amp_norm, mean_norm, thresh_rel, confidence):
    """改进的相对阈值化方法"""
    if thresh_rel is None:
        thresh_rel = 0.3
    
    # 调整阈值参数
    adjusted_thresh = thresh_rel * (1.5 - confidence * 0.5)
    
    mod_threshold = np.percentile(mod_norm, 100 * (1 - adjusted_thresh))
    amp_threshold = np.percentile(amp_norm, 100 * (1 - adjusted_thresh * 0.8))
    brightness_threshold = np.percentile(mean_norm, 25)
    
    mask = (mod_norm >= mod_threshold) & \
           (amp_norm >= amp_threshold) & \
           (mean_norm > brightness_threshold)
    
    return mask

def _improved_morphological_processing(mask, min_area):
    """改进的形态学处理 - 专门针对投影区域的完整性优化"""
    mask = mask.astype(np.uint8)
    
    # 第一步：轻微的开运算，去除小的噪声点
    kernel_small = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_small)
    
    # 第二步：强化的闭运算，连接近邻区域
    kernel_medium = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel_medium)
    
    # 第三步：移除小的连通区域
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_area//4)
    
    # 第四步：强化空洞填充 - 投影仪投影的是完整面，不应该有空洞
    # 使用更大的阈值填充所有合理大小的空洞
    mask = morphology.remove_small_holes(mask, area_threshold=min_area * 2)  # 增加阈值
    
    # 第五步：额外的空洞填充 - 使用形态学闭运算进一步填充
    kernel_large = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    mask = cv.morphologyEx(mask.astype(np.uint8), cv.MORPH_CLOSE, kernel_large)
    
    # 第六步：再次强化空洞填充
    mask = morphology.remove_small_holes(mask.astype(bool), area_threshold=min_area * 5)
    
    # 第七步：最终的小区域清理
    mask = morphology.remove_small_objects(mask, min_size=min_area)
    
    return mask.astype(np.uint8)

def _improved_border_processing(mask, border_trim_px):
    """改进的边界处理"""
    if not border_trim_px or border_trim_px <= 0:
        return mask
    
    h, w = mask.shape
    # 限制边界处理的范围
    trim_size = min(border_trim_px, min(h // 20, w // 20))
    
    if trim_size > 0:
        # 检查边界区域的掩膜密度
        border_density_threshold = 0.15
        
        # 上边界
        if np.mean(mask[:trim_size, :]) < border_density_threshold:
            mask[:trim_size, :] = 0
        
        # 下边界
        if np.mean(mask[-trim_size:, :]) < border_density_threshold:
            mask[-trim_size:, :] = 0
        
        # 左边界
        if np.mean(mask[:, :trim_size]) < border_density_threshold:
            mask[:, :trim_size] = 0
        
        # 右边界
        if np.mean(mask[:, -trim_size:]) < border_density_threshold:
            mask[:, -trim_size:] = 0
    
    return mask

def _improved_connectivity_optimization(mask):
    """改进的连通性优化"""
    # 找到所有连通分量
    num_labels, labels = cv.connectedComponents(mask.astype(np.uint8))
    
    if num_labels <= 2:  # 只有背景和一个前景
        return mask
    
    # 计算每个连通分量的面积和紧密度
    areas = []
    compactness = []
    
    for i in range(1, num_labels):
        component_mask = (labels == i)
        area = np.sum(component_mask)
        areas.append(area)
        
        # 计算紧密度（面积与外接矩形面积的比值）
        if area > 0:
            coords = np.where(component_mask)
            if len(coords[0]) > 0:
                bbox_area = (np.max(coords[0]) - np.min(coords[0]) + 1) * \
                           (np.max(coords[1]) - np.min(coords[1]) + 1)
                compact = area / bbox_area if bbox_area > 0 else 0
            else:
                compact = 0
        else:
            compact = 0
        compactness.append(compact)
    
    if len(areas) == 0:
        return mask
    
    # 选择保留的连通分量
    max_area = max(areas)
    area_threshold = max_area * 0.2  # 面积阈值
    compactness_threshold = 0.1      # 紧密度阈值
    
    new_mask = np.zeros_like(mask)
    for i, (area, compact) in enumerate(zip(areas, compactness)):
        # 保留大面积或者紧密的连通分量
        if area >= area_threshold or (area >= max_area * 0.05 and compact >= compactness_threshold):
            new_mask[labels == (i + 1)] = 1
    
    return new_mask.astype(np.uint8)

def _ensure_projection_integrity(mask):
    """
    确保投影区域的完整性和边界平滑性
    投影仪投影时是一个完整的面，不应该有内部空洞，且边界应该相对平滑
    """
    mask = mask.astype(np.uint8)
    
    # 1. 找到最大的连通分量（主要投影区域）
    num_labels, labels = cv.connectedComponents(mask)
    
    if num_labels <= 1:  # 没有前景区域
        return mask
    
    # 找到最大连通分量
    areas = [(labels == i).sum() for i in range(1, num_labels)]
    if len(areas) == 0:
        return mask
    
    max_area_idx = np.argmax(areas) + 1
    main_projection = (labels == max_area_idx).astype(np.uint8)
    
    # 2. 边界平滑处理
    # 使用形态学开运算和闭运算来平滑边界
    kernel_smooth = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    
    # 先进行闭运算填充小的凹陷
    smoothed = cv.morphologyEx(main_projection, cv.MORPH_CLOSE, kernel_smooth)
    # 再进行开运算去除小的突起
    smoothed = cv.morphologyEx(smoothed, cv.MORPH_OPEN, kernel_smooth)
    
    # 3. 计算主投影区域的凸包（用于参考）
    contours, _ = cv.findContours(smoothed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return mask
    
    # 找到最大轮廓
    main_contour = max(contours, key=cv.contourArea)
    
    # 计算凸包
    hull = cv.convexHull(main_contour)
    
    # 4. 创建凸包掩膜
    hull_mask = np.zeros_like(mask)
    cv.fillPoly(hull_mask, [hull], 255)
    
    # 5. 渐进式边界平滑
    # 使用多次小幅度的形态学操作来逐步平滑边界
    final_mask = smoothed.copy()
    
    # 多次小幅度的平滑操作
    for i in range(3):
        kernel_size = 3 + i * 2  # 逐渐增大核大小 (3, 5, 7)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 轻微的闭运算
        temp = cv.morphologyEx(final_mask, cv.MORPH_CLOSE, kernel)
        
        # 只在凸包范围内应用平滑
        final_mask = np.where(hull_mask > 0, temp, final_mask)
    
    # 6. 填充内部空洞
    final_mask = morphology.remove_small_holes(final_mask.astype(bool), area_threshold=1000)
    
    # 7. 最终边界优化：使用高斯模糊 + 阈值化来进一步平滑边界
    # 将二值掩膜转换为浮点数
    float_mask = final_mask.astype(np.float32)
    
    # 应用轻微的高斯模糊
    blurred = cv.GaussianBlur(float_mask, (5, 5), 1.0)
    
    # 重新二值化，使用稍低的阈值以保持区域大小
    _, final_mask = cv.threshold(blurred, 0.3, 1.0, cv.THRESH_BINARY)
    
    # 8. 最后一次连通性检查和小区域清理
    final_mask = final_mask.astype(np.uint8)
    final_mask = morphology.remove_small_objects(final_mask.astype(bool), min_size=500)
    final_mask = morphology.remove_small_holes(final_mask, area_threshold=2000)
    
    return final_mask.astype(np.uint8)

def _apply_noise_reduction_three_freq(A, M, S, I_mean):
    """应用噪声抑制 - 保持向后兼容"""
    return _apply_noise_reduction_improved(M, A, S, I_mean)


def _adaptive_thresholding_three_freq(A_norm, M_norm, S_norm, I_norm, confidence):
    """自适应阈值化方法"""
    # 计算每个特征的多级阈值
    th_A_low = np.percentile(A_norm[A_norm > 0], 25)
    th_A_high = np.percentile(A_norm[A_norm > 0], 60)
    
    th_M_low = np.percentile(M_norm[M_norm > 0], 25)
    th_M_high = np.percentile(M_norm[M_norm > 0], 60)
    
    th_I = np.percentile(I_norm, 40)
    
    # 稳定性阈值
    th_S = np.percentile(S_norm, 75)
    
    # 多级判断
    high_confidence = (A_norm >= th_A_high) & (M_norm >= th_M_high)
    medium_confidence = ((A_norm >= th_A_low) | (M_norm >= th_M_low)) & (S_norm <= th_S)
    low_confidence = (A_norm >= th_A_low) | (M_norm >= th_M_low)
    
    # 强度约束
    intensity_valid = I_norm > th_I
    
    # 根据置信度参数选择区域
    if confidence >= 0.8:
        mask = high_confidence & intensity_valid
    elif confidence >= 0.6:
        mask = high_confidence & intensity_valid
    elif confidence >= 0.4:
        mask = (high_confidence | (medium_confidence & (A_norm >= th_A_high * 0.8))) & intensity_valid
    elif confidence >= 0.2:
        mask = (high_confidence | medium_confidence) & intensity_valid
    else:
        mask = low_confidence & intensity_valid
    
    return mask


def _otsu_thresholding_three_freq(A_norm, M_norm, I_norm):
    """改进的Otsu方法"""
    _, mask_A = cv.threshold(A_norm, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    _, mask_M = cv.threshold(M_norm, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    # 降低强度阈值，使更多区域通过
    th_I = np.percentile(I_norm, 30)
    mask_I = (I_norm > th_I).astype(np.uint8) * 255
    
    mask = np.logical_or(mask_A > 0, mask_M > 0)
    mask = np.logical_and(mask, mask_I > 0)
    
    return mask


def _relative_thresholding_three_freq(A_norm, M_norm, I_norm, thresh_rel, confidence):
    """改进的相对阈值方法"""
    if thresh_rel is None:
        thresh_rel = 0.3
    
    # 调整相对阈值计算，使其更宽松
    adjusted_thresh_rel = float(thresh_rel) * (2.5 - float(confidence))
    th_A = np.percentile(A_norm, 100 * (1 - adjusted_thresh_rel))
    th_M = np.percentile(M_norm, 100 * (1 - adjusted_thresh_rel))
    th_I = np.percentile(I_norm, 30)
    
    mask = np.logical_or(A_norm >= th_A, M_norm >= th_M)
    mask = np.logical_and(mask, I_norm > th_I)
    
    return mask


def _progressive_morphological_processing_three_freq(mask, min_area):
    """渐进式形态学处理"""
    mask = mask.astype(np.uint8)
    
    # 更温和的形态学处理
    mask = morphology.binary_closing(mask, morphology.disk(2))
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_area//8)
    mask = morphology.binary_closing(mask, morphology.disk(3))
    mask = morphology.remove_small_holes(mask, area_threshold=min_area//4)
    mask = morphology.remove_small_objects(mask, min_size=min_area//2)
    
    return mask.astype(np.uint8)


def _smart_border_trimming_three_freq(mask, border_trim_px):
    """智能边界处理"""
    if not border_trim_px or border_trim_px <= 0:
        return mask
    
    h, w = mask.shape
    bt = min(border_trim_px, min(h // 15, w // 15))
    
    if bt > 0:
        border_noise_ratio = 0.1
        
        # 检查上下边界
        top_ratio = np.mean(mask[:bt, :])
        bottom_ratio = np.mean(mask[-bt:, :])
        left_ratio = np.mean(mask[:, :bt])
        right_ratio = np.mean(mask[:, -bt:])
        
        if top_ratio < border_noise_ratio:
            mask[:bt, :] = 0
        if bottom_ratio < border_noise_ratio:
            mask[-bt:, :] = 0
        if left_ratio < border_noise_ratio:
            mask[:, :bt] = 0
        if right_ratio < border_noise_ratio:
            mask[:, -bt:] = 0
    
    return mask


def _connectivity_optimization_three_freq(mask):
    """连通性优化"""
    # 保留最大的几个连通分量
    num_labels, labels = cv.connectedComponents(mask.astype(np.uint8))
    
    if num_labels <= 2:  # 背景 + 1个前景
        return mask
    
    # 计算每个连通分量的面积
    areas = [(labels == i).sum() for i in range(1, num_labels)]
    
    if len(areas) == 0:
        return mask
    
    # 保留最大的连通分量，以及面积超过最大面积30%的其他分量
    max_area = max(areas)
    area_threshold = max_area * 0.3
    
    new_mask = np.zeros_like(mask)
    for i, area in enumerate(areas):
        if area >= area_threshold:
            new_mask[labels == (i + 1)] = 1
    
    return new_mask.astype(np.uint8)

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
        self.images = images      # 相移图像
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
            image: ndarray，相移图像组，形状为[step, height, width]
            
        返回:
            result: ndarray，归一化的包裹相位图
            amp: ndarray，调制幅值
            offset: ndarray，亮度偏移
        """
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
        # 0. 生成掩膜（如果启用）
        if self.use_mask:
            print(f"正在生成投影区域掩膜 (使用Otsu自适应阈值方法, 置信度: {self.mask_confidence})...")
            
            # 使用所有图像生成掩膜（可以使用全部24张图像以获得更好的掩膜质量）
            all_images_list = [self.images[i] for i in range(len(self.images))]
            
            self.mask = generate_projection_mask_three_freq(
                images=all_images_list,
                algorithm=PhaseShiftingAlgorithm.four_step if self.step == 4 else PhaseShiftingAlgorithm.three_step,
                method='otsu',  # 固定使用otsu方法
                confidence=self.mask_confidence,
                min_area=500,
                border_trim_px=10,
                save_debug_images=True,
                output_dir=self.output_dir
            )
            
            # 统计掩膜覆盖率
            mask_coverage = np.sum(self.mask) / self.mask.size
            print(f"掩膜生成完成，有效区域覆盖率: {mask_coverage:.1%}")
            
            if mask_coverage < 0.1:
                print(f"警告: 掩膜覆盖率过低 ({mask_coverage:.1%})，建议调整掩膜参数或检查图像质量")
        else:
            print("未启用掩膜，将处理整个图像")
            self.mask = None
        
        # 1. 解码各个频率的相位
        # 解码垂直方向的三个频率的相位
        phase_1y,amp1_y,offset1_y = self.decode_phase(image=self.images[0:4])   # 高频
        phase_2y,amp2_y,offset2_y = self.decode_phase(image=self.images[4:8])   # 中频
        phase_3y,amp3_y,offset3_y = self.decode_phase(image=self.images[8:12])  # 低频

        # 解码水平方向的三个频率的相位
        phase_1x,amp1_x,offset1_x = self.decode_phase(image=self.images[12:16]) # 高频
        phase_2x,amp2_x,offset2_x = self.decode_phase(image=self.images[16:20]) # 中频
        phase_3x,amp3_x,offset3_x = self.decode_phase(image=self.images[20:24]) # 低频
        
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

        # 2. 外差法获取逐级展开的相位差
        # 计算垂直方向相位差
        phase_12y = self.phase_diff(phase_1y,phase_2y)  # 频率1和2的差异
        phase_23y = self.phase_diff(phase_2y,phase_3y)  # 频率2和3的差异
        phase_123y = self.phase_diff(phase_12y,phase_23y) # 差异的差异(等效最低频)

        # 计算水平方向相位差
        phase_12x = self.phase_diff(phase_1x,phase_2x)  # 频率1和2的差异
        phase_23x = self.phase_diff(phase_2x,phase_3x)  # 频率2和3的差异
        phase_123x = self.phase_diff(phase_12x,phase_23x) # 差异的差异(等效最低频)
        
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

        # 3. 平滑最低等效频率相位以提高鲁棒性
        phase_123y = cv.GaussianBlur(phase_123y,(3,3),0)
        phase_123x = cv.GaussianBlur(phase_123x,(3,3),0)
        
        # 保存第二次多频外差结果
        self._save_phase_image(phase_123y, 'phase_123y.png', '3_second_heterodyne')
        self._save_phase_image(phase_123x, 'phase_123x.png', '3_second_heterodyne')

        # 4. 相位展开流程 - 自底向上展开
        # 使用最低等效频率相位(phase_123y/x)展开中等频率相位差(phase_12y/x和phase_23y/x)
        unwarp_phase_12_y = self.unwarpphase(phase_123y,phase_12y,1,self.f12)
        unwarp_phase_23_y = self.unwarpphase(phase_123y,phase_23y,1,self.f23)

        unwarp_phase_12_x = self.unwarpphase(phase_123x,phase_12x,1,self.f12)
        unwarp_phase_23_x = self.unwarpphase(phase_123x,phase_23x,1,self.f23)
        
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
        
        # 5. 使用展开后的中等频率相位差(unwarp_phase_12_y/x和unwarp_phase_23_y/x)
        # 展开中频相位(phase_2y/x)
        unwarp_phase2_y_12 = self.unwarpphase(unwarp_phase_12_y,phase_2y,self.f12,self.f[1])
        unwarp_phase2_y_23 = self.unwarpphase(unwarp_phase_23_y,phase_2y,self.f23,self.f[1])

        unwarp_phase2_x_12 = self.unwarpphase(unwarp_phase_12_x,phase_2x,self.f12,self.f[1])
        unwarp_phase2_x_23 = self.unwarpphase(unwarp_phase_23_x,phase_2x,self.f23,self.f[1])
        
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

        # 6. 取两个展开路径的平均值以提高鲁棒性
        unwarp_phase_y = (unwarp_phase2_y_12+unwarp_phase2_y_23)/2
        unwarp_phase_x = (unwarp_phase2_x_12+unwarp_phase2_x_23)/2

        # 7. 归一化相位结果
        unwarp_phase_y/=self.f[1]  # 以中频为基准归一化
        unwarp_phase_x/=self.f[1]  # 以中频为基准归一化
        
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

        # 8. 计算相位质量，使用调制度/偏移比值的最小值
        ratio_x = np.min([amp1_x/offset1_x,amp2_x/offset2_x,amp3_x/offset3_x],axis=0)
        ratio_y = np.min([amp1_y/offset1_y,amp2_y/offset2_y,amp3_y/offset3_y],axis=0)

        ratio = np.min([ratio_x,ratio_y],axis=0)  # 取水平和垂直方向的最小值作为最终质量图
        
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
