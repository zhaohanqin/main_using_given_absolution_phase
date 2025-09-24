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
    确保投影区域的完整性
    投影仪投影时是一个完整的面，不应该有内部空洞
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
    
    # 2. 计算主投影区域的凸包
    contours, _ = cv.findContours(main_projection, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return mask
    
    # 找到最大轮廓
    main_contour = max(contours, key=cv.contourArea)
    
    # 计算凸包
    hull = cv.convexHull(main_contour)
    
    # 3. 创建凸包掩膜
    hull_mask = np.zeros_like(mask)
    cv.fillPoly(hull_mask, [hull], 255)
    
    # 4. 使用凸包和原始掩膜的交集，但填充内部空洞
    # 先用原始掩膜，然后在凸包范围内填充所有空洞
    combined_mask = mask.copy()
    
    # 在凸包区域内，填充所有空洞
    hull_bool = hull_mask > 0
    
    # 使用形态学操作填充凸包内的空洞
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    filled_in_hull = cv.morphologyEx(main_projection, cv.MORPH_CLOSE, kernel)
    
    # 进一步填充空洞
    filled_in_hull = morphology.remove_small_holes(filled_in_hull.astype(bool), area_threshold=1000)
    
    # 5. 创建最终掩膜：在凸包范围内使用填充后的掩膜
    final_mask = mask.copy()
    
    # 在凸包区域内，如果原掩膜有投影区域，则填充该区域的所有空洞
    if np.any(main_projection):
        # 创建一个膨胀的版本来填充空洞
        dilated = cv.dilate(main_projection, kernel, iterations=3)
        eroded = cv.erode(dilated, kernel, iterations=2)
        
        # 只在凸包范围内应用
        final_mask = np.where(hull_bool, np.maximum(final_mask, eroded), final_mask)
    
    # 6. 最后一次空洞填充
    final_mask = morphology.remove_small_holes(final_mask.astype(bool), area_threshold=2000)
    
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
    def __init__(self, f, step, images, ph0, mask=None, use_mask=True, mask_method='otsu', mask_confidence=0.5, output_dir=None):
        """
        初始化多频相位解包裹对象
        
        参数:
            f: 列表，包含多个频率值，按从高到低排序，例如[64,8,1]
            step: 整数，相移步数(通常为3或4)
            images: ndarray，所有条纹图像组成的数组
            ph0: 浮点数，相移初始相位偏移量
            mask: ndarray，投影区域掩膜，True表示有效区域
            use_mask: bool，是否使用掩膜约束
            mask_method: str，掩膜生成方法 ('otsu', 'adaptive', 'relative')
            mask_confidence: float，掩膜置信度阈值 (0.1-0.9)
            output_dir: str，输出目录，用于保存调试图像
        """
        self.f = f                          # 频率列表
        self.images = images               # 相移图像
        self.step = step                   # 相移步数
        self.ph0 = ph0                     # 相移初始相位
        self.f12 = f[0] - f[1]            # 第1和第2个频率的差值(高频-中频)
        self.f23 = f[1] - f[2]            # 第2和第3个频率的差值(中频-低频)
        self.use_mask = use_mask           # 是否使用掩膜约束
        self.mask_method = mask_method     # 掩膜生成方法
        self.mask_confidence = mask_confidence  # 掩膜置信度
        self._output_dir = output_dir      # 输出目录
        
        # 生成或使用提供的掩膜
        if use_mask:
            if mask is not None:
                self.mask = mask.astype(bool)
                print(f"使用提供的掩膜，有效像素数: {np.sum(self.mask)}")
            else:
                print(f"生成投影区域掩膜，方法: {mask_method}")
                # 使用前4张图像生成掩膜（第一个频率的图像）
                first_freq_images = [images[i] for i in range(min(step, len(images)))]
                
                # 确定相移算法类型
                if step == 3:
                    algorithm = PhaseShiftingAlgorithm.three_step
                elif step == 4:
                    algorithm = PhaseShiftingAlgorithm.four_step
                else:
                    algorithm = PhaseShiftingAlgorithm.n_step
                
                # 对于条纹图像，优先使用基于调制度的方法
                if mask_method == 'otsu':
                    actual_method = 'modulation'  # 将otsu方法替换为更适合的调制度方法
                else:
                    actual_method = mask_method
                
                self.mask = generate_projection_mask_three_freq(
                    first_freq_images, 
                    algorithm=algorithm,
                    method=actual_method,
                    confidence=mask_confidence,
                    save_debug_images=True,
                    output_dir=getattr(self, '_output_dir', None)
                )
                print(f"掩膜生成完成，有效像素数: {np.sum(self.mask)}")
        else:
            # 不使用掩膜，创建全True的掩膜
            self.mask = np.ones((images[0].shape[0], images[0].shape[1]), dtype=bool)
            print("未使用掩膜约束")

    

    def decode_phase(self, image):
        """
        N步相移算法解码相位（在掩膜约束下进行）
        
        使用正弦和余弦项计算相移图像的包裹相位，并计算幅值和偏移量
        只在掩膜区域内进行计算，掩膜外区域设为0
        
        参数:
            image: ndarray，相移图像组，形状为[step, height, width]
            
        返回:
            result: ndarray，归一化的包裹相位图，掩膜外区域为0
            amp: ndarray，调制幅值，掩膜外区域为0
            offset: ndarray，亮度偏移，掩膜外区域为0
        """
        # 生成相移角度数组(0,2π/N,4π/N...)
        temp = 2*np.pi*np.arange(self.step,dtype=np.float32)/self.step
        temp.shape = -1, 1, 1  # 调整形状以便于广播运算
        
        # 如果使用掩膜，先对图像进行掩膜约束
        if self.use_mask:
            # 创建掩膜约束的图像
            if isinstance(image, list):
                # 如果image是列表，转换为numpy数组
                image = np.array(image, dtype=np.float32)
            else:
                # 如果image已经是numpy数组，复制并转换类型
                image = image.copy().astype(np.float32)
            
            for i in range(image.shape[0]):
                image[i][~self.mask] = 0  # 掩膜外区域设为0
        else:
            # 如果不使用掩膜，也需要确保image是numpy数组
            if isinstance(image, list):
                image = np.array(image, dtype=np.float32)
        
        # 计算正弦项(分子)和余弦项(分母)
        molecule = np.sum(image*np.sin(temp), axis=0)      # 正弦项
        denominator = np.sum(image*np.cos(temp), axis=0)   # 余弦项

        # 使用arctan2计算相位，保证相位值在[-π,π]范围内
        result = -np.arctan2(molecule, denominator)
        
        # 计算调制幅值和亮度偏移
        amp = 2/self.step*molecule        # 调制幅值
        offset = 2/self.step*denominator  # 亮度偏移

        # 归一化相位至[0,1]区间并减去初始相位
        result = (result+np.pi)/(2*np.pi)-self.ph0
        
        # 确保掩膜外区域为0
        if self.use_mask:
            result[~self.mask] = 0
            amp[~self.mask] = 0
            offset[~self.mask] = 0

        return result, amp, offset
    
    def phase_diff(self, image1, image2):
        """
        计算两个相位图之间的差值（在掩膜约束下进行）
        
        实现了外差法的核心操作，确保相位差在[0,1]范围内
        只在掩膜区域内进行计算，掩膜外区域设为0
        
        参数:
            image1: 高频相位图
            image2: 低频相位图
            
        返回:
            result: 两相位图的归一化差值，掩膜外区域为0
        """
        result = image1 - image2     # 计算相位差
        result[result < 0] += 1      # 处理负值，保证结果在[0,1]区间
        
        # 确保掩膜外区域为0
        if self.use_mask:
            result[~self.mask] = 0
            
        return result

    def unwarpphase(self, reference, phase, reference_f, phase_f):
        """
        基于低频参考相位展开高频相位（在掩膜约束下进行）
        
        参数:
            reference: 参考(低频)相位图
            phase: 需展开的(高频)包裹相位图
            reference_f: 参考相位的频率
            phase_f: 需展开相位的频率
            
        返回:
            unwarp_phase: 展开后的相位图，掩膜外区域为0
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
        
        # 只在掩膜区域内检测异常点
        if self.use_mask:
            order_flag = order_flag & self.mask
        
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
            
            # 只在掩膜区域内检测异常点
            if self.use_mask:
                order_flag2 = order_flag2 & self.mask
            
            if np.sum(order_flag2) > 0:
                unwarp_error2 = unwarp_phase[order_flag2]
                unwarp_error_direct2 = unwarp_phase_noise[order_flag2]
                
                # 根据噪声方向调整条纹序数
                unwarp_error2[unwarp_error_direct2 > 0] -= 1  # 正向噪声减少一个周期
                unwarp_error2[unwarp_error_direct2 < 0] += 1  # 负向噪声增加一个周期
                
                # 应用修复结果
                unwarp_phase[order_flag2] = unwarp_error2

        # 确保掩膜外区域为0
        if self.use_mask:
            unwarp_phase[~self.mask] = 0

        return unwarp_phase

    
    def get_phase(self):
        """
        多频相移解包裹主流程
        
        处理所有频率的相位图，分别对水平和垂直方向进行解包裹
        
        返回:
            unwarp_phase_y: 垂直方向展开的相位图
            unwarp_phase_x: 水平方向展开的相位图
            ratio: 相位质量图(基于调制度与偏移比)
        """
        # 1. 解码各个频率的相位
        # 解码垂直方向的三个频率的相位
        phase_1y,amp1_y,offset1_y = self.decode_phase(image=self.images[0:4])   # 高频
        phase_2y,amp2_y,offset2_y = self.decode_phase(image=self.images[4:8])   # 中频
        phase_3y,amp3_y,offset3_y = self.decode_phase(image=self.images[8:12])  # 低频

        # 解码水平方向的三个频率的相位
        phase_1x,amp1_x,offset1_x = self.decode_phase(image=self.images[12:16]) # 高频
        phase_2x,amp2_x,offset2_x = self.decode_phase(image=self.images[16:20]) # 中频
        phase_3x,amp3_x,offset3_x = self.decode_phase(image=self.images[20:24]) # 低频

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

        # 4. 相位展开流程 - 自底向上展开
        # 使用最低等效频率相位(phase_123y/x)展开中等频率相位差(phase_12y/x和phase_23y/x)
        unwarp_phase_12_y = self.unwarpphase(phase_123y,phase_12y,1,self.f12)
        unwarp_phase_23_y = self.unwarpphase(phase_123y,phase_23y,1,self.f23)

        unwarp_phase_12_x = self.unwarpphase(phase_123x,phase_12x,1,self.f12)
        unwarp_phase_23_x = self.unwarpphase(phase_123x,phase_23x,1,self.f23)
        
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
        # 避免除零，添加小的常数
        eps = 1e-10
        ratio_x = np.min([amp1_x/(offset1_x + eps), amp2_x/(offset2_x + eps), amp3_x/(offset3_x + eps)], axis=0)
        ratio_y = np.min([amp1_y/(offset1_y + eps), amp2_y/(offset2_y + eps), amp3_y/(offset3_y + eps)], axis=0)

        ratio = np.min([ratio_x, ratio_y], axis=0)  # 取水平和垂直方向的最小值作为最终质量图
        
        # 确保掩膜外区域的质量图为0
        if self.use_mask:
            ratio[~self.mask] = 0
            # 确保最终相位结果的掩膜外区域也为0
            unwarp_phase_y[~self.mask] = 0
            unwarp_phase_x[~self.mask] = 0
            phase_2y[~self.mask] = 0
            phase_2x[~self.mask] = 0
        
        # 注释掉所有中间过程的可视化代码
        """
        # 显示相位质量图
        plt.figure()
        plt.imshow(ratio, cmap='viridis')
        plt.title('相位质量图', fontproperties=font)
        plt.colorbar()
        """
        
        return unwarp_phase_y, unwarp_phase_x, ratio, phase_2y, phase_2x

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
