import numpy as np
import cv2
from typing import List, Optional
from enum import Enum
from skimage import morphology
import skimage.filters as filters


class PhaseShiftingAlgorithm(Enum):
    """相移算法类型枚举"""
    three_step = 0      # 三步相移
    four_step = 1       # 四步相移
    n_step = 2          # N步相移


def compute_amplitude_from_images(images: List[np.ndarray], algorithm: PhaseShiftingAlgorithm = PhaseShiftingAlgorithm.four_step) -> np.ndarray:
    """
    从相移图像计算振幅（调制度），用于生成投影区域掩膜
    
    参数:
        images: 相移图像列表
        algorithm: 相移算法类型
    
    返回:
        amplitude: 振幅图，值越大表示投影强度越高
    """
    if len(images) < 3:
        raise ValueError(f"相移图像数量不足。至少需要3张图像，但只提供了{len(images)}张。")
    
    float_images = [img.astype(np.float32) for img in images]
    n = len(float_images)
    
    # 计算振幅（调制度）
    if algorithm == PhaseShiftingAlgorithm.three_step:
        I1, I2, I3 = float_images[0], float_images[1], float_images[2]
        # 三步相移的振幅计算
        sin_sum = (np.sqrt(3)/2) * (I2 - I3)
        cos_sum = I1 - 0.5 * (I2 + I3)
    elif algorithm == PhaseShiftingAlgorithm.four_step:
        I1, I2, I3, I4 = float_images[0], float_images[1], float_images[2], float_images[3]
        # 四步相移的振幅计算
        sin_sum = I2 - I4
        cos_sum = I1 - I3
    elif algorithm == PhaseShiftingAlgorithm.n_step:
        delta = 2 * np.pi / n
        sin_sum = sum(float_images[i] * np.sin(i * delta) for i in range(n))
        cos_sum = sum(float_images[i] * np.cos(i * delta) for i in range(n))
    else:
        raise ValueError(f"不支持的相移算法: {algorithm}")
    
    # 计算振幅（调制度）
    amplitude = np.sqrt(sin_sum**2 + cos_sum**2) * (2 / n)
    
    return amplitude


def _otsu_thresholding(A_norm, M_norm, I_norm):
    """改进的Otsu方法"""
    _, mask_A = cv2.threshold(A_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_M = cv2.threshold(M_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 降低强度阈值，使更多区域通过
    th_I = np.percentile(I_norm, 30)  # 从50降低到30
    mask_I = (I_norm > th_I).astype(np.uint8) * 255
    
    mask = np.logical_or(mask_A > 0, mask_M > 0)
    mask = np.logical_and(mask, mask_I > 0)
    
    return mask


def _progressive_morphological_processing(mask, min_area):
    """渐进式形态学处理"""
    mask = mask.astype(np.uint8)
    
    # 更温和的形态学处理
    # 小尺度闭运算
    mask = morphology.binary_closing(mask, morphology.disk(2))  # 减小核大小
    
    # 移除超小对象（更宽松）
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_area//8)  # 更小的阈值
    
    # 中等尺度闭运算
    mask = morphology.binary_closing(mask, morphology.disk(3))  # 减小核大小
    
    # 填充小孔洞（更宽松）
    mask = morphology.remove_small_holes(mask, area_threshold=min_area//4)  # 更小的阈值
    
    # 最终移除小对象（更宽松）
    mask = morphology.remove_small_objects(mask, min_size=min_area//2)  # 更小的阈值
    
    return mask.astype(np.uint8)


def _smart_border_trimming(mask, border_trim_px):
    """智能边界处理"""
    if not border_trim_px or border_trim_px <= 0:
        return mask
    
    h, w = mask.shape
    bt = min(border_trim_px, min(h // 15, w // 15))  # 更保守的边界收缩
    
    if bt > 0:
        # 只在边界区域确实存在噪声时才进行收缩
        border_noise_ratio = 0.1  # 边界噪声比例阈值
        
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


def _connectivity_optimization(mask):
    """连通性优化"""
    # 保留最大的几个连通分量
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    
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


def generate_projection_mask(images: List[np.ndarray], 
                           algorithm: PhaseShiftingAlgorithm = PhaseShiftingAlgorithm.four_step,
                           method: str = 'otsu', 
                           thresh_rel: Optional[float] = None, 
                           min_area: int = 500,
                           confidence: float = 0.5,
                           border_trim_px: int = 10) -> np.ndarray:
    """
    基于相移图像生成投影区域掩膜，用于区分投影区域和背景
    使用Otsu自适应阈值方法
    
    参数:
        images: 相移图像列表
        algorithm: 相移算法类型
        method: 阈值化方法（固定为'otsu'）
        thresh_rel: 未使用（保留参数兼容性）
        min_area: 最小连通区域面积
        confidence: 掩膜置信度阈值（对Otsu方法影响较小）
        border_trim_px: 边界收缩像素数
    
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

    # 基本特征计算
    A = (I_max - I_min) / 2.0  # 振幅
    M = (I_max - I_min) / (I_max + I_min + 1e-9)  # 调制度
    
    # 归一化特征
    A_norm = cv2.normalize(A, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    M_norm = cv2.normalize(M, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    I_norm = cv2.normalize(I_mean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 使用Otsu阈值化方法
    mask = _otsu_thresholding(A_norm, M_norm, I_norm)
    
    print(f"掩膜生成参数: 方法=Otsu自适应阈值")
    
    # 渐进式形态学处理
    mask = _progressive_morphological_processing(mask, min_area)
    
    # 智能边界处理
    mask = _smart_border_trimming(mask, border_trim_px)
    
    # 连通性分析和优化
    mask = _connectivity_optimization(mask)
    
    return mask.astype(bool)


def save_mask_visualization(images: List[np.ndarray], mask: np.ndarray, output_dir: str):
    """
    保存掩膜相关的可视化图像（振幅、调制度、平均强度、最终掩膜）
    
    参数:
        images: 相移图像列表
        mask: 生成的掩膜
        output_dir: 输出目录
    """
    import os
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算特征图
    imgs = np.stack([img.astype(np.float32) for img in images], axis=2)
    I_max = np.max(imgs, axis=2)
    I_min = np.min(imgs, axis=2)
    I_mean = np.mean(imgs, axis=2)
    
    A = (I_max - I_min) / 2.0  # 振幅
    M = (I_max - I_min) / (I_max + I_min + 1e-9)  # 调制度
    
    # 归一化并保存
    A_img = cv2.normalize(A, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    M_img = cv2.normalize(M, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    I_img = cv2.normalize(I_mean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    cv2.imwrite(os.path.join(output_dir, 'Amplitude.png'), A_img)
    cv2.imwrite(os.path.join(output_dir, 'Modulation.png'), M_img)
    cv2.imwrite(os.path.join(output_dir, 'Mean Intensity.png'), I_img)
    cv2.imwrite(os.path.join(output_dir, 'Final Mask.png'), (mask.astype(np.uint8) * 255))
    
    print(f"掩膜可视化图像已保存至: {output_dir}")


def load_mask_from_file(mask_path: str) -> Optional[np.ndarray]:
    """
    从文件加载掩膜
    
    参数:
        mask_path: 掩膜文件路径
    
    返回:
        mask: 布尔类型的掩膜数组，如果加载失败则返回None
    """
    try:
        mask_img = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            print(f"无法加载掩膜文件: {mask_path}")
            return None
        mask = (mask_img > 0)
        print(f"成功从文件加载掩膜: {mask_path}")
        return mask
    except Exception as e:
        print(f"加载掩膜时出错: {e}")
        return None


def get_or_create_mask(images: List[np.ndarray],
                       algorithm: PhaseShiftingAlgorithm = PhaseShiftingAlgorithm.four_step,
                       use_shared_mask: bool = True,
                       shared_mask_path: Optional[str] = None,
                       mask_method: str = 'otsu',
                       thresh_rel: Optional[float] = None,
                       min_area: int = 500,
                       confidence: float = 0.5,
                       border_trim_px: int = 10,
                       save_visualization: bool = True,
                       visualization_dir: Optional[str] = None) -> np.ndarray:
    """
    获取或创建掩膜的便捷函数（使用Otsu自适应阈值方法）
    
    参数:
        images: 相移图像列表
        algorithm: 相移算法类型
        use_shared_mask: 是否尝试使用共享掩膜
        shared_mask_path: 共享掩膜的路径
        mask_method: 掩膜生成方法（固定为'otsu'）
        thresh_rel: 未使用（保留参数兼容性）
        min_area: 最小连通区域面积
        confidence: 置信度阈值（对Otsu方法影响较小）
        border_trim_px: 边界收缩像素数
        save_visualization: 是否保存可视化图像
        visualization_dir: 可视化图像保存目录
    
    返回:
        mask: 布尔类型的掩膜数组
    """
    mask = None
    
    # 尝试加载共享掩膜
    if use_shared_mask and shared_mask_path is not None:
        import os
        if os.path.isfile(shared_mask_path):
            mask = load_mask_from_file(shared_mask_path)
            if mask is not None:
                print(f"使用共享掩膜: {shared_mask_path}")
                return mask
    
    # 如果无法加载共享掩膜，则生成新掩膜
    print(f"生成新掩膜，方法: {mask_method}")
    mask = generate_projection_mask(
        images=images,
        algorithm=algorithm,
        method=mask_method,
        thresh_rel=thresh_rel,
        min_area=min_area,
        confidence=confidence,
        border_trim_px=border_trim_px
    )
    
    print(f"掩膜生成完成，投影区域像素数: {np.sum(mask)}")
    
    # 保存共享掩膜
    if use_shared_mask and shared_mask_path is not None:
        import os
        os.makedirs(os.path.dirname(shared_mask_path), exist_ok=True)
        cv2.imwrite(shared_mask_path, mask.astype(np.uint8) * 255)
        print(f"共享掩膜已保存至: {shared_mask_path}")
    
    # 保存可视化图像
    if save_visualization and visualization_dir is not None:
        save_mask_visualization(images, mask, visualization_dir)
    
    return mask


if __name__ == '__main__':
    print("掩膜生成模块")
    print("使用方法:")
    print("  from Mask_generation import get_or_create_mask, PhaseShiftingAlgorithm")
    print("  mask = get_or_create_mask(images, algorithm=PhaseShiftingAlgorithm.four_step)")
    print("\n掩膜生成方法:")
    print("  - 'otsu': Otsu自适应阈值方法（基于Otsu算法的自动阈值化）")
    print("\n参数说明:")
    print("  - confidence: 掩膜置信度 (0.1-0.9，对Otsu方法影响较小)")
    print("  - min_area: 最小连通区域面积 (默认500)")
    print("  - border_trim_px: 边界收缩像素数 (默认10)")

