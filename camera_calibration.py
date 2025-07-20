#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机标定独立程序

该程序允许用户使用一组棋盘格或圆形标定板图像对相机进行标定，获取相机的内参矩阵和畸变系数。
可以直接运行此程序，按照提示输入相关参数。

使用方法:
1. 准备多张不同角度拍摄的标定板图像，保存在一个文件夹中
2. 运行程序，按照提示输入图像文件夹路径、标定板类型、标定点数量和实际尺寸
3. 程序将自动进行标定，并保存结果

支持的标定板类型：
- 棋盘格标定板 (chessboard)：使用findChessboardCorners检测
- 圆形标定板 (circles)：使用findCirclesGrid检测，默认为白底黑圆
- 空心圆环标定板 (ring_circles)：使用findCirclesGrid检测，白底空心圆环

作者: [Your Name]
日期: [Date]
"""

import os
import sys
import numpy as np
import cv2
import glob
import json
from matplotlib import pyplot as plt
import argparse
from datetime import datetime

# 配置matplotlib支持中文显示
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans', 'Arial']  # 用来正常显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    print("已配置matplotlib支持中文显示")
except:
    print("配置matplotlib中文支持失败，图像中的中文可能无法正常显示")

# 定义OpenCV显示中文文本的函数
def put_chinese_text(img, text, position, font_size=30, color=(0, 0, 255)):
    """在OpenCV图像上显示中文文本
    
    参数:
        img: 输入图像
        text: 要显示的文本
        position: 位置元组 (x, y)
        font_size: 字体大小
        color: 字体颜色 (B, G, R)
    
    返回:
        添加文本后的图像
    """
    # 深拷贝图像，以免修改原图
    img_copy = img.copy()
    
    # 在matplotlib中绘制文本再转换为OpenCV图像格式
    # 创建空白图像，仅用于生成文本
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    
    fig = Figure(figsize=(1, 1), dpi=font_size*2)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, text, horizontalalignment='center', verticalalignment='center',
            fontsize=font_size, color=(color[2]/255.0, color[1]/255.0, color[0]/255.0))
    ax.axis('off')
    fig.tight_layout(pad=0)
    canvas.draw()
    
    # 将matplotlib图像转换为OpenCV图像
    buf = canvas.buffer_rgba()
    text_img = np.asarray(buf)
    text_img = cv2.cvtColor(text_img, cv2.COLOR_RGBA2BGRA)
    
    # 创建透明蒙版
    alpha = text_img[:,:,3]/255.0
    alpha = np.repeat(alpha[:,:,np.newaxis], 3, axis=2)
    
    # 提取文本颜色
    text_img = text_img[:,:,:3]
    
    # 计算放置文本的区域
    x, y = position
    h, w = text_img.shape[:2]
    
    # 确保坐标不越界
    if y + h > img_copy.shape[0]:
        h = img_copy.shape[0] - y
    if x + w > img_copy.shape[1]:
        w = img_copy.shape[1] - x
    
    if h <= 0 or w <= 0:
        return img_copy
        
    # 融合文本和图像
    roi = img_copy[y:y+h, x:x+w]
    roi_result = (1 - alpha[:h,:w]) * roi + alpha[:h,:w] * text_img[:h,:w]
    img_copy[y:y+h, x:x+w] = roi_result.astype(np.uint8)
    
    return img_copy

def select_calibration_board_type():
    """
    交互式选择标定板类型
    
    返回:
        board_type: 选择的标定板类型字符串 ('chessboard', 'circles' 或 'ring_circles')
    """
    print("\n请选择标定板类型：")
    print("1. 棋盘格标定板 (适合精确角点检测)")
    print("2. 圆形标定板-白底黑圆 (适合光照变化较大的场景)")
    print("3. 空心圆环标定板-白底空心圆 (适合高反光或特殊光照条件)")
    
    while True:
        choice = input("请输入选择 [1-3，默认：1]: ").strip()
        if choice == '' or choice == '1':
            return 'chessboard'
        elif choice == '2':
            return 'circles'
        elif choice == '3':
            return 'ring_circles'
        else:
            print("无效选择，请重新输入！")

def configure_detection_parameters(board_type):
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

def show_calibration_tips(board_type):
    """
    为用户提供针对所选标定板类型的最佳实践建议
    
    参数:
        board_type: 标定板类型
    """
    print("\n===== 标定板使用建议 =====")
    
    if board_type == 'chessboard':
        print("【棋盘格标定板最佳实践】")
        print("- 确保棋盘格打印精确，无变形")
        print("- 标定板应固定在坚硬平面上，避免弯曲")
        print("- 拍摄时覆盖相机视场的不同区域（特别是边角）")
        print("- 以不同角度（约15-45度）拍摄至少10张图像")
        print("- 避免强反光和不均匀光照")
        print("- 确保整个标定板清晰可见")
    elif board_type == 'circles':
        print("【圆形标定板(白底黑圆)最佳实践】")
        print("- 适用于光照条件不理想的场景")
        print("- 圆点大小应适中，过大或过小都会影响检测")
        print("- 拍摄时避免极端视角，防止圆变成椭圆影响检测")
        print("- 确保光照均匀，减少阴影")
        print("- 对于光滑表面，考虑使用哑光材料打印以减少反光")
    elif board_type == 'ring_circles':
        print("【空心圆环标定板最佳实践】")
        print("- 最适合强光或反光条件下使用")
        print("- 圆环厚度应适中，太细可能检测不到")
        print("- 可以在白纸上用黑色记号笔手绘，确保圆环闭合")
        print("- 特别适合玻璃表面或有反光的场景")
        print("- 保持圆环形状规则，避免变形")
    
    print("\n【通用技巧】")
    print("- 标定板与相机距离应适中，不要太近或太远")
    print("- 确保图像中的标定点至少占图像的1/3区域")
    print("- 保持标定板完全位于相机视场内")
    print("- 图像应该清晰无模糊")
    print("============================")

def preprocess_image(img, board_type):
    """
    根据标定板类型优化图像预处理
    
    参数:
        img: 输入图像
        board_type: 标定板类型
        
    返回:
        处理后的灰度图像
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
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

def assess_calibration_quality(reprojection_error, board_type):
    """
    评估标定质量并提供改进建议
    
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

def calibrate_camera(images_folder, board_type="chessboard", board_size=(9, 6), square_size=20.0,
                   visualize=True, delay=500):
    """
    使用棋盘格或圆形标定板图像进行相机标定
    
    参数:
        images_folder: 包含标定板图像的文件夹路径
        board_type: 标定板类型 ('chessboard', 'circles' 或 'ring_circles')
        board_size: 标定板内角点数量 (宽, 高)
        square_size: 标定板方格尺寸或圆心间距(mm)
        visualize: 是否可视化角点检测结果
        delay: 可视化图像显示的延迟时间(ms)
    
    返回:
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
        rvecs: 旋转向量
        tvecs: 平移向量
        reprojection_error: 重投影误差
        image_size: 图像尺寸
    """
    # 根据标定板类型获取检测参数
    detection_params = configure_detection_parameters(board_type)
    
    # 准备对象点
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp = objp * square_size  # 应用实际尺寸
    
    # 存储所有图像的对象点和图像点
    objpoints = []  # 3D空间中的点
    imgpoints = []  # 图像平面上的点
    
    # 获取图像文件列表
    images = glob.glob(os.path.join(images_folder, '*.jpg'))
    images.extend(glob.glob(os.path.join(images_folder, '*.png')))
    
    if len(images) == 0:
        raise ValueError(f"在文件夹 '{images_folder}' 中未找到jpg或png图像文件")
    
    # 图像尺寸
    img_shape = None
    
    print(f"找到 {len(images)} 张校准图像")
    
    # 创建结果文件夹
    results_folder = os.path.join(images_folder, "calibration_results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # 如果是圆形标定板，获取圆形检测器
    blob_detector = None
    if board_type in ['circles', 'ring_circles']:
        blob_detector = detection_params['detector']
    
    # 处理每张图像
    successful_images = []
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"无法读取图像: {fname}")
            continue
            
        # 应用优化的图像预处理
        gray = preprocess_image(img, board_type)
        
        if img_shape is None:
            img_shape = gray.shape[::-1]  # (width, height)
        
        # 寻找标定板角点或圆心
        found = False
        if board_type == 'chessboard':
            # 寻找棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, board_size, None)
            
            # 如果找到角点，优化角点位置
            if ret:
                found = True
                # 使用亚像素精度优化角点位置
                criteria = detection_params['criteria']
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                
                # 绘制并显示角点
                if visualize:
                    cv2.drawChessboardCorners(img, board_size, corners2, ret)
        elif board_type in ['circles', 'ring_circles']:
            # 寻找圆形网格模式
            flags = detection_params['flags']
            ret, centers = cv2.findCirclesGrid(
                image=gray,
                patternSize=board_size,
                flags=flags,
                blobDetector=blob_detector
            )
            
            if ret:
                found = True
                imgpoints.append(centers)
                
                # 绘制并显示圆心
                if visualize:
                    cv2.drawChessboardCorners(img, board_size, centers, ret)
        
        if found:
            objpoints.append(objp)
            successful_images.append(os.path.basename(fname))
            
            # 可视化结果
            if visualize:
                img_display = img.copy()
                
                # 保存标记了角点或圆心的图像
                marked_img_path = os.path.join(results_folder, f"points_{os.path.basename(fname)}")
                cv2.imwrite(marked_img_path, img_display)
                
                # 显示图像
                cv2.imshow('Calibration Points', cv2.resize(img_display, (0, 0), fx=0.5, fy=0.5))
                cv2.waitKey(delay)
                
                # 显示进度
                sys.stdout.write(f"\r处理图像 {i+1}/{len(images)}: {os.path.basename(fname)}")
                sys.stdout.flush()
        else:
            print(f"\n未能在图像 {fname} 中找到所有标定点")
    
    print(f"\n成功处理了 {len(objpoints)}/{len(images)} 张图像")
    
    if visualize:
        cv2.destroyAllWindows()
    
    if len(objpoints) == 0:
        raise ValueError(f"未找到任何有效的{board_type}标定板图像，无法进行标定")
    
    # 执行相机标定
    flags = 0
    # 可以添加额外的标志，如 cv2.CALIB_FIX_K3
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None, flags=flags
    )
    
    # 计算重投影误差
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    reprojection_error = total_error / len(objpoints)
    print(f"平均重投影误差: {reprojection_error} 像素")
    
    # 评估标定质量
    assess_calibration_quality(reprojection_error, board_type)
    
    return camera_matrix, dist_coeffs, rvecs, tvecs, reprojection_error, img_shape, successful_images

def save_calibration_results(output_folder, output_basename, camera_matrix, dist_coeffs, 
                           image_size, reprojection_error, board_info, successful_images):
    """
    保存标定结果到文件
    
    参数:
        output_folder: 输出文件夹
        output_basename: 输出文件基本名称
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
        image_size: 图像尺寸
        reprojection_error: 重投影误差
        board_info: 标定板信息 (类型、尺寸和实际大小)
        successful_images: 成功处理的图像列表
    """
    # 创建结果目录
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 生成当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存为NumPy格式 (.npy)
    npy_file = os.path.join(output_folder, f"{output_basename}_{timestamp}.npy")
    calibration_data_npy = {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs
    }
    np.save(npy_file, calibration_data_npy)
    
    # 保存为JSON格式 (.json)
    json_file = os.path.join(output_folder, f"{output_basename}_{timestamp}.json")
    calibration_data_json = {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist(),
        'image_size': image_size,
        'reprojection_error': float(reprojection_error),
        'board_type': board_info['type'],
        'board_size': board_info['size'],
        'square_size': board_info['square_size'],
        'calibration_time': timestamp,
        'successful_images': successful_images
    }
    
    with open(json_file, 'w') as f:
        json.dump(calibration_data_json, f, indent=4)
    
    # 同时保存一个固定名称的文件，方便其他程序直接引用
    default_json_file = os.path.join(output_folder, "camera_calibration_latest.json")
    with open(default_json_file, 'w') as f:
        json.dump(calibration_data_json, f, indent=4)
    
    print(f"标定结果已保存至:")
    print(f"  - NumPy格式: {npy_file}")
    print(f"  - JSON格式: {json_file}")
    print(f"  - 最新标定结果: {default_json_file} (供其他程序引用)")
    
    return json_file, default_json_file

def test_undistortion(image_path, camera_matrix, dist_coeffs, output_folder=None, alpha=1.0):
    """
    测试畸变校正效果
    
    参数:
        image_path: 输入图像路径
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
        output_folder: 输出文件夹路径(如果需要保存结果)
        alpha: 缩放参数(0-1), 0表示最小裁剪, 1表示无裁剪
    
    返回:
        dst: 校正后的图像
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    h, w = img.shape[:2]
    
    # 获取最佳相机矩阵
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), alpha, (w, h))
    
    # 校正图像
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
    
    # 裁剪结果
    if roi[2] > 0 and roi[3] > 0:  # 确保ROI有效
        x, y, w, h = roi
        dst_cropped = dst[y:y+h, x:x+w]
    else:
        dst_cropped = dst
    
    # 创建比较图像
    comparison = np.hstack((img, dst))
    
    # 显示原始图像和校正后的图像
    plt.figure(figsize=(15, 10))
    
    # 原始图像
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')
    
    # 校正后未裁剪的图像
    plt.subplot(222)
    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.title('校正后图像 (未裁剪)')
    plt.axis('off')
    
    # 校正后裁剪的图像
    plt.subplot(223)
    plt.imshow(cv2.cvtColor(dst_cropped, cv2.COLOR_BGR2RGB))
    plt.title('校正后图像 (裁剪)')
    plt.axis('off')
    
    # 原始与校正后的对比
    plt.subplot(224)
    plt.imshow(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
    plt.title('原始图像 vs 校正后图像')
    plt.axis('off')
    
    plt.tight_layout()
    
    # 保存结果
    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 保存校正后的图像
        undistorted_path = os.path.join(output_folder, f"{base_name}_undistorted.jpg")
        cv2.imwrite(undistorted_path, dst)
        
        # 保存裁剪后的图像
        cropped_path = os.path.join(output_folder, f"{base_name}_undistorted_cropped.jpg")
        cv2.imwrite(cropped_path, dst_cropped)
        
        # 保存对比图
        comparison_path = os.path.join(output_folder, f"{base_name}_comparison.jpg")
        cv2.imwrite(comparison_path, comparison)
        
        # 保存matplotlib图
        plt_path = os.path.join(output_folder, f"{base_name}_undistortion_results.png")
        plt.savefig(plt_path, dpi=300, bbox_inches='tight')
        
        print(f"校正结果已保存至: {output_folder}")
    
    plt.show()
    
    return dst_cropped

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='相机标定工具')
    parser.add_argument('--images', type=str, help='包含标定图像的文件夹路径')
    parser.add_argument('--board-type', type=str, default=None, 
                      choices=['chessboard', 'circles', 'ring_circles'], help='标定板类型')
    parser.add_argument('--width', type=int, default=9, help='标定板宽度点数量')
    parser.add_argument('--height', type=int, default=6, help='标定板高度点数量')
    parser.add_argument('--size', type=float, default=20.0, help='标定板方格尺寸或圆心间距(mm)')
    parser.add_argument('--no-visualize', action='store_true', help='不显示角点检测过程')
    parser.add_argument('--delay', type=int, default=500, help='角点检测可视化时的延迟(ms)')
    parser.add_argument('--test-image', type=str, help='用于测试畸变校正的图像路径')
    parser.add_argument('--output', type=str, help='自定义输出文件夹路径')
    
    args = parser.parse_args()
    
    # 如果没有通过命令行提供图像文件夹，则请用户输入
    images_folder = args.images
    if images_folder is None:
        images_folder = input("请输入包含标定板图像的文件夹路径: ").strip()
        if not images_folder:
            print("未提供有效的图像文件夹路径，退出程序。")
            return
    
    # 确保文件夹存在
    if not os.path.exists(images_folder):
        print(f"错误: 文件夹 {images_folder} 不存在")
        return
    
    # 获取标定板类型 - 无论命令行是否提供，都让用户选择
    board_type = select_calibration_board_type()
    
    # 显示标定板使用建议
    show_calibration_tips(board_type)
    
    # 获取标定板参数
    board_width = args.width
    board_height = args.height
    square_size = args.size
    
    # 如果未通过命令行提供，则询问用户
    if args.width == 9 and args.height == 6:  # 检查是否使用了默认值
        try:
            user_input = input(f"请输入标定板点数量 (宽 高) [默认: {board_width} {board_height}]: ").strip()
            if user_input:
                board_width, board_height = map(int, user_input.split())
        except:
            print(f"使用默认标定板尺寸: {board_width}x{board_height}")
    
    if args.size == 20.0:  # 检查是否使用了默认值
        try:
            user_input = input(f"请输入标定板方格尺寸或圆心间距(mm) [默认: {square_size}]: ").strip()
            if user_input:
                square_size = float(user_input)
        except:
            print(f"使用默认方格尺寸: {square_size}mm")
    
    print(f"\n使用以下参数进行标定:")
    print(f"  - 图像文件夹: {images_folder}")
    print(f"  - 标定板类型: {board_type}")
    print(f"  - 标定板点数量: {board_width}x{board_height}")
    print(f"  - 方格尺寸或圆心间距: {square_size}mm")
    print(f"  - 可视化: {'否' if args.no_visualize else '是'}")
    print()
    
    try:
        # 执行标定
        camera_matrix, dist_coeffs, rvecs, tvecs, reprojection_error, image_size, successful_images = calibrate_camera(
            images_folder,
            board_type=board_type,
            board_size=(board_width, board_height),
            square_size=square_size,
            visualize=not args.no_visualize,
            delay=args.delay
        )
        
        # 打印标定结果
        print("\n相机标定结果:")
        print("相机内参矩阵:")
        print(camera_matrix)
        print("\n畸变系数 (k1, k2, p1, p2, k3):")
        print(dist_coeffs)
        print(f"\n图像尺寸: {image_size[0]}x{image_size[1]}")
        
        # 保存标定结果
        board_info = {
            'type': board_type,
            'size': (board_width, board_height),
            'square_size': square_size
        }
        
        # 确定输出文件夹
        if args.output:
            results_folder = args.output
        else:
            results_folder = os.path.join(images_folder, "calibration_results")
        
        # 保存标定结果
        calibration_file, default_file = save_calibration_results(
            results_folder, 
            "camera_calibration", 
            camera_matrix, 
            dist_coeffs, 
            image_size, 
            reprojection_error, 
            board_info,
            successful_images
        )
        
        print(f"\n默认标定文件路径: {default_file}")
        print("可以使用此文件路径作为投影仪标定程序的输入")
        
        # 测试畸变校正
        test_image = args.test_image
        if test_image is None:
            # 如果没有提供测试图像，询问用户是否要使用一张标定图像作为测试
            response = input("\n是否要测试畸变校正? (y/n) [默认:y]: ").strip().lower()
            if not response or response == 'y':
                images = glob.glob(os.path.join(images_folder, '*.jpg'))
                images.extend(glob.glob(os.path.join(images_folder, '*.png')))
                if images:
                    test_image = images[0]  # 使用第一张图像作为测试
        
        if test_image and os.path.exists(test_image):
            print(f"\n使用图像 {test_image} 测试畸变校正...")
            test_undistortion(
                test_image, 
                camera_matrix, 
                dist_coeffs, 
                output_folder=results_folder
            )
        
        print("\n相机标定完成！")
        
    except Exception as e:
        print(f"\n标定过程中出错: {str(e)}")

if __name__ == "__main__":
    main() 