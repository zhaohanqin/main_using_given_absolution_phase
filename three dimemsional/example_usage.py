#!/usr/bin/env python3
"""
增强三维重建系统使用示例

这个脚本展示了如何使用增强三维重建系统进行三维重建。
包含了完整的使用流程和参数设置示例。
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib
from enhanced_3d_reconstruction import (
    Enhanced3DReconstruction,
    reconstruct_3d_scene_enhanced,
    load_camera_params,
    load_projector_params,
    load_extrinsics,
    load_unwrapped_phases,
    setup_chinese_font
)

# 设置中文字体
setup_chinese_font()


def create_example_data():
    """
    创建示例数据文件
    
    在实际使用中，这些数据应该来自于：
    1. 相机标定结果
    2. 投影仪标定结果
    3. 相机-投影仪外参标定结果
    4. 结构光相位解包裹结果
    """
    print("创建示例数据文件...")
    
    # 创建示例目录
    example_dir = "example_data"
    os.makedirs(example_dir, exist_ok=True)
    
    # 1. 创建相机内参文件
    camera_params = {
        "camera_matrix": [
            [1000.0, 0.0, 320.0],
            [0.0, 1000.0, 240.0],
            [0.0, 0.0, 1.0]
        ]
    }
    
    with open(os.path.join(example_dir, "camera_params.json"), 'w') as f:
        json.dump(camera_params, f, indent=2)
    
    # 2. 创建投影仪内参文件
    projector_params = {
        "projector_matrix": [
            [1100.0, 0.0, 640.0],
            [0.0, 1100.0, 400.0],
            [0.0, 0.0, 1.0]
        ],
        "projector_width": 1280,
        "projector_height": 800
    }
    
    with open(os.path.join(example_dir, "projector_params.json"), 'w') as f:
        json.dump(projector_params, f, indent=2)
    
    # 3. 创建外参文件
    extrinsics = {
        "R": [
            [0.95, -0.05, 0.1],
            [0.05, 0.98, 0.02],
            [-0.1, 0.02, 0.99]
        ],
        "T": [150.0, 30.0, 250.0]
    }
    
    with open(os.path.join(example_dir, "extrinsics.json"), 'w') as f:
        json.dump(extrinsics, f, indent=2)
    
    # 4. 创建示例相位图
    height, width = 600, 800

    # 模拟结构光相位图，包含三个球形物体
    x_coords = np.arange(width)
    y_coords = np.arange(height)

    # 基础相位图 - 模拟投影仪的条纹图案
    # X方向：水平条纹，相位范围 [0, 2π]
    phase_x = np.tile(x_coords * 2 * np.pi / width, (height, 1))
    # Y方向：垂直条纹，相位范围 [0, 2π]
    phase_y = np.tile(y_coords.reshape(-1, 1) * 2 * np.pi / height, (1, width))

    # 定义三个球形物体（使用高斯分布模拟，更容易重建）
    objects = [
        {"center": (200, 150), "sigma": 40, "amplitude": 0.4, "name": "大球"},  # 左上角球体
        {"center": (600, 300), "sigma": 30, "amplitude": 0.3, "name": "中球"},  # 右中球体
        {"center": (400, 450), "sigma": 50, "amplitude": 0.5, "name": "小球"}   # 下方球体
    ]

    print("生成三个球形物体的相位变化...")

    # 为每个物体添加相位变化（使用高斯分布）
    for i, obj in enumerate(objects):
        center_x, center_y = obj["center"]
        sigma = obj["sigma"]
        amplitude = obj["amplitude"]
        name = obj["name"]

        print(f"  {name}: 中心({center_x}, {center_y}), σ={sigma}, 幅度={amplitude}")

        # 创建高斯分布的相位偏移
        y_grid, x_grid = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        gaussian = np.exp(-((x_grid - center_x)**2 + (y_grid - center_y)**2) / (2 * sigma**2))

        # 添加相位变化
        phase_shift = gaussian * amplitude
        phase_x += phase_shift
        phase_y += phase_shift

    # 添加适量噪声（模拟实际测量噪声）
    noise_level = 0.05  # 减少噪声以获得更好的重建效果
    phase_x += np.random.normal(0, noise_level, (height, width))
    phase_y += np.random.normal(0, noise_level, (height, width))
    
    # 保存相位图
    np.save(os.path.join(example_dir, "phase_x.npy"), phase_x)
    np.save(os.path.join(example_dir, "phase_y.npy"), phase_y)

    # 可视化生成的相位图
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(phase_x, cmap='jet')
    plt.colorbar(label='相位 (弧度)')
    plt.title('X方向相位图（包含三个球体）')

    plt.subplot(132)
    plt.imshow(phase_y, cmap='jet')
    plt.colorbar(label='相位 (弧度)')
    plt.title('Y方向相位图（包含三个球体）')

    # 显示物体位置和相位偏移
    plt.subplot(133)
    total_shift = np.zeros((height, width))
    for i, obj in enumerate(objects):
        center_x, center_y = obj["center"]
        sigma = obj["sigma"]
        amplitude = obj["amplitude"]
        y_grid, x_grid = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        gaussian = np.exp(-((x_grid - center_x)**2 + (y_grid - center_y)**2) / (2 * sigma**2))
        total_shift += gaussian * amplitude

    plt.imshow(total_shift, cmap='viridis')
    plt.colorbar(label='相位偏移')
    plt.title('总相位偏移分布')

    # 标注物体中心
    for i, obj in enumerate(objects):
        plt.plot(obj["center"][0], obj["center"][1], 'w*', markersize=10)
        plt.text(obj["center"][0], obj["center"][1] - 30, obj["name"],
                ha='center', va='bottom', color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(example_dir, "generated_phase_maps.png"), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"示例数据已创建在目录: {example_dir}")
    print("已生成相位图可视化，显示了三个球形物体的位置")
    return example_dir


def run_basic_reconstruction():
    """运行基础三维重建示例"""
    print("\n" + "="*50)
    print("基础三维重建示例")
    print("="*50)
    
    # 创建示例数据
    example_dir = create_example_data()
    
    # 设置文件路径
    camera_file = os.path.join(example_dir, "camera_params.json")
    projector_file = os.path.join(example_dir, "projector_params.json")
    extrinsics_file = os.path.join(example_dir, "extrinsics.json")
    phase_x_file = os.path.join(example_dir, "phase_x.npy")
    phase_y_file = os.path.join(example_dir, "phase_y.npy")
    
    # 加载参数
    print("加载标定参数...")
    camera_matrix = load_camera_params(camera_file)
    projector_matrix, proj_width, proj_height = load_projector_params(projector_file)
    R, T = load_extrinsics(extrinsics_file)
    phase_x, phase_y = load_unwrapped_phases(phase_x_file, phase_y_file)
    
    # 执行重建（使用优化参数以更好地重建球形物体）
    print("执行三维重建（优化参数用于球形物体）...")
    reconstruct_3d_scene_enhanced(
        phase_x, phase_y,
        camera_matrix, projector_matrix, R, T,
        proj_width, proj_height,
        output_dir="basic_reconstruction_output",
        use_pso=False,  # 不使用PSO，速度更快
        step_size=5,    # 较小步长，提高精度
        create_mesh=True,  # 创建网格以更好地显示球形
        mask_percentile=92.0  # 更宽松的掩码以保留球形边缘
    )


def run_optimized_reconstruction():
    """运行优化的三维重建示例"""
    print("\n" + "="*50)
    print("优化三维重建示例（使用粒子群优化）")
    print("="*50)
    
    example_dir = "example_data"  # 使用之前创建的数据
    
    # 设置文件路径
    camera_file = os.path.join(example_dir, "camera_params.json")
    projector_file = os.path.join(example_dir, "projector_params.json")
    extrinsics_file = os.path.join(example_dir, "extrinsics.json")
    phase_x_file = os.path.join(example_dir, "phase_x.npy")
    phase_y_file = os.path.join(example_dir, "phase_y.npy")
    
    # 加载参数
    print("加载标定参数...")
    camera_matrix = load_camera_params(camera_file)
    projector_matrix, proj_width, proj_height = load_projector_params(projector_file)
    R, T = load_extrinsics(extrinsics_file)
    phase_x, phase_y = load_unwrapped_phases(phase_x_file, phase_y_file)
    
    # 执行优化重建
    print("执行三维重建（粒子群优化）...")
    reconstruct_3d_scene_enhanced(
        phase_x, phase_y,
        camera_matrix, projector_matrix, R, T,
        proj_width, proj_height,
        output_dir="optimized_reconstruction_output",
        use_pso=True,   # 使用PSO优化
        step_size=8,    # 较小步长，提高精度
        create_mesh=True,  # 创建网格
        mask_percentile=98.0
    )


def run_custom_reconstruction():
    """运行自定义参数的三维重建示例"""
    print("\n" + "="*50)
    print("自定义参数三维重建示例")
    print("="*50)
    
    example_dir = "example_data"
    
    # 创建重建对象
    camera_matrix = load_camera_params(os.path.join(example_dir, "camera_params.json"))
    projector_matrix, proj_width, proj_height = load_projector_params(
        os.path.join(example_dir, "projector_params.json")
    )
    R, T = load_extrinsics(os.path.join(example_dir, "extrinsics.json"))
    
    reconstructor = Enhanced3DReconstruction(
        camera_matrix, projector_matrix, R, T, proj_width, proj_height
    )
    
    # 加载相位图
    phase_x, phase_y = load_unwrapped_phases(
        os.path.join(example_dir, "phase_x.npy"),
        os.path.join(example_dir, "phase_y.npy")
    )
    
    # 创建自定义掩码
    mask = reconstructor.create_mask(phase_x, phase_y, percentile_threshold=99.0)
    
    # 生成点云（使用自定义参数）
    print("生成点云（自定义参数）...")
    points, colors, qualities = reconstructor.phase_to_pointcloud_optimized(
        phase_x, phase_y, mask, 
        use_pso=True, 
        step_size=6  # 中等精度
    )
    
    print(f"生成了 {len(points)} 个三维点")
    
    # 过滤点云
    filtered_points, filtered_colors = reconstructor.filter_pointcloud(
        points, colors, qualities, quality_threshold=3.0  # 较严格的质量阈值
    )
    
    print(f"过滤后保留 {len(filtered_points)} 个三维点")
    
    # 创建和保存点云
    if len(filtered_points) > 0:
        pcd = reconstructor.create_open3d_pointcloud(filtered_points, filtered_colors)
        
        # 保存点云
        output_dir = "custom_reconstruction_output"
        os.makedirs(output_dir, exist_ok=True)
        
        import open3d as o3d
        o3d.io.write_point_cloud(os.path.join(output_dir, "custom_pointcloud.ply"), pcd)
        print(f"点云已保存到: {output_dir}/custom_pointcloud.ply")
        
        # 可视化
        print("显示点云...")
        reconstructor.visualize_pointcloud(pcd, "自定义参数重建结果")


def main():
    """主函数"""
    print("增强三维重建系统使用示例")
    print("="*60)
    
    try:
        # 运行不同的重建示例
        run_basic_reconstruction()
        
        # 询问是否继续运行更复杂的示例
        choice = input("\n是否运行优化重建示例？(y/n): ").strip().lower()
        if choice == 'y':
            run_optimized_reconstruction()
        
        choice = input("\n是否运行自定义参数示例？(y/n): ").strip().lower()
        if choice == 'y':
            run_custom_reconstruction()
        
        print("\n示例运行完成!")
        
    except Exception as e:
        print(f"示例运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
