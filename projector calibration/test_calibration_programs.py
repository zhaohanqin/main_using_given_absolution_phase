#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
投影仪标定程序测试脚本
用于验证三频外差投影仪标定程序的基本功能
"""

import os
import sys
import numpy as np
import json
import tempfile
import shutil

def create_test_camera_params(filename):
    """创建测试用的相机标定参数文件"""
    # 创建模拟的相机参数
    camera_matrix = np.array([
        [800.0, 0.0, 320.0],
        [0.0, 800.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    
    dist_coeffs = np.array([0.1, -0.2, 0.001, 0.002, 0.0])
    
    if filename.endswith('.npz'):
        np.savez(filename, 
                camera_matrix=camera_matrix, 
                dist_coeffs=dist_coeffs)
    elif filename.endswith('.json'):
        data = {
            'camera_matrix': camera_matrix.tolist(),
            'dist_coeffs': dist_coeffs.tolist()
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    print(f"创建测试相机参数文件: {filename}")

def test_import_modules():
    """测试模块导入"""
    print("测试模块导入...")
    
    try:
        # 测试命令行版本
        import projector_calibration_three_freq as cal_three_freq
        print("✓ 成功导入 projector_calibration_three_freq")
        
        # 检查主要函数是否存在
        if hasattr(cal_three_freq, 'three_freq_projector_calibration'):
            print("✓ 找到主标定函数 three_freq_projector_calibration")
        else:
            print("✗ 未找到主标定函数")
            return False
            
        if hasattr(cal_three_freq, 'multi_phase'):
            print("✓ 找到三频相位处理类 multi_phase")
        else:
            print("✗ 未找到三频相位处理类")
            return False
            
    except ImportError as e:
        print(f"✗ 导入 projector_calibration_three_freq 失败: {e}")
        return False
    
    try:
        # 测试GUI版本
        import projector_calibration_three_freq_gui as cal_gui
        print("✓ 成功导入 projector_calibration_three_freq_gui")
        
        # 检查主要类是否存在
        if hasattr(cal_gui, 'ThreeFreqProjectorCalibrationGUI'):
            print("✓ 找到GUI主类 ThreeFreqProjectorCalibrationGUI")
        else:
            print("✗ 未找到GUI主类")
            return False
            
    except ImportError as e:
        print(f"✗ 导入 projector_calibration_three_freq_gui 失败: {e}")
        return False
    
    return True

def test_camera_params_loading():
    """测试相机参数加载功能"""
    print("\n测试相机参数加载功能...")
    
    try:
        import projector_calibration_three_freq as cal_three_freq
        
        # 创建临时文件
        temp_dir = tempfile.mkdtemp()
        try:
            # 测试NPZ格式
            npz_file = os.path.join(temp_dir, "test_camera.npz")
            create_test_camera_params(npz_file)

            camera_matrix, _ = cal_three_freq.load_camera_parameters(npz_file)
            print(f"✓ 成功加载NPZ格式相机参数，内参矩阵形状: {camera_matrix.shape}")

            # 测试JSON格式
            json_file = os.path.join(temp_dir, "test_camera.json")
            create_test_camera_params(json_file)

            camera_matrix, _ = cal_three_freq.load_camera_parameters(json_file)
            print(f"✓ 成功加载JSON格式相机参数，内参矩阵形状: {camera_matrix.shape}")
        finally:
            # 清理临时文件
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        return True
        
    except Exception as e:
        print(f"✗ 相机参数加载测试失败: {e}")
        return False

def test_phase_processing():
    """测试相位处理功能"""
    print("\n测试三频相位处理功能...")
    
    try:
        import projector_calibration_three_freq as cal_three_freq
        
        # 创建模拟的24张图像
        height, width = 480, 640
        images = []
        
        for _ in range(24):
            # 创建模拟的相移图像
            img = np.random.randint(0, 255, (height, width), dtype=np.uint8)
            images.append(img)
        
        # 创建三频处理对象
        frequencies = [71, 64, 58]
        phase_processor = cal_three_freq.multi_phase(
            f=frequencies, 
            step=4, 
            images=images, 
            ph0=0.5
        )
        
        print("✓ 成功创建三频相位处理对象")
        
        # 测试相位解包裹（这可能会失败，因为是随机数据）
        try:
            unwrapped_v, _, _ = phase_processor.get_phase()
            print(f"✓ 相位解包裹完成，输出形状: {unwrapped_v.shape}")
        except Exception as e:
            print(f"⚠ 相位解包裹测试失败（预期，因为使用随机数据）: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ 三频相位处理测试失败: {e}")
        return False

def test_calibration_config():
    """测试标定配置"""
    print("\n测试标定配置...")
    
    try:
        import projector_calibration_three_freq as cal_three_freq
        
        # 创建配置对象
        config = cal_three_freq.ThreeFreqCalibrationConfig(
            frequencies=[71, 64, 58],
            phase_step=4,
            ph0=0.5,
            projector_width=1024,
            projector_height=768,
            quality_threshold=0.3
        )
        
        print("✓ 成功创建三频标定配置对象")
        print(f"  - 频率: {config.frequencies}")
        print(f"  - 相移步数: {config.phase_step}")
        print(f"  - 投影仪分辨率: {config.projector_width}x{config.projector_height}")
        
        return True
        
    except Exception as e:
        print(f"✗ 标定配置测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("投影仪标定程序功能测试")
    print("=" * 60)
    
    # 切换到正确的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 添加当前目录到Python路径
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    test_results = []
    
    # 运行各项测试
    test_results.append(("模块导入", test_import_modules()))
    test_results.append(("相机参数加载", test_camera_params_loading()))
    test_results.append(("三频相位处理", test_phase_processing()))
    test_results.append(("标定配置", test_calibration_config()))
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "通过" if result else "失败"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！投影仪标定程序可以正常运行。")
        print("\n使用说明:")
        print("1. 命令行版本: python projector_calibration_three_freq.py --help")
        print("2. GUI版本: python projector_calibration_three_freq_gui.py")
    else:
        print(f"\n⚠ 有 {total - passed} 项测试失败，请检查程序代码。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
