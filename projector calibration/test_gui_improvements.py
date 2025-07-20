#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试GUI程序的改进功能
验证标定板类型的中文显示和动态标签更新
"""

import sys
import os

def test_gui_imports():
    """测试GUI程序的导入"""
    print("测试GUI程序导入...")
    
    try:
        # 测试PySide6导入
        from PySide6.QtWidgets import QApplication
        print("✓ PySide6导入成功")
        
        # 测试GUI程序导入
        sys.path.append(os.path.dirname(__file__))
        import projector_calibration_three_freq_gui as gui
        print("✓ GUI程序导入成功")
        
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_board_type_mapping():
    """测试标定板类型映射"""
    print("\n测试标定板类型映射...")
    
    try:
        import projector_calibration_three_freq_gui as gui
        
        # 创建应用程序实例（不显示窗口）
        app = gui.QApplication.instance()
        if app is None:
            app = gui.QApplication([])
        
        # 创建GUI实例
        window = gui.ThreeFreqProjectorCalibrationGUI()
        
        # 测试标定板类型映射
        expected_mapping = {
            "棋盘格标定板": "chessboard",
            "圆形标定板": "circles", 
            "环形标定板": "ring_circles"
        }
        
        assert hasattr(window, 'board_type_mapping'), "board_type_mapping属性不存在"
        assert window.board_type_mapping == expected_mapping, "标定板类型映射不正确"
        
        print("✓ 标定板类型映射正确")
        
        # 测试下拉框选项
        combo_items = [window.board_type_combo.itemText(i) for i in range(window.board_type_combo.count())]
        expected_items = ["棋盘格标定板", "圆形标定板", "环形标定板"]
        
        assert combo_items == expected_items, f"下拉框选项不正确: {combo_items}"
        print("✓ 下拉框选项正确")
        
        return True
    except Exception as e:
        print(f"✗ 标定板类型映射测试失败: {e}")
        return False

def test_dynamic_labels():
    """测试动态标签更新"""
    print("\n测试动态标签更新...")
    
    try:
        import projector_calibration_three_freq_gui as gui
        
        # 创建应用程序实例
        app = gui.QApplication.instance()
        if app is None:
            app = gui.QApplication([])
        
        # 创建GUI实例
        window = gui.ThreeFreqProjectorCalibrationGUI()
        
        # 测试棋盘格标定板标签
        window.board_type_combo.setCurrentText("棋盘格标定板")
        window.update_board_type_label()
        
        assert window.board_width_label.text() == "标定板宽度(内角点):", "棋盘格宽度标签不正确"
        assert window.board_height_label.text() == "标定板高度(内角点):", "棋盘格高度标签不正确"
        assert window.square_size_label.text() == "方格尺寸:", "棋盘格尺寸标签不正确"
        print("✓ 棋盘格标定板标签正确")
        
        # 测试圆形标定板标签
        window.board_type_combo.setCurrentText("圆形标定板")
        window.update_board_type_label()
        
        assert window.board_width_label.text() == "圆形数量(宽):", "圆形标定板宽度标签不正确"
        assert window.board_height_label.text() == "圆形数量(高):", "圆形标定板高度标签不正确"
        assert window.square_size_label.text() == "圆形直径:", "圆形标定板尺寸标签不正确"
        print("✓ 圆形标定板标签正确")
        
        # 测试环形标定板标签
        window.board_type_combo.setCurrentText("环形标定板")
        window.update_board_type_label()
        
        assert window.board_width_label.text() == "圆形数量(宽):", "环形标定板宽度标签不正确"
        assert window.board_height_label.text() == "圆形数量(高):", "环形标定板高度标签不正确"
        assert window.square_size_label.text() == "圆形直径:", "环形标定板尺寸标签不正确"
        print("✓ 环形标定板标签正确")
        
        return True
    except Exception as e:
        print(f"✗ 动态标签测试失败: {e}")
        return False

def test_parameter_conversion():
    """测试参数转换"""
    print("\n测试参数转换...")
    
    try:
        import projector_calibration_three_freq_gui as gui
        
        # 创建应用程序实例
        app = gui.QApplication.instance()
        if app is None:
            app = gui.QApplication([])
        
        # 创建GUI实例
        window = gui.ThreeFreqProjectorCalibrationGUI()
        
        # 设置一些测试值
        window.camera_params_edit.setText("test_camera.npz")
        window.phase_images_edit.setText("test_images")
        window.output_folder_edit.setText("test_output")
        
        # 测试棋盘格参数转换
        window.board_type_combo.setCurrentText("棋盘格标定板")
        params = window.get_calibration_params()
        assert params["board_type"] == "chessboard", "棋盘格类型转换失败"
        print("✓ 棋盘格参数转换正确")
        
        # 测试圆形标定板参数转换
        window.board_type_combo.setCurrentText("圆形标定板")
        params = window.get_calibration_params()
        assert params["board_type"] == "circles", "圆形标定板类型转换失败"
        print("✓ 圆形标定板参数转换正确")
        
        # 测试环形标定板参数转换
        window.board_type_combo.setCurrentText("环形标定板")
        params = window.get_calibration_params()
        assert params["board_type"] == "ring_circles", "环形标定板类型转换失败"
        print("✓ 环形标定板参数转换正确")
        
        return True
    except Exception as e:
        print(f"✗ 参数转换测试失败: {e}")
        return False

def test_default_values():
    """测试默认值设置"""
    print("\n测试默认值设置...")
    
    try:
        import projector_calibration_three_freq_gui as gui
        
        # 创建应用程序实例
        app = gui.QApplication.instance()
        if app is None:
            app = gui.QApplication([])
        
        # 创建GUI实例
        window = gui.ThreeFreqProjectorCalibrationGUI()
        
        # 测试棋盘格默认值
        window.board_type_combo.setCurrentText("棋盘格标定板")
        window.update_board_type_label()
        
        assert window.board_width_spin.value() == 9, "棋盘格默认宽度不正确"
        assert window.board_height_spin.value() == 6, "棋盘格默认高度不正确"
        assert window.square_size_spin.value() == 20.0, "棋盘格默认尺寸不正确"
        print("✓ 棋盘格默认值正确")
        
        # 测试圆形标定板默认值
        window.board_type_combo.setCurrentText("圆形标定板")
        window.update_board_type_label()
        
        assert window.board_width_spin.value() == 4, "圆形标定板默认宽度不正确"
        assert window.board_height_spin.value() == 11, "圆形标定板默认高度不正确"
        assert window.square_size_spin.value() == 20.0, "圆形标定板默认尺寸不正确"
        print("✓ 圆形标定板默认值正确")
        
        return True
    except Exception as e:
        print(f"✗ 默认值测试失败: {e}")
        return False

def show_improvement_summary():
    """显示改进总结"""
    print("\n" + "="*60)
    print("GUI程序改进总结")
    print("="*60)
    
    print("\n✅ 已完成的改进:")
    print("1. 标定板类型选择改为中文显示:")
    print("   - 棋盘格标定板 (chessboard)")
    print("   - 圆形标定板 (circles)")
    print("   - 环形标定板 (ring_circles)")
    
    print("\n2. 动态标签更新:")
    print("   - 棋盘格: 标定板宽度/高度(内角点), 方格尺寸")
    print("   - 圆形/环形: 圆形数量(宽/高), 圆形直径")
    
    print("\n3. 智能默认值:")
    print("   - 棋盘格: 9x6, 20mm")
    print("   - 圆形/环形: 4x11, 20mm")
    
    print("\n4. 工具提示信息:")
    print("   - 为每种标定板类型添加了详细说明")
    
    print("\n5. 参数转换:")
    print("   - 中文界面显示，英文参数传递")
    print("   - 确保与后端标定程序兼容")
    
    print("\n🔧 技术特点:")
    print("- 保持与projector_calibration_three_freq.py的完全兼容")
    print("- 环形标定板使用与camera_calibration.py相同的检测方法")
    print("- 用户友好的中文界面")
    print("- 智能的参数验证和默认值设置")

def main():
    """主函数"""
    print("GUI程序改进功能测试")
    print("="*60)
    
    # 运行所有测试
    tests = [
        test_gui_imports,
        test_board_type_mapping,
        test_dynamic_labels,
        test_parameter_conversion,
        test_default_values
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    # 显示测试结果
    print("\n" + "="*60)
    print("测试结果总结:")
    print("="*60)
    
    test_names = [
        "GUI程序导入测试",
        "标定板类型映射测试",
        "动态标签更新测试",
        "参数转换测试",
        "默认值设置测试"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{i+1}. {name}: {status}")
    
    all_passed = all(results)
    print(f"\n总体结果: {'✓ 所有测试通过' if all_passed else '✗ 部分测试失败'}")
    
    if all_passed:
        print("\n🎉 GUI程序改进完成！")
        show_improvement_summary()
    else:
        print("\n⚠️  请检查失败的测试项目")

if __name__ == "__main__":
    main()
