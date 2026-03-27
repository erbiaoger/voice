#!/usr/bin/env python3
"""
测试 SpectrogramCanvas 的放大缩小功能
"""

import sys
import numpy as np
from PyQt6 import QtWidgets
import matplotlib
matplotlib.use("QtAgg")

# 导入我们的类
from voice6 import SpectrogramCanvas, WaveformCanvas

def create_test_data():
    """创建测试数据"""
    # 时间轴
    t = np.linspace(0, 10, 1000)
    
    # 频率轴
    f = np.linspace(0, 100, 200)
    
    # 创建测试谱图数据（时频图）
    t_grid, f_grid = np.meshgrid(t, f)
    # 创建一个随时间变化的频率信号
    signal = np.sin(2 * np.pi * (10 + 5 * t_grid) * t_grid) * np.exp(-f_grid / 50)
    absZ = np.abs(signal)
    
    # 创建测试波形数据
    y_sig = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 15 * t)
    
    return absZ, t, f, y_sig

def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # 创建主窗口
    window = QtWidgets.QMainWindow()
    window.setWindowTitle("测试放大缩小功能")
    window.resize(1000, 800)
    
    # 创建中央部件
    central = QtWidgets.QWidget()
    window.setCentralWidget(central)
    
    # 创建布局
    layout = QtWidgets.QVBoxLayout(central)
    
    # 创建测试数据
    absZ, t, f, y_sig = create_test_data()
    
    # 创建时频图画布
    spec_canvas = SpectrogramCanvas()
    spec_canvas.draw_content(absZ, t, f, 0.01, 0.1)
    
    # 创建波形画布
    wave_canvas = WaveformCanvas()
    wave_canvas.draw_content(t, y_sig)
    
    # 添加到布局
    layout.addWidget(spec_canvas, stretch=2)
    layout.addWidget(wave_canvas, stretch=1)
    
    # 显示窗口
    window.show()
    
    print("测试说明：")
    print("1. 时频图上方会显示 matplotlib 工具栏")
    print("2. 点击放大镜图标可以放大缩小")
    print("3. 点击手形图标可以平移视图")
    print("4. 点击房子图标可以重置视图")
    print("5. 鼠标滚轮也可以缩放")
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
