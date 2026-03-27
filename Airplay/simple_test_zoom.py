#!/usr/bin/env python3
"""
简单的放大缩小功能测试脚本
不依赖外部模块，只测试基本的 matplotlib 功能
"""

import sys
import numpy as np
from PyQt6 import QtWidgets, QtCore
import matplotlib
matplotlib.use("QtAgg")

# 创建一个简单的 FigureCanvas 来测试
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class SimpleTestCanvas(FigureCanvas):
    """简单的测试画布，用于验证放大缩小功能"""
    
    def __init__(self, parent=None):
        fig = Figure(figsize=(8, 6))
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        
        # 延迟初始化导航工具
        QtCore.QTimer.singleShot(100, self._init_navigation_tools)
    
    def _init_navigation_tools(self):
        """初始化导航工具"""
        try:
            if hasattr(self.figure.canvas, 'manager') and self.figure.canvas.manager is not None:
                self.figure.canvas.manager.set_window_title("Test Canvas")
                self.figure.canvas.manager.toolbar.setVisible(True)
                self.figure.canvas.manager.toolbar.pan()
                self.figure.canvas.manager.toolbar.zoom()
                print("✓ 导航工具初始化成功")
            else:
                print("✗ 无法获取 canvas manager")
        except Exception as e:
            print(f"✗ 初始化导航工具失败: {e}")
    
    def plot_test_data(self):
        """绘制测试数据"""
        x = np.linspace(0, 10, 100)
        y = np.sin(x) * np.exp(-x/5)
        
        self.ax.clear()
        self.ax.plot(x, y, 'b-', linewidth=2, label='sin(x) * exp(-x/5)')
        self.ax.plot(x, np.cos(x) * 0.5, 'r--', linewidth=1, label='0.5 * cos(x)')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('测试图表 - 支持放大缩小')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.draw()
        
        # 重新启用导航工具
        try:
            if hasattr(self.figure.canvas, 'manager') and self.figure.canvas.manager is not None:
                self.figure.canvas.manager.toolbar.pan()
                self.figure.canvas.manager.toolbar.zoom()
        except Exception:
            pass

def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # 创建主窗口
    window = QtWidgets.QMainWindow()
    window.setWindowTitle("放大缩小功能测试")
    window.resize(800, 600)
    
    # 创建中央部件
    central = QtWidgets.QWidget()
    window.setCentralWidget(central)
    
    # 创建布局
    layout = QtWidgets.QVBoxLayout(central)
    
    # 创建测试画布
    canvas = SimpleTestCanvas()
    canvas.plot_test_data()
    
    # 添加说明标签
    info_label = QtWidgets.QLabel(
        "测试说明：\n"
        "1. 上方应该显示 matplotlib 工具栏\n"
        "2. 点击放大镜图标进入缩放模式\n"
        "3. 点击手形图标进入平移模式\n"
        "4. 使用鼠标滚轮或框选进行缩放\n"
        "5. 点击房子图标重置视图"
    )
    info_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; border: 1px solid #ccc; }")
    
    # 添加到布局
    layout.addWidget(canvas, stretch=1)
    layout.addWidget(info_label, stretch=0)
    
    # 显示窗口
    window.show()
    
    print("程序启动成功！")
    print("请检查是否显示了 matplotlib 工具栏")
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
