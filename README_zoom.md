# 放大缩小功能说明

## 新增功能

为 `SpectrogramCanvas` 和 `WaveformCanvas` 类添加了 matplotlib 内置的放大缩小和平移功能。

## 功能特性

### 1. 工具栏按钮
- **放大镜图标**：启用缩放模式
- **手形图标**：启用平移模式  
- **房子图标**：重置视图到原始状态
- **返回箭头**：返回上一个视图状态
- **前进箭头**：前进到下一个视图状态

### 2. 鼠标操作
- **鼠标滚轮**：在缩放模式下滚动可以放大/缩小
- **拖拽**：在平移模式下拖拽可以移动视图
- **框选**：在缩放模式下框选区域可以放大到该区域

### 3. 键盘快捷键
- **Home**：重置视图
- **Backspace**：返回上一个视图
- **Ctrl+Z**：撤销操作

## 使用方法

### 基本操作
1. 点击工具栏上的放大镜图标进入缩放模式
2. 使用鼠标滚轮或框选区域进行缩放
3. 点击手形图标进入平移模式
4. 拖拽鼠标移动视图
5. 点击房子图标重置视图

### 与播放功能的兼容性
- 放大缩小功能与现有的播放线（红色虚线）完全兼容
- 播放线会跟随视图缩放和平移
- 点击画布仍然可以设置播放位置

## 技术实现

### 导入的模块
```python
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
```

### 启用的功能
```python
# 在 __init__ 方法中
self.figure.canvas.manager.set_window_title("Spectrogram")
self.figure.canvas.manager.toolbar.setVisible(True)
self.figure.canvas.manager.toolbar.pan()
self.figure.canvas.manager.toolbar.zoom()

# 在 draw_content 方法中重新启用
self.figure.canvas.manager.toolbar.pan()
self.figure.canvas.manager.toolbar.zoom()
```

## 测试

运行测试脚本验证功能：
```bash
cd vehicle_track/preprocess
python test_zoom.py
```

## 注意事项

1. 每次调用 `draw_content` 后会自动重新启用导航工具
2. 工具栏会显示在每个画布的上方
3. 放大缩小操作不会影响音频播放功能
4. 视图状态会在画布重新绘制时保持

## 兼容性

- 支持 PyQt6
- 需要 matplotlib 3.0+
- 与现有的交互式播放线功能完全兼容
