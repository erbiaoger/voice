# -*- coding: utf-8 -*-
import sys
import os
import tempfile
from pathlib import Path

import numpy as np
from obspy import read
from scipy.signal import resample, lfilter
from scipy.io import wavfile

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal, QUrl
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer, QMediaDevices
from PyQt6.QtGui import QAction

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ===== dasQt：项目内部函数 =====
from dasQt.process import spectrogram as sp_spectrogram
from dasQt.process import (
    bandpass, bandstop, lowpass,
    lowpass_cheby_2, highpass, envelope,
    median_filter2
)

# ===================== 音频与时频工具 =====================
def sac_to_audio_wav(
    data, fs_orig,
    audio_rate=16000,
    gain=50.0,
    speed_up=20.0,
    ear_enhance=False,
    ear_bright_db=3.0,
    comp_ratio=2.0
):
    """
    将地震段写成临时 WAV（Qt 可播），返回 wav_path 与压缩后时长（秒）
    - 先按 speed_up 压缩时间（提高频率）
    - 归一化并放大 gain
    - 可选“人耳增强”：预加重 + 轻度压缩 + 亮度提升（高频微抬）
    """
    x = data.astype(np.float64)

    # === 预加重（在“地震原采样率域”做一点点高频提升，轻微）
    # 目的是让后续加速后中高频更清晰；系数取值很温和，避免刺耳
    if ear_enhance:
        pre_emph = 0.93
        x = lfilter([1.0, -pre_emph], [1.0], x)

    # === 时间压缩 ===
    # 目标是把原始 fs 映射到 audio_rate，同时保证加速因子 speed_up
    fs_target = fs_orig * speed_up
    n_samples = int(len(x) * audio_rate / fs_target)
    n_samples = max(n_samples, 2)
    y = resample(x, n_samples)

    # === 亮度提升（在“音频域”做一点点高频抬升）===
    if ear_enhance and ear_bright_db != 0:
        # 一阶高架滤波（简单的 shelf 近似）：y[n] = y[n] + b*(y[n]-y[n-1])
        # 系数用一个很温和的比例，避免噪声被过度拉起
        # 将 dB 转增益
        g = 10 ** (ear_bright_db / 20.0)
        # 把差分作为“高频分量”的近似
        hf = np.concatenate([[0.0], np.diff(y)])
        y = y + (g - 1.0) * 0.2 * hf  # 0.2 控制量级

    # === 轻度动态压缩（避免尖峰导致听感失真）===
    if ear_enhance and comp_ratio > 1.0:
        # 简易压缩器（静态曲线）：大于阈值的部分按比例压缩
        thr = 0.25  # 轻微阈值
        mag = np.abs(y)
        sign = np.sign(y)
        over = np.maximum(mag - thr, 0.0)
        mag_comp = thr + over / comp_ratio
        y = sign * np.where(mag > thr, mag_comp, mag)

    # === 归一化 + 增益 ===
    m = np.max(np.abs(y)) if y.size else 1.0
    if m > 0:
        y = y / m
    y = y * gain

    # 二次限幅，防溢出
    y = np.clip(y, -1.0, 1.0)
    audio_int16 = np.clip(y * 32767.0, -32768, 32767).astype(np.int16)

    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    wavfile.write(wav_path, int(audio_rate), audio_int16)
    return wav_path, len(audio_int16) / float(audio_rate)


def compute_spectrogram(data, fs, nperseg=512, noverlap=256, end_freq=999, vmin=0.01, vmax=0.1):
    """用 dasQt 的 spectrogram（返回 Zxx, f, t）生成归一化谱图，并裁剪到 end_freq"""
    Zxx, f, t = sp_spectrogram(
        data, fs=fs, nperseg=nperseg, noverlap=noverlap,
        nfft=None, detrend=False, boundary='zeros'
    )
    absZ = np.abs(Zxx)
    m = np.max(absZ) if absZ.size else 1.0
    if m > 0:
        absZ = absZ / m
    # 裁到 end_freq
    end_idx = np.argmin(np.abs(f - end_freq))
    absZ = absZ[:end_idx, :]
    f = f[:end_idx]
    return absZ, t, f, float(vmin), float(vmax)


# ===================== 滤波封装（保持 dasQt 调用） =====================
def apply_filter(data: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    """
    根据 cfg['type'] 选择滤波器。
    cfg = {
        'enabled': bool,
        'type': 'bandpass'|'bandstop'|'lowpass'|'highpass'|'lowpass_cheby_2'|'envelope'|'median',
        'freqmin': float, 'freqmax': float, 'corners': int,
        'zerophase': bool, 'detrend': bool, 'taper': bool,
        'median_kernel': int
    }
    """
    if not cfg.get('enabled', False):
        return data

    ftype = cfg.get('type', 'bandpass')
    freqmin = cfg.get('freqmin', 0.1)
    freqmax = cfg.get('freqmax', 2.0)
    corners = int(cfg.get('corners', 4))
    zerophase = bool(cfg.get('zerophase', True))
    detr = bool(cfg.get('detrend', False))
    tap = bool(cfg.get('taper', False))

    try:
        if ftype == 'bandpass':
            y = bandpass(data, fs, freqmin=freqmin, freqmax=freqmax,
                         corners=corners, zerophase=zerophase,
                         detrend=detr, taper=tap)
        elif ftype == 'bandstop':
            y = bandstop(data, fs, freqmin=freqmin, freqmax=freqmax,
                         corners=corners, zerophase=zerophase,
                         detrend=detr, taper=tap)
        elif ftype == 'lowpass':
            y = lowpass(data, fs, freq=freqmax, corners=corners,
                        zerophase=zerophase, detrend=detr, taper=tap)
        elif ftype == 'lowpass_cheby_2':
            y = lowpass_cheby_2(data, fs, freq=freqmax, zerophase=zerophase)
        elif ftype == 'highpass':
            y = highpass(data, fs, freq=freqmin, corners=corners,
                         zerophase=zerophase, detrend=detr, taper=tap)
        elif ftype == 'envelope':
            env = envelope(data)
            y = env if freqmax <= 0 else lowpass(env, fs, freq=freqmax,
                                                 corners=max(2, corners),
                                                 zerophase=True)
        elif ftype == 'median':
            k = int(cfg.get('median_kernel', 9))
            k = max(3, k | 1)  # 确保奇数 >= 3
            try:
                y = median_filter2(data, kernel_size=k)
            except TypeError:
                y = median_filter2(data, k)
        else:
            y = data
    except Exception as e:
        print(f"[filter] {ftype} failed: {e}")
        y = data

    if np.any(~np.isfinite(y)):
        y = np.nan_to_num(y, copy=False)
    return y


# ===================== Matplotlib 交互画布 =====================
class BaseInteractiveCanvas(FigureCanvas):
    """带可拖动垂直红线的通用画布；释放鼠标时发射 seekRequested(original_time)"""
    seekRequested = pyqtSignal(float)  # 原始时间（秒）

    def __init__(self, figsize=(9, 3.6), dpi=100, parent=None):
        fig = Figure(figsize=figsize, dpi=dpi)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.playline = None
        self._xlim = (0.0, 1.0)
        self._dragging = False
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setMouseTracking(True)
        self._artists_to_redraw = []

    def draw_content(self, *args, **kwargs):
        raise NotImplementedError

    def init_playline(self, x0=0.0):
        if self.playline is not None:
            try:
                self.playline.remove()
            except Exception:
                pass
        self.playline = self.ax.axvline(x0, color='red', lw=1.8, ls='--')
        self.draw_idle()

    def update_playhead(self, x):
        if self.playline is None:
            return
        self.playline.set_xdata([x, x])
        self.ax.draw_artist(self.ax.patch)
        for a in self._artists_to_redraw:
            self.ax.draw_artist(a)
        self.ax.draw_artist(self.playline)
        self.figure.canvas.update()
        self.figure.canvas.flush_events()

    # Qt 像素坐标 → Axes 数据坐标（仅 x）
    def _mouse_x_to_data(self, qevent: QtGui.QMouseEvent):
        px_qt = qevent.position().x()
        py_qt = qevent.position().y()
        dpr = float(self.devicePixelRatioF())
        px = px_qt * dpr
        fig_height = float(self.figure.bbox.height)
        py = fig_height - (py_qt * dpr)
        bbox = self.ax.get_window_extent()
        x0, y0, x1, y1 = float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)
        if not (x0 <= px <= x1 and y0 <= py <= y1):
            return None
        x_min, x_max = self.ax.get_xbound()
        if x1 == x0:
            return None
        frac = (px - x0) / (x1 - x0)
        xdata = x_min + frac * (x_max - x_min)
        xdata = max(min(xdata, self._xlim[1]), self._xlim[0])
        return float(xdata)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            x = self._mouse_x_to_data(event)
            if x is not None:
                self._dragging = True
                self.update_playhead(x)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self._dragging:
            x = self._mouse_x_to_data(event)
            if x is not None:
                self.update_playhead(x)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            x = self._mouse_x_to_data(event)
            if x is not None:
                self.seekRequested.emit(float(x))
        super().mouseReleaseEvent(event)


class SpectrogramCanvas(BaseInteractiveCanvas):
    def __init__(self, parent=None):
        super().__init__(figsize=(9, 4.2), dpi=100, parent=parent)
        self.im = None
        self.current_cmap = "jet"

    def draw_content(self, absZ, t, f, vmin, vmax, cmap_name=None):
        if cmap_name is not None:
            self.current_cmap = cmap_name

        self.ax.clear()
        self.im = self.ax.imshow(
            absZ, aspect='auto', origin='lower',
            cmap=self.current_cmap,
            interpolation='bicubic',
            extent=[t[0], t[-1], f[0], f[-1]],
            vmin=vmin, vmax=vmax
        )
        self._artists_to_redraw = [self.im]
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Frequency (Hz)')
        self._xlim = (float(t[0]), float(t[-1]))
        self.init_playline(self._xlim[0])
        self.figure.tight_layout()
        self.draw()


class WaveformCanvas(BaseInteractiveCanvas):
    def __init__(self, parent=None):
        super().__init__(figsize=(9, 2.8), dpi=100, parent=parent)
        self.line = None

    def draw_content(self, t_sig, y_sig, title="Waveform"):
        self.ax.clear()
        (self.line,) = self.ax.plot(t_sig, y_sig, lw=0.9)
        self._artists_to_redraw = [self.line]
        self.ax.margins(x=0)  # 左右顶格
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title(title, fontsize=10)
        self._xlim = (float(t_sig[0]), float(t_sig[-1]))
        ypad = 0.05 * (np.max(y_sig) - np.min(y_sig) + 1e-12)
        self.ax.set_ylim(np.min(y_sig) - ypad, np.max(y_sig) + ypad)
        self.init_playline(self._xlim[0])
        self.figure.tight_layout()
        self.draw()


# ===================== 主窗口 =====================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAC → 音频（人耳增强可选） + 时频/波形同步（PyQt6, dasQt）")
        self.resize(1300, 980)

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        vlayout = QtWidgets.QVBoxLayout(central)

        # ---- 控件区 ----
        ctrl = QtWidgets.QGridLayout()
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("选择 .sac 文件...")
        self.path_edit.setText("")  # 你可以填默认路径
        browse_btn = QtWidgets.QPushButton("浏览"); browse_btn.clicked.connect(self.browse_sac)

        self.start_sec = QtWidgets.QDoubleSpinBox(); self.start_sec.setRange(0.0, 1e9)
        self.start_sec.setDecimals(1); self.start_sec.setSingleStep(1.0); self.start_sec.setValue(0.0)
        self.win_len = QtWidgets.QDoubleSpinBox(); self.win_len.setRange(1.0, 1e6)
        self.win_len.setDecimals(1); self.win_len.setValue(60.0)

        self.gain = QtWidgets.QDoubleSpinBox(); self.gain.setRange(0.1, 1e4)
        self.gain.setDecimals(1); self.gain.setValue(50.0)
        self.speed_up = QtWidgets.QDoubleSpinBox(); self.speed_up.setRange(1.0, 1e4)
        self.speed_up.setDecimals(1); self.speed_up.setValue(20.0)
        self.audio_rate = QtWidgets.QSpinBox(); self.audio_rate.setRange(8000, 192000); self.audio_rate.setValue(16000)

        self.nperseg = QtWidgets.QSpinBox(); self.nperseg.setRange(64, 8192); self.nperseg.setValue(512)
        self.noverlap = QtWidgets.QSpinBox(); self.noverlap.setRange(0, 8000); self.noverlap.setValue(256)
        self.end_freq = QtWidgets.QDoubleSpinBox(); self.end_freq.setRange(1.0, 1e6); self.end_freq.setDecimals(0); self.end_freq.setValue(999.0)

        # vmin / vmax
        self.vmin_box = QtWidgets.QDoubleSpinBox(); self.vmin_box.setRange(-10.0, 10.0); self.vmin_box.setDecimals(6); self.vmin_box.setValue(0.0001)
        self.vmax_box = QtWidgets.QDoubleSpinBox(); self.vmax_box.setRange(-10.0, 10.0); self.vmax_box.setDecimals(6); self.vmax_box.setValue(0.02)

        # cmap 下拉
        self.cmap_combo = QtWidgets.QComboBox()
        cmap_list = [
            "jet", "viridis", "plasma", "inferno", "magma", "cividis", "turbo",
            "Spectral", "RdYlBu", "coolwarm", "terrain", "ocean", "cubehelix",
            "Greys", "gray", "bone", "hot"
        ]
        self.cmap_combo.addItems(cmap_list)
        self.cmap_combo.setCurrentText("jet")
        self.cmap_combo.currentTextChanged.connect(self.on_cmap_changed)

        # 播放与设备
        self.load_btn = QtWidgets.QPushButton("载入并绘制"); self.load_btn.clicked.connect(self.load_and_plot)
        self.play_btn = QtWidgets.QPushButton("播放 / 暂停"); self.play_btn.setEnabled(False); self.play_btn.clicked.connect(self.toggle_play)

        self.dev_combo = QtWidgets.QComboBox(); self.dev_combo.setMinimumWidth(280)
        for dev in QMediaDevices.audioOutputs():
            self.dev_combo.addItem(dev.description(), dev)
        self.dev_combo.currentIndexChanged.connect(self._on_device_changed)

        # 滤波相关
        self.filter_enable = QtWidgets.QCheckBox("启用滤波"); self.filter_enable.setChecked(False)
        self.filter_type = QtWidgets.QComboBox()
        self.filter_type.addItems([
            "bandpass", "bandstop", "lowpass", "highpass",
            "lowpass_cheby_2", "envelope", "median"
        ])
        self.filter_type.setCurrentText("bandpass")
        self.fmin_box = QtWidgets.QDoubleSpinBox(); self.fmin_box.setRange(0.0, 1e5); self.fmin_box.setDecimals(3); self.fmin_box.setValue(0.1)
        self.fmax_box = QtWidgets.QDoubleSpinBox(); self.fmax_box.setRange(0.0, 1e5); self.fmax_box.setDecimals(3); self.fmax_box.setValue(5.0)
        self.corners_box = QtWidgets.QSpinBox(); self.corners_box.setRange(1, 12); self.corners_box.setValue(4)
        self.median_box = QtWidgets.QSpinBox(); self.median_box.setRange(3, 1001); self.median_box.setSingleStep(2); self.median_box.setValue(9)
        self.zerophase_box = QtWidgets.QCheckBox("zerophase"); self.zerophase_box.setChecked(True)
        self.detrend_box  = QtWidgets.QCheckBox("detrend"); self.detrend_box.setChecked(False)
        self.taper_box    = QtWidgets.QCheckBox("taper"); self.taper_box.setChecked(False)

        # 人耳增强
        self.ear_enable = QtWidgets.QCheckBox("人耳增强(预加重+压缩+亮度)")
        self.ear_enable.setChecked(True)
        self.ear_bright = QtWidgets.QDoubleSpinBox(); self.ear_bright.setRange(-12.0, 12.0)
        self.ear_bright.setDecimals(1); self.ear_bright.setValue(3.0)
        self.comp_ratio = QtWidgets.QDoubleSpinBox(); self.comp_ratio.setRange(1.0, 20.0)
        self.comp_ratio.setDecimals(1); self.comp_ratio.setValue(2.0)

        # ====== 控件布局 ======
        r = 0
        ctrl.addWidget(QtWidgets.QLabel("SAC 文件:"), r, 0); ctrl.addWidget(self.path_edit, r, 1, 1, 6); ctrl.addWidget(browse_btn, r, 7)
        r += 1
        ctrl.addWidget(QtWidgets.QLabel("起始(s):"), r, 0); ctrl.addWidget(self.start_sec, r, 1)
        ctrl.addWidget(QtWidgets.QLabel("时长(s):"), r, 2); ctrl.addWidget(self.win_len, r, 3)
        ctrl.addWidget(QtWidgets.QLabel("增益:"), r, 4); ctrl.addWidget(self.gain, r, 5)
        ctrl.addWidget(QtWidgets.QLabel("加速倍数:"), r, 6); ctrl.addWidget(self.speed_up, r, 7)
        r += 1
        ctrl.addWidget(QtWidgets.QLabel("音频采样率:"), r, 0); ctrl.addWidget(self.audio_rate, r, 1)
        ctrl.addWidget(QtWidgets.QLabel("nperseg:"), r, 2); ctrl.addWidget(self.nperseg, r, 3)
        ctrl.addWidget(QtWidgets.QLabel("noverlap:"), r, 4); ctrl.addWidget(self.noverlap, r, 5)
        ctrl.addWidget(QtWidgets.QLabel("最大频率(Hz):"), r, 6); ctrl.addWidget(self.end_freq, r, 7)
        r += 1
        ctrl.addWidget(QtWidgets.QLabel("vmin:"), r, 0); ctrl.addWidget(self.vmin_box, r, 1)
        ctrl.addWidget(QtWidgets.QLabel("vmax:"), r, 2); ctrl.addWidget(self.vmax_box, r, 3)
        ctrl.addWidget(QtWidgets.QLabel("cmap:"), r, 4); ctrl.addWidget(self.cmap_combo, r, 5)
        ctrl.addWidget(QtWidgets.QLabel("输出设备:"), r, 6); ctrl.addWidget(self.dev_combo, r, 7)

        # 滤波布局
        r += 1
        ctrl.addWidget(self.filter_enable, r, 0)
        ctrl.addWidget(QtWidgets.QLabel("滤波类型:"), r, 1); ctrl.addWidget(self.filter_type, r, 2)
        ctrl.addWidget(QtWidgets.QLabel("fmin:"), r, 3); ctrl.addWidget(self.fmin_box, r, 4)
        ctrl.addWidget(QtWidgets.QLabel("fmax:"), r, 5); ctrl.addWidget(self.fmax_box, r, 6)

        r += 1
        ctrl.addWidget(QtWidgets.QLabel("corners:"), r, 0); ctrl.addWidget(self.corners_box, r, 1)
        ctrl.addWidget(QtWidgets.QLabel("median_k:"), r, 2); ctrl.addWidget(self.median_box, r, 3)
        ctrl.addWidget(self.zerophase_box, r, 4)
        ctrl.addWidget(self.detrend_box,  r, 5)
        ctrl.addWidget(self.taper_box,    r, 6)

        # 人耳增强布局
        r += 1
        ctrl.addWidget(self.ear_enable, r, 0)
        ctrl.addWidget(QtWidgets.QLabel("亮度提升(dB):"), r, 1); ctrl.addWidget(self.ear_bright, r, 2)
        ctrl.addWidget(QtWidgets.QLabel("压缩比:"), r, 3); ctrl.addWidget(self.comp_ratio, r, 4)
        ctrl.addWidget(self.load_btn, r, 6); ctrl.addWidget(self.play_btn, r, 7)

        vlayout.addLayout(ctrl)

        # ---- 画布区：上时频，中滤波后波形，下原始波形 ----
        self.spec_canvas = SpectrogramCanvas(self)
        self.wave_canvas_filt = WaveformCanvas(self)
        self.wave_canvas_raw  = WaveformCanvas(self)

        # 连接拖拽跳播
        self.spec_canvas.seekRequested.connect(self.on_seek_requested)
        self.wave_canvas_filt.seekRequested.connect(self.on_seek_requested)
        self.wave_canvas_raw.seekRequested.connect(self.on_seek_requested)

        vlayout.addWidget(self.spec_canvas, stretch=2)
        vlayout.addWidget(self.wave_canvas_filt, stretch=1)
        vlayout.addWidget(self.wave_canvas_raw, stretch=1)

        # ---- 多媒体 ----
        self.player = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        default_dev = QMediaDevices.defaultAudioOutput()
        self.audio_out.setDevice(default_dev)
        self.audio_out.setMuted(False)
        self.audio_out.setVolume(1.0)
        self.player.setAudioOutput(self.audio_out)

        self.timer = QtCore.QTimer(self); self.timer.setInterval(30); self.timer.timeout.connect(self.on_timer)

        # 信号
        self.player.positionChanged.connect(self.on_position_changed)
        self.player.playbackStateChanged.connect(self.on_state_changed)
        self.player.mediaStatusChanged.connect(self._on_media_status)
        self.player.errorOccurred.connect(self._on_media_error)

        # 内部状态
        self.fs = None
        self.seg_t = None           # 时频横轴（秒）
        self.t_sig = None           # 波形横轴（秒）
        self.seg_raw = None         # 当前片段（原始）
        self.seg_filt = None        # 当前片段（滤波后）
        self.wav_path = None
        self.wav_duration = 0.0
        self.current_speedup = 1.0
        self.last_spec = None       # (absZ, t, f, vmin, vmax)

        # 菜单
        exit_act = QAction("退出", self); exit_act.triggered.connect(self.close)
        menubar = self.menuBar(); file_menu = menubar.addMenu("文件"); file_menu.addAction(exit_act)
        self.statusBar().showMessage("就绪")

        # 启动时根据下拉框设置设备
        self._on_device_changed(self.dev_combo.currentIndex())

    # ========== 事件与槽 ==========
    def _on_device_changed(self, idx):
        dev = self.dev_combo.itemData(idx)
        if dev:
            self.audio_out.setDevice(dev)
            self.statusBar().showMessage(f"切换输出设备：{dev.description()}")

    def _on_media_status(self, status):
        self.statusBar().showMessage(f"[mediaStatus] {status}")

    def _on_media_error(self, err, *args):
        err_str = ""
        try:
            err_str = self.player.errorString()
        except Exception:
            pass
        self.statusBar().showMessage(f"[mediaError] code={err} {err_str}")

    def browse_sac(self):
        pth, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 SAC 文件", str(Path.home()), "SAC Files (*.sac);;All Files (*)"
        )
        if pth:
            self.path_edit.setText(pth)

    def load_and_plot(self):
        try:
            sac_path = Path(self.path_edit.text()).expanduser()
            if not sac_path.exists():
                QtWidgets.QMessageBox.warning(self, "错误", f"找不到文件：\n{sac_path}")
                return

            st = read(str(sac_path)); tr = st[0]
            self.fs = float(tr.stats.sampling_rate)
            data = tr.data.astype(np.float64)

            start      = float(self.start_sec.value())
            length     = float(self.win_len.value())
            gain       = float(self.gain.value())
            speed_up   = float(self.speed_up.value())
            audio_rate = int(self.audio_rate.value())
            nperseg    = int(self.nperseg.value())
            noverlap   = int(self.noverlap.value())
            end_freq   = float(self.end_freq.value())
            vmin       = float(self.vmin_box.value())
            vmax       = float(self.vmax_box.value())
            cmap_name  = self.cmap_combo.currentText()

            if vmax <= vmin:
                vmax = vmin + 1e-9
                self.vmax_box.setValue(vmax)

            i0 = max(int(start * self.fs), 0)
            i1 = min(int((start + length) * self.fs), len(data))
            if i1 - i0 < max(2, nperseg):
                QtWidgets.QMessageBox.warning(self, "错误", "选择窗口过短，无法计算谱图。")
                return

            seg = data[i0:i1]
            self.seg_raw = seg.copy()

            # === 读取滤波配置并应用 ===
            f_cfg = {
                'enabled': self.filter_enable.isChecked(),
                'type': self.filter_type.currentText(),
                'freqmin': float(self.fmin_box.value()),
                'freqmax': float(self.fmax_box.value()),
                'corners': int(self.corners_box.value()),
                'zerophase': bool(self.zerophase_box.isChecked()),
                'detrend': bool(self.detrend_box.isChecked()),
                'taper': bool(self.taper_box.isChecked()),
                'median_kernel': int(self.median_box.value()),
            }
            if f_cfg['type'] in ('bandpass', 'bandstop') and f_cfg['freqmax'] <= f_cfg['freqmin']:
                f_cfg['freqmax'] = f_cfg['freqmin'] + 1e-6
                self.fmax_box.setValue(f_cfg['freqmax'])

            self.seg_filt = apply_filter(seg, self.fs, f_cfg)

            # === 计算谱图（基于滤波后） ===
            absZ, t, f, vmin, vmax = compute_spectrogram(
                self.seg_filt, self.fs, nperseg=nperseg, noverlap=noverlap,
                end_freq=end_freq, vmin=vmin, vmax=vmax
            )
            self.last_spec = (absZ, t, f, vmin, vmax)
            self.spec_canvas.draw_content(absZ, t, f, vmin, vmax, cmap_name=cmap_name)
            self.seg_t = t

            # === 画波形：滤波后 + 原始 ===
            self.t_sig = np.arange(len(seg), dtype=np.float64) / self.fs
            self.wave_canvas_filt.draw_content(self.t_sig, self.seg_filt, title="Filtered Waveform")
            self.wave_canvas_raw.draw_content(self.t_sig, self.seg_raw, title="Raw Waveform")

            self.current_speedup = speed_up

            # === 生成 WAV（基于滤波后 + 人耳增强可选） ===
            if self.wav_path and Path(self.wav_path).exists():
                try:
                    os.remove(self.wav_path)
                except Exception:
                    pass

            self.wav_path, self.wav_duration = sac_to_audio_wav(
                self.seg_filt, self.fs,
                audio_rate=audio_rate,
                gain=gain,
                speed_up=speed_up,
                ear_enhance=self.ear_enable.isChecked(),
                ear_bright_db=float(self.ear_bright.value()),
                comp_ratio=float(self.comp_ratio.value())
            )
            try:
                rate_check, data_check = wavfile.read(self.wav_path)
                print("[WAV] readback:", rate_check, data_check.dtype, data_check.shape)
            except Exception as e:
                print("[WAV] readback error:", e)

            self.player.setSource(QUrl.fromLocalFile(self.wav_path))
            self.play_btn.setEnabled(True)
            self.statusBar().showMessage(f"片段已载入，压缩后音频时长 {self.wav_duration:.2f}s。")
            # 光标归零
            for c in (self.spec_canvas, self.wave_canvas_filt, self.wave_canvas_raw):
                c.update_playhead(0.0)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "载入错误", str(e))
            raise

    def on_cmap_changed(self, cmap_name: str):
        """切换 colormap 时即时重绘时频图（不需要重新计算谱图）"""
        if self.last_spec is None:
            return
        absZ, t, f, vmin, vmax = self.last_spec
        self.spec_canvas.draw_content(absZ, t, f, vmin, vmax, cmap_name=cmap_name)

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            if self.player.source().isLocalFile():
                self.audio_out.setMuted(False); self.audio_out.setVolume(1.0)
                self.player.play(); self.timer.start()
            else:
                QtWidgets.QMessageBox.warning(self, "无音频", "请先载入片段。")

    def on_position_changed(self, pos_ms: int):
        # Qt 音频时间（压缩后） → 原始时间（未加速）
        t_audio = pos_ms / 1000.0
        x_original = t_audio * self.current_speedup
        if self.seg_t is not None:
            xmin, xmax = float(self.seg_t[0]), float(self.seg_t[-1])
            x_original = min(max(x_original, xmin), xmax)
        for c in (self.spec_canvas, self.wave_canvas_filt, self.wave_canvas_raw):
            c.update_playhead(x_original)
        self.statusBar().showMessage(f"Audio {t_audio:.2f}s | Original {x_original:.2f}s")

    def on_state_changed(self, new_state):
        if new_state != QMediaPlayer.PlaybackState.PlayingState:
            self.timer.stop()
        if new_state == QMediaPlayer.PlaybackState.StoppedState and self.seg_t is not None:
            end_t = float(self.seg_t[-1])
            for c in (self.spec_canvas, self.wave_canvas_filt, self.wave_canvas_raw):
                c.update_playhead(end_t)

    def on_timer(self):
        self.on_position_changed(self.player.position())

    def on_seek_requested(self, x_original: float):
        if self.seg_t is None:
            return
        xmin, xmax = float(self.seg_t[0]), float(self.seg_t[-1])
        x_original = min(max(x_original, xmin), xmax)
        t_audio_sec = x_original / max(self.current_speedup, 1e-9)
        self.player.setPosition(int(t_audio_sec * 1000.0))
        for c in (self.spec_canvas, self.wave_canvas_filt, self.wave_canvas_raw):
            c.update_playhead(x_original)
        if self.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            self.audio_out.setMuted(False); self.audio_out.setVolume(1.0)
            self.player.play(); self.timer.start()

    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            if self.wav_path and Path(self.wav_path).exists():
                os.remove(self.wav_path)
        except Exception:
            pass
        return super().closeEvent(event)


# ===================== 入口 =====================
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()