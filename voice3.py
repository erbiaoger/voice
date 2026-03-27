import sys
import os
import tempfile
from pathlib import Path

import numpy as np
from obspy import read
from scipy.signal import resample
from dasQt.process.freqattributes import spectrogram as sp_spectrogram
from scipy.io import wavfile

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer, QMediaDevices
from PyQt6.QtGui import QAction

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def sac_to_audio_wav(data, fs_orig, audio_rate=16000, gain=50.0, speed_up=20.0):
    """将地震段写成临时 WAV（Qt 可播），返回 wav_path 与压缩后时长（秒）"""
    data = data.astype(np.float64) * gain
    fs_target = fs_orig * speed_up
    n_samples = int(len(data) * audio_rate / fs_target)
    n_samples = max(n_samples, 2)
    audio = resample(data, n_samples)
    max_val = np.max(np.abs(audio)) if audio.size else 1.0
    if max_val > 0:
        audio = audio / max_val
    audio_int16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    fd, wav_path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    wavfile.write(wav_path, audio_rate, audio_int16)
    return wav_path, len(audio_int16) / float(audio_rate)


def compute_spectrogram(data, fs, nperseg=256, noverlap=128, end_freq=999, cmap_scale=0.1):
    """用你项目的 spectrogram 函数（返回 Zxx, f, t）生成归一化谱图"""
    Zxx, f, t = sp_spectrogram(
        data, fs=fs, nperseg=nperseg, noverlap=noverlap,
        nfft=None, detrend=False, boundary='zeros'
    )
    absZ = np.abs(Zxx)
    m = np.max(absZ) if absZ.size else 1.0
    if m > 0:
        absZ = absZ / m
    end_idx = np.argmin(np.abs(f - end_freq))
    absZ = absZ[:end_idx, :]
    f = f[:end_idx]
    vmin, vmax = 0.01 * cmap_scale, cmap_scale
    return absZ, t, f, vmin, vmax


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
            try: self.playline.remove()
            except Exception: pass
        # 红色虚线游标
        self.playline = self.ax.axvline(x0, color='red', lw=1.8, ls='--')
        self.draw_idle()

    def update_playhead(self, x):
        if self.playline is None:
            return
        self.playline.set_xdata([x, x])
        # 局部重绘更流畅
        self.ax.draw_artist(self.ax.patch)
        for a in self._artists_to_redraw:
            self.ax.draw_artist(a)
        self.ax.draw_artist(self.playline)
        self.figure.canvas.update()
        self.figure.canvas.flush_events()

    # === 关键修正：Qt 像素坐标 → Axes 数据坐标（仅 x） ===
    def _mouse_x_to_data(self, qevent: QtGui.QMouseEvent):
        # 1) 取 Qt 鼠标位置（部件坐标，左上为原点）
        px_qt = qevent.position().x()
        py_qt = qevent.position().y()

        # 2) 适配 HiDPI：乘以设备像素比
        dpr = float(self.devicePixelRatioF())
        px = px_qt * dpr
        # Matplotlib 显示坐标系原点在画布左下，需要把 y 翻转
        fig_height = float(self.figure.bbox.height)
        py = fig_height - (py_qt * dpr)

        # 3) 取得 Axes 在画布上的像素包围盒
        bbox = self.ax.get_window_extent()
        x0, y0, x1, y1 = float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)

        # 4) 判断是否在 Axes 内
        if not (x0 <= px <= x1 and y0 <= py <= y1):
            return None

        # 5) 映射像素到数据（线性）
        x_min, x_max = self.ax.get_xbound()
        if x1 == x0:
            return None
        frac = (px - x0) / (x1 - x0)
        xdata = x_min + frac * (x_max - x_min)

        # Clamp 到当前 xlim
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

    def draw_content(self, absZ, t, f, vmin, vmax):
        self.ax.clear()
        self.im = self.ax.imshow(
            absZ, aspect='auto', origin='lower', cmap='jet',
            interpolation='bicubic', extent=[t[0], t[-1], f[0], f[-1]],
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
        super().__init__(figsize=(9, 2.6), dpi=100, parent=parent)
        self.line = None

    def draw_content(self, t_sig, y_sig):
        self.ax.clear()
        (self.line,) = self.ax.plot(t_sig, y_sig, lw=0.8)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude')
        self._xlim = (float(t_sig[0]), float(t_sig[-1]))
        ypad = 0.05 * (np.max(y_sig) - np.min(y_sig) + 1e-12)
        self.ax.set_ylim(np.min(y_sig) - ypad, np.max(y_sig) + ypad)
        self._artists_to_redraw = [self.line]
        self.init_playline(self._xlim[0])
        self.figure.tight_layout()
        self.draw()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAC 音频播放 + 时频图/波形同步光标 (PyQt6)")
        self.resize(1180, 860)

        # ===== 中心与布局 =====
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        vlayout = QtWidgets.QVBoxLayout(central)

        # ----- 控件区 -----
        ctrl = QtWidgets.QGridLayout()
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("选择 .sac 文件...")
        self.path_edit.setText(str(Path("/Users/zhangzhiyu/453015084.00000001.2025.07.28.09.15.21.000.z.sac")))
        browse_btn = QtWidgets.QPushButton("浏览"); browse_btn.clicked.connect(self.browse_sac)

        self.start_sec = QtWidgets.QDoubleSpinBox(); self.start_sec.setRange(0.0, 1e9)
        self.start_sec.setDecimals(1); self.start_sec.setSingleStep(1.0); self.start_sec.setValue(3*3600 + 2800)
        self.win_len = QtWidgets.QDoubleSpinBox(); self.win_len.setRange(1.0, 1e6)
        self.win_len.setDecimals(1); self.win_len.setValue(400.0)
        self.gain = QtWidgets.QDoubleSpinBox(); self.gain.setRange(0.1, 1e5)
        self.gain.setDecimals(1); self.gain.setValue(200.0)
        self.speed_up = QtWidgets.QDoubleSpinBox(); self.speed_up.setRange(1.0, 1e4)
        self.speed_up.setDecimals(1); self.speed_up.setValue(30.0)
        self.audio_rate = QtWidgets.QSpinBox(); self.audio_rate.setRange(8000, 192000); self.audio_rate.setValue(16000)
        self.nperseg = QtWidgets.QSpinBox(); self.nperseg.setRange(64, 8192); self.nperseg.setValue(256)
        self.noverlap = QtWidgets.QSpinBox(); self.noverlap.setRange(0, 8000); self.noverlap.setValue(128)
        self.end_freq = QtWidgets.QDoubleSpinBox(); self.end_freq.setRange(1.0, 1e6); self.end_freq.setDecimals(0); self.end_freq.setValue(999.0)
        self.cmap_scale = QtWidgets.QDoubleSpinBox(); self.cmap_scale.setRange(1e-4, 10.0); self.cmap_scale.setDecimals(3); self.cmap_scale.setValue(0.1)

        self.load_btn = QtWidgets.QPushButton("载入并绘制"); self.load_btn.clicked.connect(self.load_and_plot)
        self.play_btn = QtWidgets.QPushButton("播放 / 暂停"); self.play_btn.setEnabled(False); self.play_btn.clicked.connect(self.toggle_play)

        self.dev_combo = QtWidgets.QComboBox(); self.dev_combo.setMinimumWidth(260)
        for dev in QMediaDevices.audioOutputs():
            self.dev_combo.addItem(dev.description(), dev)
        self.dev_combo.currentIndexChanged.connect(self._on_device_changed)

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
        ctrl.addWidget(QtWidgets.QLabel("色标尺度:"), r, 0); ctrl.addWidget(self.cmap_scale, r, 1)
        ctrl.addWidget(QtWidgets.QLabel("输出设备:"), r, 2); ctrl.addWidget(self.dev_combo, r, 3, 1, 3)
        ctrl.addWidget(self.load_btn, r, 6); ctrl.addWidget(self.play_btn, r, 7)
        vlayout.addLayout(ctrl)

        # ----- 画布区：上时频，下波形 -----
        self.spec_canvas = SpectrogramCanvas(self)
        self.wave_canvas = WaveformCanvas(self)
        self.spec_canvas.seekRequested.connect(self.on_seek_requested)
        self.wave_canvas.seekRequested.connect(self.on_seek_requested)
        vlayout.addWidget(self.spec_canvas, stretch=2)
        vlayout.addWidget(self.wave_canvas, stretch=1)

        # ----- 多媒体 -----
        self.player = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        default_dev = QMediaDevices.defaultAudioOutput()
        self.audio_out.setDevice(default_dev); self.audio_out.setMuted(False); self.audio_out.setVolume(1.0)
        self.player.setAudioOutput(self.audio_out)

        # 定时器用于更平滑的光标更新
        self.timer = QtCore.QTimer(self); self.timer.setInterval(30); self.timer.timeout.connect(self.on_timer)

        # 信号
        self.player.positionChanged.connect(self.on_position_changed)
        self.player.playbackStateChanged.connect(self.on_state_changed)
        self.player.mediaStatusChanged.connect(self._on_media_status)
        self.player.errorOccurred.connect(self._on_media_error)

        # 内部状态
        self.fs = None
        self.seg_t = None      # 时频横轴（秒）
        self.t_sig = None      # 波形横轴（秒）
        self.seg = None        # 当前片段（波形）
        self.wav_path = None
        self.wav_duration = 0.0
        self.current_speedup = 1.0

        # 菜单
        exit_act = QAction("退出", self); exit_act.triggered.connect(self.close)
        menubar = self.menuBar(); file_menu = menubar.addMenu("文件"); file_menu.addAction(exit_act)
        self.statusBar().showMessage("就绪")

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
        try: err_str = self.player.errorString()
        except Exception: pass
        self.statusBar().showMessage(f"[mediaError] code={err} {err_str}")

    def browse_sac(self):
        pth, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 SAC 文件", str(Path.home()), "SAC Files (*.sac);;All Files (*)")
        if pth:
            self.path_edit.setText(pth)

    def load_and_plot(self):
        try:
            sac_path = Path(self.path_edit.text()).expanduser()
            if not sac_path.exists():
                QtWidgets.QMessageBox.warning(self, "错误", f"找不到文件：\n{sac_path}"); return

            st = read(str(sac_path)); tr = st[0]
            self.fs = float(tr.stats.sampling_rate)
            data = tr.data.astype(np.float64)

            start = float(self.start_sec.value())
            length = float(self.win_len.value())
            gain = float(self.gain.value())
            speed_up = float(self.speed_up.value())
            audio_rate = int(self.audio_rate.value())
            nperseg = int(self.nperseg.value())
            noverlap = int(self.noverlap.value())
            end_freq = float(self.end_freq.value())
            cmap_scale = float(self.cmap_scale.value())

            i0 = max(int(start * self.fs), 0)
            i1 = min(int((start + length) * self.fs), len(data))
            if i1 - i0 < max(2, nperseg):
                QtWidgets.QMessageBox.warning(self, "错误", "选择窗口过短，无法计算谱图。"); return
            seg = data[i0:i1]; self.seg = seg

            absZ, t, f, vmin, vmax = compute_spectrogram(
                seg, self.fs, nperseg=nperseg, noverlap=noverlap,
                end_freq=end_freq, cmap_scale=cmap_scale
            )
            self.spec_canvas.draw_content(absZ, t, f, vmin, vmax)
            self.seg_t = t

            self.t_sig = np.arange(len(seg), dtype=np.float64) / self.fs
            self.wave_canvas.draw_content(self.t_sig, seg)

            self.current_speedup = speed_up

            if self.wav_path and Path(self.wav_path).exists():
                try: os.remove(self.wav_path)
                except Exception: pass
            self.wav_path, self.wav_duration = sac_to_audio_wav(
                seg, self.fs, audio_rate=audio_rate, gain=gain, speed_up=speed_up
            )
            try:
                rate_check, data_check = wavfile.read(self.wav_path)
                print("[WAV] readback:", rate_check, data_check.dtype, data_check.shape)
            except Exception as e:
                print("[WAV] readback error:", e)

            from PyQt6.QtCore import QUrl
            self.player.setSource(QUrl.fromLocalFile(self.wav_path))
            self.play_btn.setEnabled(True)
            self.statusBar().showMessage(f"片段已载入，压缩后音频时长 {self.wav_duration:.2f}s。")
            self.spec_canvas.update_playhead(0.0); self.wave_canvas.update_playhead(0.0)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "载入错误", str(e)); raise

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
        t_audio = pos_ms / 1000.0
        x_original = t_audio * self.current_speedup
        if self.seg_t is not None:
            xmin, xmax = float(self.seg_t[0]), float(self.seg_t[-1])
            x_original = min(max(x_original, xmin), xmax)
        self.spec_canvas.update_playhead(x_original)
        self.wave_canvas.update_playhead(x_original)
        self.statusBar().showMessage(f"Audio {t_audio:.2f}s | Original {x_original:.2f}s")

    def on_state_changed(self, new_state):
        if new_state != QMediaPlayer.PlaybackState.PlayingState:
            self.timer.stop()
        if new_state == QMediaPlayer.PlaybackState.StoppedState and self.seg_t is not None:
            end_t = float(self.seg_t[-1])
            self.spec_canvas.update_playhead(end_t)
            self.wave_canvas.update_playhead(end_t)

    def on_timer(self):
        self.on_position_changed(self.player.position())

    def on_seek_requested(self, x_original: float):
        if self.seg_t is None:
            return
        xmin, xmax = float(self.seg_t[0]), float(self.seg_t[-1])
        x_original = min(max(x_original, xmin), xmax)
        t_audio_sec = x_original / max(self.current_speedup, 1e-9)
        self.player.setPosition(int(t_audio_sec * 1000.0))
        self.spec_canvas.update_playhead(x_original)
        self.wave_canvas.update_playhead(x_original)
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


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()