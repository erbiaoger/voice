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
from PyQt6.QtCore import Qt
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer, QMediaDevices
from PyQt6.QtGui import QAction

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def sac_to_audio_wav(data, fs_orig, audio_rate=16000, gain=50.0, speed_up=20.0):
    """
    把地震段数据写成临时 WAV（Qt 可播放）
    返回: wav_path (str), audio_duration_sec (float)
    """
    data = data.astype(np.float64) * gain

    # 时间压缩等效为提高原采样率
    fs_target = fs_orig * speed_up
    n_samples = int(len(data) * audio_rate / fs_target)
    if n_samples <= 1:
        n_samples = 2

    audio = resample(data, n_samples)

    # 归一化到 int16
    max_val = np.max(np.abs(audio)) if audio.size else 1.0
    if max_val > 0:
        audio = audio / max_val
    audio_int16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)

    # 临时文件
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    wavfile.write(wav_path, audio_rate, audio_int16)

    audio_duration_sec = len(audio_int16) / float(audio_rate)
    return wav_path, audio_duration_sec


def compute_spectrogram(data, fs, nperseg=256, noverlap=128, end_freq=999, cmap_scale=0.1):
    """
    调用你项目里的 spectrogram（返回 Zxx, f, t）
    输出与原笔记本一致：absZ, t, f, vmin, vmax
    """
    Zxx, f, t = sp_spectrogram(
        data, fs=fs, nperseg=nperseg, noverlap=noverlap,
        nfft=None, detrend=False, boundary='zeros'
    )
    print("compute_spectrogram -> f.shape, t.shape, Zxx.shape:", f.shape, t.shape, Zxx.shape)

    absZ = np.abs(Zxx)
    m = np.max(absZ) if absZ.size else 1.0
    if m > 0:
        absZ = absZ / m

    # 频率截断
    end_idx = np.argmin(np.abs(f - end_freq))
    absZ = absZ[:end_idx, :]
    f = f[:end_idx]

    vmin = 0.01 * cmap_scale
    vmax = cmap_scale
    return absZ, t, f, vmin, vmax


class SpectrogramCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(9, 4), dpi=100)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.im = None
        self.playline = None

    def plot_spectrogram(self, absZ, t, f, vmin, vmax):
        self.ax.clear()
        im = self.ax.imshow(
            absZ, aspect='auto', origin='lower', cmap='jet',
            interpolation='bicubic',
            extent=[t[0], t[-1], f[0], f[-1]],
            vmin=vmin, vmax=vmax
        )
        self.im = im
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Frequency (Hz)')
        # cbar = self.figure.colorbar(im, ax=self.ax, pad=0.04, extend='both')
        # from matplotlib.ticker import ScalarFormatter
        # formatter = ScalarFormatter(useMathText=True)
        # formatter.set_powerlimits((-2, 2))
        # cbar.ax.yaxis.set_major_formatter(formatter)
        # 播放光标
        self.playline = self.ax.axvline(0.0, color='r', lw=1.5, ls='--')
        self.figure.tight_layout()
        self.draw()

    def update_playhead(self, x):
        if self.playline is None:
            return
        self.playline.set_xdata([x, x])
        # 局部重绘更流畅
        self.ax.draw_artist(self.ax.patch)
        if self.im:
            self.ax.draw_artist(self.im)
        self.ax.draw_artist(self.playline)
        self.figure.canvas.update()
        self.figure.canvas.flush_events()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAC 音频播放 + 时频图同步光标 (PyQt6)")
        self.resize(1180, 720)

        # 中心区
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # 控件区
        ctrl_layout = QtWidgets.QGridLayout()

        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("选择 .sac 文件...")
        # 预填你给的路径，按需修改
        self.path_edit.setText(str(Path("/Users/zhangzhiyu/453015084.00000001.2025.07.28.09.15.21.000.z.sac")))

        browse_btn = QtWidgets.QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_sac)

        self.start_sec = QtWidgets.QDoubleSpinBox()
        self.start_sec.setRange(0.0, 1e9)
        self.start_sec.setDecimals(1)
        self.start_sec.setSingleStep(1.0)
        self.start_sec.setValue(3*3600 + 2800)

        self.win_len = QtWidgets.QDoubleSpinBox()
        self.win_len.setRange(1.0, 1e6)
        self.win_len.setDecimals(1)
        self.win_len.setValue(400.0)

        self.gain = QtWidgets.QDoubleSpinBox()
        self.gain.setRange(0.1, 1e5)
        self.gain.setDecimals(1)
        self.gain.setValue(200.0)

        self.speed_up = QtWidgets.QDoubleSpinBox()
        self.speed_up.setRange(1.0, 1e4)
        self.speed_up.setDecimals(1)
        self.speed_up.setValue(30.0)

        self.audio_rate = QtWidgets.QSpinBox()
        self.audio_rate.setRange(8000, 192000)
        self.audio_rate.setValue(16000)

        self.nperseg = QtWidgets.QSpinBox()
        self.nperseg.setRange(64, 8192)
        self.nperseg.setValue(256)

        self.noverlap = QtWidgets.QSpinBox()
        self.noverlap.setRange(0, 8000)
        self.noverlap.setValue(128)

        self.end_freq = QtWidgets.QDoubleSpinBox()
        self.end_freq.setRange(1.0, 1e6)
        self.end_freq.setDecimals(0)
        self.end_freq.setValue(999.0)

        self.cmap_scale = QtWidgets.QDoubleSpinBox()
        self.cmap_scale.setRange(1e-4, 10.0)
        self.cmap_scale.setDecimals(3)
        self.cmap_scale.setValue(0.1)

        self.load_btn = QtWidgets.QPushButton("载入并绘制")
        self.load_btn.clicked.connect(self.load_and_plot)

        self.play_btn = QtWidgets.QPushButton("播放 / 暂停")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.toggle_play)

        self.status_lab = QtWidgets.QLabel("Ready")

        # 设备下拉框（显式选择输出设备）
        self.dev_combo = QtWidgets.QComboBox()
        self.dev_combo.setMinimumWidth(260)
        for dev in QMediaDevices.audioOutputs():
            self.dev_combo.addItem(dev.description(), dev)
        self.dev_combo.currentIndexChanged.connect(self._on_device_changed)

        row = 0
        ctrl_layout.addWidget(QtWidgets.QLabel("SAC 文件:"), row, 0)
        ctrl_layout.addWidget(self.path_edit, row, 1, 1, 6)
        ctrl_layout.addWidget(browse_btn, row, 7)

        row += 1
        ctrl_layout.addWidget(QtWidgets.QLabel("起始(s):"), row, 0)
        ctrl_layout.addWidget(self.start_sec, row, 1)
        ctrl_layout.addWidget(QtWidgets.QLabel("时长(s):"), row, 2)
        ctrl_layout.addWidget(self.win_len, row, 3)
        ctrl_layout.addWidget(QtWidgets.QLabel("增益:"), row, 4)
        ctrl_layout.addWidget(self.gain, row, 5)
        ctrl_layout.addWidget(QtWidgets.QLabel("加速倍数:"), row, 6)
        ctrl_layout.addWidget(self.speed_up, row, 7)

        row += 1
        ctrl_layout.addWidget(QtWidgets.QLabel("音频采样率:"), row, 0)
        ctrl_layout.addWidget(self.audio_rate, row, 1)
        ctrl_layout.addWidget(QtWidgets.QLabel("nperseg:"), row, 2)
        ctrl_layout.addWidget(self.nperseg, row, 3)
        ctrl_layout.addWidget(QtWidgets.QLabel("noverlap:"), row, 4)
        ctrl_layout.addWidget(self.noverlap, row, 5)
        ctrl_layout.addWidget(QtWidgets.QLabel("最大频率(Hz):"), row, 6)
        ctrl_layout.addWidget(self.end_freq, row, 7)

        row += 1
        ctrl_layout.addWidget(QtWidgets.QLabel("色标尺度:"), row, 0)
        ctrl_layout.addWidget(self.cmap_scale, row, 1)
        ctrl_layout.addWidget(QtWidgets.QLabel("输出设备:"), row, 2)
        ctrl_layout.addWidget(self.dev_combo, row, 3, 1, 3)
        ctrl_layout.addWidget(self.load_btn, row, 6)
        ctrl_layout.addWidget(self.play_btn, row, 7)

        layout.addLayout(ctrl_layout)

        # 画布
        self.canvas = SpectrogramCanvas(self)
        layout.addWidget(self.canvas, stretch=1)

        # 多媒体（关键：设备/音量/日志）
        self.player = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)

        # 显式设置默认输出设备 + 解除静音 + 音量 100%
        default_dev = QMediaDevices.defaultAudioOutput()
        self.audio_out.setDevice(default_dev)
        self.audio_out.setMuted(False)
        self.audio_out.setVolume(1.0)  # 0..1

        self.player.setAudioOutput(self.audio_out)

        # 调试：列出设备
        dev_names = [d.description() for d in QMediaDevices.audioOutputs()]
        print("[AudioOutputs]", dev_names)
        print("[DefaultOutput]", default_dev.description())

        # 定时器：让光标更平滑
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(30)  # ~33 FPS
        self.timer.timeout.connect(self.on_timer)

        # 连接信号（状态/错误/位置）
        self.player.positionChanged.connect(self.on_position_changed)
        self.player.playbackStateChanged.connect(self.on_state_changed)
        self.player.mediaStatusChanged.connect(self._on_media_status)
        self.player.errorOccurred.connect(self._on_media_error)

        # 内部状态
        self.fs = None
        self.seg_t = None
        self.wav_path = None
        self.wav_duration = 0.0
        self.current_speedup = 1.0

        # 菜单
        exit_act = QAction("退出", self)
        exit_act.triggered.connect(self.close)
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")
        file_menu.addAction(exit_act)

        # 状态栏
        self.statusBar().showMessage("就绪")

    # 设备切换
    def _on_device_changed(self, idx):
        dev = self.dev_combo.itemData(idx)
        if dev:
            self.audio_out.setDevice(dev)
            print("[SwitchOutputDevice]", dev.description())
            self.statusBar().showMessage(f"切换输出设备：{dev.description()}")

    # 媒体状态日志
    def _on_media_status(self, status):
        self.statusBar().showMessage(f"[mediaStatus] {status}")
        print("[mediaStatus]", status)

    # 错误日志
    def _on_media_error(self, err, *args):
        try:
            err_str = self.player.errorString()
        except Exception:
            err_str = ""
        msg = f"[mediaError] code={err} {err_str}"
        self.statusBar().showMessage(msg)
        print(msg)

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

            st = read(str(sac_path))
            tr = st[0]
            self.fs = float(tr.stats.sampling_rate)
            data = tr.data.astype(np.float64)
            print("原始数据 shape:", data.shape, "fs:", self.fs)

            start      = float(self.start_sec.value())
            length     = float(self.win_len.value())
            gain       = float(self.gain.value())
            speed_up   = float(self.speed_up.value())
            audio_rate = int(self.audio_rate.value())
            nperseg    = int(self.nperseg.value())
            noverlap   = int(self.noverlap.value())
            end_freq   = float(self.end_freq.value())
            cmap_scale = float(self.cmap_scale.value())

            # 切片（单位：秒）
            i0 = int(start * self.fs)
            i1 = int((start + length) * self.fs)
            i0 = max(i0, 0)
            i1 = min(i1, len(data))
            if i1 - i0 < max(2, nperseg):
                QtWidgets.QMessageBox.warning(self, "错误", "选择窗口过短，无法计算谱图。")
                return
            seg = data[i0:i1]
            print("片段 shape:", seg.shape)

            # 频谱图（横轴为原始时间秒）
            absZ, t, f, vmin, vmax = compute_spectrogram(
                seg, self.fs, nperseg=nperseg, noverlap=noverlap,
                end_freq=end_freq, cmap_scale=cmap_scale
            )
            f_min = 50.

            ind_fmin = np.argmin(np.abs(f - f_min))
            f = f[ind_fmin:]
            absZ = absZ[ind_fmin:, :]
            vmin = 0.01 * cmap_scale
            vmax = cmap_scale
            print("谱图 -> absZ, t, f:", absZ.shape, t.shape, f.shape, "vmin/vmax:", vmin, vmax)
            self.canvas.plot_spectrogram(absZ, t, f, vmin, vmax)
            self.seg_t = t
            self.current_speedup = speed_up

            # 生成 WAV
            if self.wav_path and Path(self.wav_path).exists():
                try:
                    os.remove(self.wav_path)
                except Exception:
                    pass
            self.wav_path, self.wav_duration = sac_to_audio_wav(
                seg, self.fs, audio_rate=audio_rate, gain=gain, speed_up=speed_up
            )
            print("[WAV] path:", self.wav_path, "duration(s):", self.wav_duration)

            # 读回检查
            try:
                rate_check, data_check = wavfile.read(self.wav_path)
                print("[WAV] readback:", rate_check, data_check.dtype, data_check.shape)
            except Exception as e:
                print("[WAV] readback error:", e)

            # 设置媒体源
            from PyQt6.QtCore import QUrl
            self.player.setSource(QUrl.fromLocalFile(self.wav_path))
            print("[SetSource]", self.wav_path, "| duration(ms) before load:", self.player.duration())

            self.play_btn.setEnabled(True)
            self.statusBar().showMessage(f"片段已载入，压缩后音频时长 {self.wav_duration:.2f}s。")
            self.canvas.update_playhead(0.0)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "载入错误", str(e))
            raise

    def toggle_play(self):
        state = self.player.playbackState()
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            if self.player.source().isLocalFile():
                # 确保不静音且音量足够
                self.audio_out.setMuted(False)
                self.audio_out.setVolume(1.0)
                self.player.play()
                self.timer.start()
            else:
                QtWidgets.QMessageBox.warning(self, "无音频", "请先载入片段。")

    def on_position_changed(self, pos_ms: int):
        # 音频时间 → 原始时间（秒）
        t_audio = pos_ms / 1000.0
        x_original = t_audio * self.current_speedup
        if self.seg_t is not None:
            xmin, xmax = float(self.seg_t[0]), float(self.seg_t[-1])
            x_original = min(max(x_original, xmin), xmax)
        self.canvas.update_playhead(x_original)
        self.statusBar().showMessage(f"Audio {t_audio:.2f}s | Original {x_original:.2f}s")

    def on_state_changed(self, new_state):
        if new_state != QMediaPlayer.PlaybackState.PlayingState:
            self.timer.stop()
        if new_state == QMediaPlayer.PlaybackState.StoppedState:
            if self.seg_t is not None:
                self.canvas.update_playhead(float(self.seg_t[-1]))

    def on_timer(self):
        # 平滑更新光标（补齐 positionChanged 的间隙）
        pos_ms = self.player.position()
        self.on_position_changed(pos_ms)

    def closeEvent(self, event: QtGui.QCloseEvent):
        # 清理临时 WAV
        try:
            if self.wav_path and Path(self.wav_path).exists():
                os.remove(self.wav_path)
        except Exception:
            pass
        return super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()