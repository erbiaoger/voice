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
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtGui import QAction

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def sac_to_audio_wav(data, fs_orig, audio_rate=16000, gain=50.0, speed_up=20.0):
    """
    Convert a seismic segment to a temp WAV file playable by Qt.
    - data: 1D numpy array (segment in original sampling rate)
    - fs_orig: original sampling rate (Hz)
    - audio_rate: output audio sampling rate (Hz)
    - gain: amplitude gain applied before resampling
    - speed_up: time compression factor (playback speed multiplier)
    Returns: wav_path (str), audio_duration_sec (float)
    """
    # Gain (keep your own DC handling policy — many seismo traces are zero-mean already)
    data = data.astype(np.float64) * gain

    # Effective target sampling before conversion: speed-up implies we "pretend" fs is fs_orig*speed_up
    fs_target = fs_orig * speed_up
    n_samples = int(len(data) * audio_rate / fs_target)
    if n_samples <= 1:
        n_samples = 2

    audio = resample(data, n_samples)

    # Normalize to avoid clipping then convert to int16
    max_val = np.max(np.abs(audio)) if audio.size else 1.0
    if max_val > 0:
        audio = audio / max_val
    audio_int16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)

    # Write to a temp WAV
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)  # We'll write via scipy.io.wavfile
    wavfile.write(wav_path, audio_rate, audio_int16)

    audio_duration_sec = len(audio_int16) / float(audio_rate)
    return wav_path, audio_duration_sec


def compute_spectrogram(data, fs, nperseg=256, noverlap=128, end_freq=999, cmap_scale=0.1):
    """
    Compute spectrogram like your notebook code and pack plotting kwargs.
    Returns: (abs_Z, t, f), vmin, vmax
    """
    Zxx, f, t = sp_spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap,
                               nfft=None, detrend=False, boundary='zeros')
    print(f.shape, t.shape, Zxx.shape)
    absZ = np.abs(Zxx)
    m = np.max(absZ) if absZ.size else 1.0
    if m > 0:
        absZ = absZ / m

    print(absZ.shape)
    # Crop to end_freq
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
        im = self.ax.imshow(absZ, aspect='auto', origin='lower', cmap='jet',
                            interpolation='bicubic',
                            extent=[t[0], t[-1], f[0], f[-1]],
                            vmin=vmin, vmax=vmax)
        self.im = im
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Frequency (Hz)')
        # cbar = self.figure.colorbar(im, ax=self.ax, pad=0.04, extend='both')
        # # Scientific notation on colorbar
        # from matplotlib.ticker import ScalarFormatter
        # formatter = ScalarFormatter(useMathText=True)
        # formatter.set_powerlimits((-2, 2))
        # cbar.ax.yaxis.set_major_formatter(formatter)
        # Init a vertical playhead line at t=0
        self.playline = self.ax.axvline(0.0, color='k', lw=1.5, ls='--')
        self.figure.tight_layout()
        self.draw()

    def update_playhead(self, x):
        if self.playline is None:
            return
        self.playline.set_xdata([x, x])
        # Efficient redraw of artists
        self.ax.draw_artist(self.ax.patch)
        if self.im:
            self.ax.draw_artist(self.im)
        self.ax.draw_artist(self.playline)
        self.figure.canvas.update()
        self.figure.canvas.flush_events()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAC Audio Player with Spectrogram Cursor (PyQt6)")
        self.resize(1100, 650)

        # --- Central Widget
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QVBoxLayout(central)

        # --- Controls
        ctrl_layout = QtWidgets.QGridLayout()

        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("Select a .sac file...")
        # Prefill with your example path; adjust as needed:
        self.path_edit.setText(str(Path("/Users/zhangzhiyu/453015084.00000001.2025.07.28.09.15.21.000.z.sac")))

        browse_btn = QtWidgets.QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_sac)

        self.start_sec = QtWidgets.QDoubleSpinBox()
        self.start_sec.setRange(0.0, 1e9)
        self.start_sec.setDecimals(1)
        self.start_sec.setSingleStep(1.0)
        self.start_sec.setValue(3*3600 + 2800)  # your example: hour=3, 2800s

        self.win_len = QtWidgets.QDoubleSpinBox()
        self.win_len.setRange(1.0, 1e6)
        self.win_len.setDecimals(1)
        self.win_len.setValue(400.0)  # ~ the range you used in the demo slice (3200-2800)

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

        self.load_btn = QtWidgets.QPushButton("Load & Plot")
        self.load_btn.clicked.connect(self.load_and_plot)

        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.toggle_play)

        self.status_lab = QtWidgets.QLabel("Ready")

        row = 0
        ctrl_layout.addWidget(QtWidgets.QLabel("SAC File:"), row, 0)
        ctrl_layout.addWidget(self.path_edit, row, 1, 1, 6)
        ctrl_layout.addWidget(browse_btn, row, 7)

        row += 1
        ctrl_layout.addWidget(QtWidgets.QLabel("Start (s):"), row, 0)
        ctrl_layout.addWidget(self.start_sec, row, 1)
        ctrl_layout.addWidget(QtWidgets.QLabel("Length (s):"), row, 2)
        ctrl_layout.addWidget(self.win_len, row, 3)
        ctrl_layout.addWidget(QtWidgets.QLabel("Gain:"), row, 4)
        ctrl_layout.addWidget(self.gain, row, 5)
        ctrl_layout.addWidget(QtWidgets.QLabel("Speed-up:"), row, 6)
        ctrl_layout.addWidget(self.speed_up, row, 7)

        row += 1
        ctrl_layout.addWidget(QtWidgets.QLabel("Audio Rate:"), row, 0)
        ctrl_layout.addWidget(self.audio_rate, row, 1)
        ctrl_layout.addWidget(QtWidgets.QLabel("nperseg:"), row, 2)
        ctrl_layout.addWidget(self.nperseg, row, 3)
        ctrl_layout.addWidget(QtWidgets.QLabel("noverlap:"), row, 4)
        ctrl_layout.addWidget(self.noverlap, row, 5)
        ctrl_layout.addWidget(QtWidgets.QLabel("Max Freq (Hz):"), row, 6)
        ctrl_layout.addWidget(self.end_freq, row, 7)

        row += 1
        ctrl_layout.addWidget(QtWidgets.QLabel("Cmap scale:"), row, 0)
        ctrl_layout.addWidget(self.cmap_scale, row, 1)
        ctrl_layout.addWidget(self.load_btn, row, 6)
        ctrl_layout.addWidget(self.play_btn, row, 7)

        layout.addLayout(ctrl_layout)

        # --- Canvas
        self.canvas = SpectrogramCanvas(self)
        layout.addWidget(self.canvas, stretch=1)

        # --- Player
        self.player = QMediaPlayer()
        self.audio_out = QAudioOutput()
        self.player.setAudioOutput(self.audio_out)
        self.audio_out.setVolume(0.9)  # 0..1

        # Update cursor via timer (smoother than relying only on signals)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(30)  # ~33 FPS
        self.timer.timeout.connect(self.on_timer)

        # Also react to Qt signals
        self.player.positionChanged.connect(self.on_position_changed)
        self.player.playbackStateChanged.connect(self.on_state_changed)

        # Internal state
        self.fs = None
        self.seg_t = None  # spectrogram t-axis (seconds)
        self.wav_path = None
        self.wav_duration = 0.0
        self.current_speedup = 1.0

        # Menu: quit
        exit_act = QAction("Quit", self)
        exit_act.triggered.connect(self.close)
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(exit_act)

    # --- UI Handlers -------------------------------------------------
    def browse_sac(self):
        pth, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select SAC file", str(Path.home()), "SAC Files (*.sac);;All Files (*)"
        )
        if pth:
            self.path_edit.setText(pth)

    def load_and_plot(self):
        try:
            sac_path = Path(self.path_edit.text()).expanduser()
            if not sac_path.exists():
                QtWidgets.QMessageBox.warning(self, "Error", f"File not found:\n{sac_path}")
                return

            st = read(str(sac_path))
            tr = st[0]
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
            cmap_scale = float(self.cmap_scale.value())
            print(data.shape)

            # Slice the original data by seconds
            i0 = int(start * self.fs)
            i1 = int((start + length) * self.fs)
            if i0 < 0: i0 = 0
            if i1 > len(data): i1 = len(data)
            if i1 - i0 < max(2, nperseg):
                QtWidgets.QMessageBox.warning(self, "Error", "Selected window too short for spectrogram.")
                return
            seg = data[i0:i1]
            print(seg.shape)

            # Compute spectrogram on ORIGINAL segment (so the x-axis is original seconds)
            absZ, t, f, vmin, vmax = compute_spectrogram(seg, self.fs, nperseg=nperseg,
                                                         noverlap=noverlap, end_freq=end_freq,
                                                         cmap_scale=cmap_scale)
            print(absZ.shape, t.shape, f.shape, vmin, vmax)
            self.canvas.plot_spectrogram(absZ, t, f, vmin, vmax)
            self.seg_t = t
            self.current_speedup = speed_up

            # Prepare WAV for the SAME segment and parameters
            if self.wav_path and Path(self.wav_path).exists():
                try: os.remove(self.wav_path)
                except Exception: pass
            self.wav_path, self.wav_duration = sac_to_audio_wav(
                seg, self.fs, audio_rate=audio_rate, gain=gain, speed_up=speed_up
            )

            # Load into player
            from PyQt6.QtCore import QUrl
            self.player.setSource(QUrl.fromLocalFile(self.wav_path))
            self.play_btn.setEnabled(True)
            self.status_lab.setText(f"Loaded. Audio duration: {self.wav_duration:.2f}s (time-compressed).")
            self.statusBar().showMessage("Loaded segment and spectrogram.")

            # Reset playhead
            self.canvas.update_playhead(0.0)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))

    def toggle_play(self):
        state = self.player.playbackState()
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            # Ensure we are at a valid source
            if self.player.source().isLocalFile():
                self.player.play()
                self.timer.start()
            else:
                QtWidgets.QMessageBox.warning(self, "No audio", "Load a segment first.")

    def on_position_changed(self, pos_ms: int):
        # pos_ms is audio time; convert to ORIGINAL time via speed_up
        t_audio = pos_ms / 1000.0
        x_original = t_audio * self.current_speedup
        # Clamp to spectrogram range
        if self.seg_t is not None:
            xmin, xmax = float(self.seg_t[0]), float(self.seg_t[-1])
            if x_original < xmin: x_original = xmin
            if x_original > xmax: x_original = xmax
        self.canvas.update_playhead(x_original)
        self.statusBar().showMessage(f"Audio {t_audio:.2f}s  |  Original time {x_original:.2f}s")

    def on_state_changed(self, new_state):
        if new_state != QMediaPlayer.PlaybackState.PlayingState:
            self.timer.stop()
        if new_state == QMediaPlayer.PlaybackState.StoppedState:
            # ensure playhead at end
            if self.seg_t is not None:
                self.canvas.update_playhead(float(self.seg_t[-1]))

    def on_timer(self):
        # Smooth updates between positionChanged emissions
        pos_ms = self.player.position()
        self.on_position_changed(pos_ms)

    def closeEvent(self, event: QtGui.QCloseEvent):
        # Clean temp file
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