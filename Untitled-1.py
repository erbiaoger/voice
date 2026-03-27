# %%
from obspy import read
from scipy.signal import resample
import numpy as np
from IPython.display import Audio

def play_sac_audio(data, fs_orig, audio_rate=16000, gain=50.0, speed_up=20.0):
    """
    播放 SAC 地震信号为声音（强烈建议加速以变为可听范围）

    参数:
    - sac_file: SAC 文件路径
    - audio_rate: 最终音频采样率（建议 16000）
    - gain: 放大倍数（建议 30–100）
    - speed_up: 时间压缩倍数（将地震信号“加速”播放）
    """

    # 去直流，放大
    # data = (data - np.mean(data)) * gain
    data = data*gain

    # 时间压缩：加快 speed_up 倍，即原始数据看作加快后的采样
    fs_target = fs_orig * speed_up

    # 重新采样到音频采样率
    n_samples = int(len(data) * audio_rate / fs_target)
    data_audio = resample(data, n_samples)

    # 归一化，防止爆音
    max_val = np.max(np.abs(data_audio))
    if max_val > 0:
        data_audio = data_audio / max_val

    # 返回播放器
    return Audio(data_audio, rate=audio_rate), data



# %%
from pathlib import Path
from obspy import read
from dasQt.process.freqattributes import (
                spectrum, spectrogram, fk_transform)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dasQt.utils.imshow import setAxis

sac = Path('/Users/zhangzhiyu/453015084.00000001.2025.07.28.09.15.21.000.z.sac')

st = read(sac)
fs = st[0].stats.sampling_rate

# %%
hour = 5

# %%
## 1D Spectrogram
%matplotlib qt
Zxx, f, t = spectrogram(st[0].data[1000*60*60*hour:1000*60*60*hour+1000*3600], fs, nperseg=256,  
                noverlap=128, nfft=None, detrend=False,
                boundary='zeros')

Zxx = Zxx / np.max(np.abs(Zxx))

end_freq = 999
end_freq_idx = np.argmin(np.abs(f - end_freq))
Zxx = Zxx[:end_freq_idx, :]
f = f[:end_freq_idx]

scale = 0.1
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
im = ax.imshow(np.abs(Zxx), aspect='auto', origin='lower', cmap='jet',
               interpolation='bicubic',
           extent=[t[0], t[-1], f[0], f[-1]], vmin=0.01*scale, vmax=scale) # vmin=Zxx.min()*scale, vmax=Zxx.max()*scale
# 添加 colorbar
bar = fig.colorbar(im, ax=ax, pad=0.04, extend='both')

# 设置 colorbar 使用科学计数法
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-2, 2))  # 当数值超出 10^(-2) 到 10^2 之外时，使用科学计数法
bar.ax.yaxis.set_major_formatter(formatter)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency (Hz)')
# ax.set_title(f'Spectrogram of Trace {trace} (m)')
setAxis(ax)

fig.tight_layout()

del Zxx, f, t

# %%
# 使用方法
audio, data = play_sac_audio(st[0].data[1000*60*60*3+1000*2800:1000*60*60*3+1000*3200], fs, audio_rate=16000, gain=200.0, speed_up=30.0)
audio
