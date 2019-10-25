import librosa
import numpy as np
wav_path = 'music/wav/BMGPM_XRD_Tune_Town.wav'
sr = 22050
hop_length = int(0.0125 * sr)
win_length = int(0.05 * sr)
x, _ = librosa.load(wav_path, sr=sr)
x, _ = librosa.effects.trim(x)
y = librosa.stft(x, n_fft=2048, hop_length=hop_length, win_length=win_length)
y = np.abs(y)
print(y)
print(y.shape)
