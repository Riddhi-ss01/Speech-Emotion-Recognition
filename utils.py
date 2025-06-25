# utils.py
import numpy as np
import librosa

# Feature Extraction Function
def extract_deep_features(file_path, duration=3, sr=22050, max_len=130):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        desired_length = duration * sr
        if len(y) < desired_length:
            y = np.pad(y, (0, desired_length - len(y)))
        else:
            y = y[:desired_length]

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)

        all_feat = np.vstack([mfcc, delta, delta2, chroma, mel, contrast, tonnetz, zcr, rms]).T

        if all_feat.shape[0] < max_len:
            pad_width = max_len - all_feat.shape[0]
            all_feat = np.pad(all_feat, ((0, pad_width), (0, 0)), mode='constant')
        else:
            all_feat = all_feat[:max_len, :]

        return np.expand_dims(all_feat, axis=0)  # Model expects batch dimension

    except Exception as e:
        print(f"Error in {file_path}: {e}")
        return None
