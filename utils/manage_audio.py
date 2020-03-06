from __future__ import print_function
import librosa
import numpy as np
import pcen


class AudioPreprocessor(object):
    def __init__(self, sr=16000, n_dct_filters=40,  n_mels=40, f_max=4000, f_min=20, n_fft=480, hop_ms=10, config=None):
        super().__init__()
        self.config = config
        self.n_mels = n_mels  # 40
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2  # 4000
        self.f_min = f_min  # 20
        self.n_fft = n_fft  # duan shi fu li ye 480
        self.hop_length = sr // 1000 * hop_ms
        self.pcen_transform = pcen.StreamingPCENTransform(n_mels=n_mels, n_fft=n_fft, hop_length=self.hop_length, trainable=True)

    def compute_mfccs(self, data):
        mfcc = librosa.feature.mfcc(
            data,
            sr=self.sr,
            n_mfcc=self.n_mels,
            hop_length=self.hop_length
        )
        mfcc = np.array(mfcc, order="F").astype(np.float32)

        if self.config["feature_type"] == "log_mel":
            mel_spec = librosa.feature.melspectrogram(
                data,
                sr=self.sr,  # 16000
                n_mels=self.n_mels,  # 40
                hop_length=self.hop_length,  # 160
                n_fft=self.n_fft,  # 480
                fmin=self.f_min,  # 20
                fmax=self.f_max  # 4000
            )
            # data[data > 0] = np.log(data[data > 0])
            # data = [np.matmul(self.dct_filters, x) for x in np.split(data, data.shape[1], axis=1)]
            mel_spec = np.array(mel_spec, order="F").astype(np.float32)

            log_mel = librosa.power_to_db(mel_spec)
            delta = librosa.feature.delta(mfcc)
            delta_delta = librosa.feature.delta(delta)
            data = np.vstack([log_mel, delta, delta_delta])  # (120, 101)
            return data  # shape(120, 101, 1)
        elif self.config["feature_type"] == "MFCC":
            # print(mfcc.shape)
            return mfcc  # data shape(40，101)
        elif self.config["feature_type"] == "PCEN":
            spec = librosa.feature.melspectrogram(data, self.sr, power=1, n_mels=self.n_mels, hop_length=self.hop_length, n_fft=self.n_fft)
            pcen = librosa.pcen(spec, self.sr)  # (40,101)
            pcen = np.array(pcen, order="F").astype(np.float32)
            return pcen

    def compute_pcen(self, data):
        data = self.pcen_transform(data)
        self.pcen_transform.reset()
        return data
