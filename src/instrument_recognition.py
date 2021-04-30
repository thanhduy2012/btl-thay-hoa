import numpy as np
import librosa
from scipy.fftpack import fft
import pickle


class Feature_Audio:
    def __init__(self):
        self.hop_length = 512
        self.frame_size = 1024
        pass

    def __pad_0__(self, signal):
        if signal.shape[0] < 3 * 44100:
            number_pad = 3 * 44100 - signal.shape[0]
            pad = np.zeros(number_pad)
            return np.concatenate([signal, pad])
        else:
            return signal[:3 * 44100]

    def __amplitude_envelope__(self, signal, frame_size, hop_length):
        amplitude_envelope = []
        for i in range(0, len(signal), hop_length):
            amplitude_envelope_current_frame = max(signal[i:i + frame_size])
            amplitude_envelope.append(amplitude_envelope_current_frame)
        return np.array(amplitude_envelope)

    def __custom_fft__(self, signal, fs):
        signal = self.__pad_0__(signal)
        N = signal.shape[0]
        N = min(fs // 2, N // 2)
        yf = fft(signal)
        vals = np.abs(yf[0:N])
        return vals

    def __get_mean_AE__(self, signal):
        res = self.__amplitude_envelope__(signal, self.frame_size, self.hop_length)
        return [res.mean()]

    def __get_zero_crossings__(self, signal):
        zero_crossings = librosa.zero_crossings(signal, pad=False)
        return [sum(zero_crossings)]

    def __filter_frequenly__(self, audio, sr):
        vals = self.__custom_fft__(audio, sr)
        res = []
        res.append(vals[20:22].mean())
        res.append(vals[22:25].mean())
        res.append(vals[25:30].mean())
        res.append(vals[30:40].mean())
        res.append(vals[40:50].mean())
        res.append(vals[50:60].mean())
        res.append(vals[70:80].mean())
        res.append(vals[80:90].mean())
        res.append(vals[100:120].mean())
        res.append(vals[120:140].mean())
        res.append(vals[140:160].mean())
        res.append(vals[160:180].mean())
        res.append(vals[180:200].mean())
        res.append(vals[200:250].mean())
        res.append(vals[250:300].mean())
        res.append(vals[300:350].mean())
        res.append(vals[350:400].mean())
        res.append(vals[400:450].mean())
        res.append(vals[450:500].mean())
        res.append(vals[500:600].mean())
        res.append(vals[600:700].mean())
        res.append(vals[700:800].mean())
        res.append(vals[800:1000].mean())
        res.append(vals[1000:1200].mean())
        res.append(vals[1200:2000].mean())
        res.append(vals[2000:3000].mean())
        res.append(vals[3000:4000].mean())
        res.append(vals[4000:7000].mean())
        res.append(vals[7000:15000].mean())
        res.append(vals[15000:].mean())
        return res

    def extract_feature(self, signal, sr):
        tmp = []
        tmp += self.__get_zero_crossings__(signal)
        tmp += self.__get_mean_AE__(signal)
        tmp += self.__filter_frequenly__(signal, sr)
        return tmp


class Forcast:
    def __init__(self, model_path="./src/model/random_forest_model.sav",
                 labelencoder_path="./src/model/labelencoder.sav"):
        self.model = pickle.load(open(model_path, 'rb'))
        self.labelencoder = pickle.load(open(labelencoder_path, 'rb'))
        pass

    def predict(self, x):
        x = np.array(x)
        if len(x.shape) == 1:
            x = np.array([x])
        res = self.model.predict(x)
        return self.labelencoder.inverse_transform(res)[0]
