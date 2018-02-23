import random
import numpy as np
from scipy import signal

### Transformations
class Fork:
    def __init__(self, transform_dict):
        self.transform_dict = transform_dict

    def __call__(self, data):
        result = {}
        for fork_name, transformations in self.transform_dict.items():
            fork_data = data
            for trans in transformations:
                fork_data = trans(fork_data)
            result[fork_name] = fork_data
        return result


class Crop:
    def __init__(self, crop_len):
        self.crop_len = crop_len

    def __call__(self, data):
        crop_len = self.crop_len
        if len(data[0]) > crop_len:
            start_idx = np.random.randint(len(data[0]) - crop_len)
            data = data[:, start_idx: start_idx + crop_len]
        return data

class Threshold:
    def __init__(self, threshold=None, sigma=None):
        assert bool(threshold is None) != bool(sigma is None),\
            (bool(threshold is None), bool(sigma is None))
        self.thr = threshold
        self.sigma = sigma


    def __call__(self, data):
        if self.sigma is None:
            data[np.abs(data) > self.thr] = self.thr
        else:
            data[np.abs(data) > data.std()*self.sigma] = data.std()*self.sigma
        return data


class RandomMultiplier:
    def __init__(self, multiplier=-1.):
        self.multiplier = multiplier
    def __call__(self, data):
        multiplier = self.multiplier if random.random() < .5 else 1.
        return data * multiplier

class Logarithm:
    def __call__(self, data):
        return np.log(np.abs(data)+1e-8)


class Spectrogram:
    def __init__(self, NFFT=None, overlap=None):
        self.NFFT = NFFT
        self.overlap = overlap
        if overlap is None:
            self.overlap = NFFT - 1
    def __call__(self, data):
        data = data.squeeze()
        assert len(data.shape) == 1
        length = len(data)
        Sx = signal.spectrogram(
            x=data,
            nperseg=self.NFFT,
            noverlap=self.overlap)[-1]
        Sx = signal.resample(Sx, length, axis=1)
        return Sx
### Transformations
