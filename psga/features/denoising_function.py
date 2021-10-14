import numpy as np
import scipy.signal
from scipy.signal import wiener, filtfilt, butter, gaussian, freqz, medfilt
from scipy.ndimage import filters
import warnings
#You can create a class with static methods and use getattr to get the correct method. It's similar to what mgilson suggests but you essentially get the dict creation for free:


def moving_average_weighted(y, windows_length=120, std=10, **kwargs):
    b = gaussian(windows_length, std)  # 30,10
    ga = filters.convolve1d(y, b / b.sum())
    return ga

def median_filer(y, windows_length=120, **kwargs):
    filtered = medfilt(y, kernel_size=windows_length)
    return filtered

def moving_average(y, windows_length=120, **kwargs):
    b = np.ones(windows_length)
    ga = filters.convolve1d(y, b / b.sum())
    return ga
