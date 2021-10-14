"""
Some of these functions are a from the mne-features packages (
https://github.com/mne-tools/mne-features) with some small modifications and
all the credit goes to the authors of this package.
"""
import numpy as np
from mne.time_frequency import psd_array_welch, psd_array_multitaper

def power_spectrum(sfreq, data, fmin=0., fmax=256., psd_method='welch',
                   welch_n_fft=256, welch_n_per_seg=None, welch_n_overlap=0,
                   mt_bandwidth=None,mt_adaptive=False,mt_low_bias=True,
                   verbose=True):
    """Power Spectral Density (PSD).

    Utility function to compute the (one-sided) Power Spectral Density which
    acts as a wrapper for :func:`mne.time_frequency.psd_array_welch` (if
    ``method='welch'``) or :func:`mne.time_frequency.psd_array_multitaper`
    (if ``method='multitaper'``). The multitaper method, although more
    computationally intensive than Welch's method or FFT, should be prefered
    for 'short' windows. Welch's method is more suitable for 'long' windows.

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    data : ndarray, shape (..., n_times).

    fmin : float (default: 0.)
        Lower bound of the frequency range to consider.

    fmax : float (default: 256.)
        Upper bound of the frequency range to consider.

    psd_method : str (default: 'welch')
        Method used to estimate the PSD from the data. The valid values for
        the parameter ``method`` are: ``'welch'``, ``'fft'`` or
        ``'multitaper'``.

    welch_n_fft : int (default: 256)
        The length of the FFT used. The segments will be zero-padded if
        `welch_n_fft > welch_n_per_seg`. This parameter will be ignored if
        `method = 'fft'` or `method = 'multitaper'`.

    welch_n_per_seg : int or None (default: None)
        Length of each Welch segment (windowed with a Hamming window). If
        None, `welch_n_per_seg` is equal to `welch_n_fft`. This parameter
        will be ignored if `method = 'fft'` or `method = 'multitaper'`.

    welch_n_overlap : int (default: 0)
        The number of points of overlap between segments. Should be
        `<= welch_n_per_seg`. This parameter will be ignored if
        `method = 'fft'` or `method = 'multitaper'`.

    mt_bandwidth : int (default: None)
        The bandwidth of the multi taper windowing function in Hz.

    mt_adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD
        (slow, use n_jobs >> 1 to speed up computation).

    mt_low_bias : bool
        Only use tapers with more than 90%% spectral concentration within
        bandwidth.

    verbose : bool (default: False)
        Verbosity parameter. If True, info and warnings related to
        :func:`mne.time_frequency.psd_array_welch` or
        :func:`mne.time_frequency.psd_array_multitaper` are printed.

    Returns
    -------
    psd : ndarray, shape (..., n_freqs)
        Estimated PSD.

    freqs : ndarray, shape (n_freqs,)
        Array of frequency bins.
    """
    _verbose = 10 * (1 - int(verbose))
    _fmin, _fmax = max(0, fmin), min(fmax, sfreq / 2)
    if psd_method == 'welch':
        _n_fft = min(data.shape[-1], welch_n_fft)
        return psd_array_welch(data, sfreq, fmin=_fmin, fmax=_fmax,
                               n_fft=_n_fft, verbose=_verbose,
                               n_per_seg=welch_n_per_seg,
                               n_overlap=welch_n_overlap)
    elif psd_method == 'multitaper':
        return psd_array_multitaper(data, sfreq, fmin=_fmin, fmax=_fmax,
                                    bandwidth=mt_bandwidth,adaptive=mt_adaptive,
                                    low_bias=mt_low_bias,normalization='full',
                                    verbose='CRITICAL')
    elif psd_method == 'fft':
        n_times = data.shape[-1]
        m = np.mean(data, axis=-1)
        _data = data - m[..., None]
        spect = np.fft.rfft(_data, n_times)
        mag = np.abs(spect)
        freqs = np.fft.rfftfreq(n_times, 1. / sfreq)
        psd = np.power(mag, 2) / (n_times ** 2)
        psd *= 2.
        psd[..., 0] /= 2.
        if n_times % 2 == 0:
            psd[..., -1] /= 2.
        mask = np.logical_and(freqs >= _fmin, freqs <= _fmax)
        return psd[..., mask], freqs[mask]
    else:
        raise ValueError('The given method (%s) is not implemented. Valid '
                         'methods for the computation of the PSD are: '
                         '`welch`, `fft` or `multitaper`.' % str(psd_method))


def _psd_params_checker(params,psd_method):
    """Utility function to check parameters to be passed to `power_spectrum`.

    Parameters
    ----------
    params : dict or None
        Optional parameters to be passed to
        :func:`mne_features.utils.power_spectrum`. If `params` contains a key
        which is not an optional parameter of
        :func:`mne_features.utils.power_spectrum`, an error is raised.

    psd_method : str
        Method used for the estimation of the Power Spectral Density (PSD).
        Valid methods are: ``'welch'``, ``'multitaper'`` or ``'fft'``.

    Returns
    -------
    valid_params : dict
    """
    if params is None:
        return dict()
    elif not isinstance(params, dict):
        raise ValueError('The parameter `psd_params` has type %s. Expected '
                         'dict instead.' % type(params))
    else:
        expected_keys = ['']
        if psd_method=='welch':
            expected_keys = ['welch_n_fft', 'welch_n_per_seg', 'welch_n_overlap']
        if psd_method=='multitaper':
            expected_keys = ['mt_bandwidth','mt_adaptive','mt_low_bias']
        valid_keys = list()
        for n in params:
            if n not in expected_keys:
                raise ValueError('The key %s in `psd_params` is not valid and '
                                 'will be ignored. Valid keys are: %s' %
                                 (n, str(expected_keys)))
            else:
                valid_keys.append(n)
        valid_params = {n: params[n] for n in valid_keys}
        return valid_params





