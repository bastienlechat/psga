"""
This is a combination of multiple function to extract spectral features from a
mne Epochs objects. Some of these functions are a from the mne-features packages
(https://github.com/mne-tools/mne-features) with some small modifications and
all the credit goes to the authors of this package.
"""

import numpy as np
from scipy import integrate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, explained_variance_score
from .utils import (power_spectrum, _psd_params_checker)

def _freq_bands_helper(sfreq, freq_bands):
    """Utility function to define frequency bands.

    This utility function is to be used with :func:`compute_pow_freq_bands` and
    :func:`compute_energy_freq_bands`. It essentially checks if the given
    parameter ``freq_bands`` is valid and raises an error if not.

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    freq_bands : ndarray, shape (n_freq_bands + 1,) or (n_freq_bands, 2)
        Array defining frequency bands.

    Returns
    -------
    valid_freq_bands : ndarray, shape (n_freq_bands, 2)
    """
    if not np.logical_and(freq_bands >= 0, freq_bands <= sfreq / 2).all():
        raise ValueError('The entries of the given `freq_bands` parameter '
                         '(%s) must be positive and less than the Nyquist '
                         'frequency.' % str(freq_bands))
    else:
        if freq_bands.ndim == 1:
            n_freq_bands = freq_bands.shape[0] - 1
            valid_freq_bands = np.empty((n_freq_bands, 2))
            for j in range(n_freq_bands):
                valid_freq_bands[j, :] = freq_bands[j:j + 2]
        elif freq_bands.ndim == 2 and freq_bands.shape[-1] == 2:
            valid_freq_bands = freq_bands
        else:
            raise ValueError('The given value (%s) for the `freq_bands` '
                             'parameter is not valid. Only 1D or 2D arrays '
                             'with shape (n_freq_bands, 2) are accepted.'
                             % str(freq_bands))
        return valid_freq_bands

def compute_absol_pow_freq_bands(sfreq, data, freq_bands=np.array([0.5,4.5,8,12,16,35]),
                            psd_method='welch',
                           psd_params=None, precomputed_psd=None):
    """Power Spectrum (computed by frequency bands).

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    data : ndarray, shape (n_channels, n_times)

    freq_bands : ndarray or dict (default: np.array([.5, 4, 8, 13, 30, 100]))
        The parameter ``freq_bands`` should be either a ndarray with shape
        ``(n_freq_bands + 1,)`` or ``(n_freq_bands, 2)`` or a dict. If ndarray
        with shape ``(n_freq_bands + 1,)``, the entries define **contiguous**
        frequency bands as follows: the i-th frequency band is defined as:
        [freq_bands[i], freq_bands[i + 1]] (0 <= i <= n_freq_bands - 1). If
        ndarray with shape ``(n_freq_bands, 2)``, the rows of ``freq_bands``
        define **non-contiguous** frequency bands. If dict, the keys should be
        strings (names of the frequency bands) and the values, the
        corresponding bands (as ndarray with shape (2,) or list of length 2).
        When ``freq_bands`` is of type dict, the keys are used to generate the
        feature names (only used when features are extracted with
        ``return_as_df=True``). The values of ``freq_bands`` should be between
        0 and sfreq / 2 (the Nyquist frequency) as the function uses the
        one-sided PSD.

    psd_method : str (default: 'welch')
        Method used for the estimation of the Power Spectral Density (PSD).
        Valid methods are: ``'welch'``, ``'multitaper'`` or ``'fft'``.

    psd_params : dict or None (default: None)
        If not None, dict with optional parameters (`welch_n_fft`,
        `welch_n_per_seg`, `welch_n_overlap`) to be passed to
        :func:`mne_features.utils.power_spectrum`. If None, default parameters
        are used (see doc for :func:`mne_features.utils.power_spectrum`).

    precomputed_psd : dict or None (default: None)
        If None, calculate the power spectrum using :func:`mne_features.utils.power_spectrum`
        If not None, dict with parameters (`psd`, `freqs`).

    Returns
    -------
    output : ndarray, shape (n_epochs, n_freq_bands)

    Notes
    -----
    Alias of the feature function: **pow_freq_bands**. See [1]_.

    References
    ----------
    .. [1] Teixeira, C. A. et al. (2011). EPILAB: A software package for
           studies on the prediction of epileptic seizures. Journal of
           Neuroscience Methods, 200(2), 257-271.
    """
    n_channels = data.shape[0]
    if isinstance(freq_bands, dict):
        _freq_bands = np.asarray([freq_bands[n] for n in freq_bands])
    else:
        _freq_bands = np.asarray(freq_bands)
    fb = _freq_bands_helper(sfreq, _freq_bands)
    n_freq_bands = fb.shape[0]

    if precomputed_psd is None:
        _psd_params = _psd_params_checker(psd_params, psd_method)
        psd, freqs = power_spectrum(sfreq, data, psd_method=psd_method,
                                    **_psd_params)
    else:
        psd, freqs = precomputed_psd['psd'],precomputed_psd['freqs']

    pow_freq_bands = np.empty((n_channels, n_freq_bands))

    for j in range(n_freq_bands):

        mask = np.logical_and(freqs >= fb[j, 0], freqs <= fb[j, 1])
        df = freqs[1]-freqs[0]
        psd_band = psd[:, mask]

        pow_freq_bands[:, j] = integrate.simps(psd_band,dx=df, axis=-1)

    return pow_freq_bands

def compute_relative_pow_ratios(absolute_power):
    """Relative power ratios based on the absoulte power in the delta,theta,alpha,sigma and beta bands.
    Additionally calculate slowing ratio [1] and

        Parameters
        ----------
        absolute_power : ndarray, shape (n_epochs, n_freq_band)

        Returns
        -------
        output : ndarray, shape (n_epochs, n_freq_bands + 2)

        Notes
        -----
        Alias of the feature function: **pow_freq_bands**. See [1]_.

        References
        ----------
        .. [1]
        """
    delta, theta, alpha, sigma, beta = np.split(absolute_power,5,-1)


    relative_power = absolute_power/ np.sum(absolute_power, axis=-1)[:,None]

    # delta/alpha ratio
    DAR = delta/alpha

    #SlowingRatio
    SR = (delta+theta)/(alpha+sigma+beta)
    # SlowingRatio
    REMR = (theta) / (alpha + sigma + beta)
    return np.column_stack((relative_power,DAR,SR,REMR))

########################## Spectral Features ##################################################
def compute_hjorth_spect(sfreq, data, normalize=False,
                                  psd_method='welch', psd_params=None, precomputed_psd=None):
    """Hjorth mobility (per channel).

    Hjorth mobility parameter computed from the Power Spectrum of the data.

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    data : ndarray, shape (n_channels, n_times)

    normalize : bool (default: False)
        Normalize the result by the total power.

    psd_method : str (default: 'welch')
        Method used for the estimation of the Power Spectral Density (PSD).
        Valid methods are: ``'welch'``, ``'multitaper'`` or ``'fft'``.

    psd_params : dict or None (default: None)
        If not None, dict with optional parameters (`welch_n_fft`,
        `welch_n_per_seg`, `welch_n_overlap`) to be passed to
        :func:`mne_features.utils.power_spectrum`. If None, default parameters
        are used (see doc for :func:`mne_features.utils.power_spectrum`).

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **hjorth_mobility_spect**. See [1]_ and
    [2]_.

    References
    ----------
    .. [1] Mormann, F. et al. (2006). Seizure prediction: the long and
           winding road. Brain, 130(2), 314-333.

    .. [2] Teixeira, C. A. et al. (2011). EPILAB: A software package for
           studies on the prediction of epileptic seizures. Journal of
           Neuroscience Methods, 200(2), 257-271.
    """
    if precomputed_psd is None:
        _psd_params = _psd_params_checker(psd_params, psd_method)
        psd, freqs = power_spectrum(sfreq, data, psd_method=psd_method,
                                    **_psd_params)
    else:
        psd, freqs = precomputed_psd['psd'], precomputed_psd['freqs']

    w_freqs_2 = np.power(freqs, 2)
    w_freqs_4 = np.power(freqs, 4)
    complexity = np.sum(np.multiply(psd, w_freqs_4), axis=-1)

    mobility = np.sum(np.multiply(psd, w_freqs_2), axis=-1)

    if normalize:
        mobility = np.divide(mobility, np.sum(psd, axis=-1))
        complexity = np.divide(complexity, np.sum(psd, axis=-1))
    return np.column_stack((mobility, complexity))


def compute_spect_entropy(sfreq, data, psd_method='welch', psd_params=None,precomputed_psd=None):
    """Spectral Entropy (per channel).

    Spectral Entropy is defined to be the Shannon Entropy of the Power
    Spectrum of the data.

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data

    data : ndarray, shape (n_channels, n_times)

    psd_method : str (default: 'welch')
        Method used for the estimation of the Power Spectral Density (PSD).
        Valid methods are: ``'welch'``, ``'multitaper'`` or ``'fft'``.

    psd_params : dict or None (default: None)
        If not None, dict with optional parameters (`welch_n_fft`,
        `welch_n_per_seg`, `welch_n_overlap`) to be passed to
        :func:`mne_features.utils.power_spectrum`. If None, default parameters
        are used (see doc for :func:`mne_features.utils.power_spectrum`).

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **spect_entropy**. See [1]_.

    References
    ----------
    .. [1] Inouye, T. et al. (1991). Quantification of EEG irregularity by
           use of the entropy of the power spectrum. Electroencephalography
           and clinical neurophysiology, 79(3), 204-210.
    """
    if precomputed_psd is None:
        _psd_params = _psd_params_checker(psd_params, psd_method)
        psd, freqs = power_spectrum(sfreq, data, psd_method=psd_method,
                                    **_psd_params)
    else:
        psd, freqs = precomputed_psd['psd'], precomputed_psd['freqs']
    m = np.sum(psd, axis=-1)

    psd_norm = np.divide(psd[:, 1:], m[:, None])
    return -np.sum(np.multiply(psd_norm, np.log2(psd_norm)), axis=-1)


def compute_spect_slope(sfreq, data, fmin=0.1, fmax=50,
                        with_intercept=True, psd_method='welch',
                        psd_params=None,precomputed_psd=None):
    """Linear regression of the the log-log frequency-curve (per channel).

    Using a linear regression, the function estimates the slope and the
    intercept (if ``with_intercept`` is True) of the Power Spectral Density
    (PSD) in the log-log scale. In addition to this, the Mean Square Error
    (MSE) and R2 coefficient (goodness-of-fit) are returned. By default, the
    [0.1Hz, 50Hz] frequency range is used for the regression.

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    data : ndarray, shape (n_channels, n_times)

    fmin : float (default: 0.1)
        Lower bound of the frequency range considered in the linear regression.

    fmax : float (default: 50)
        Upper bound of the frequency range considered in the linear regression.

    with_intercept : bool (default: True)
        If True, the intercept of the linear regression is included among the
        features returned by the function. If False, only the slope, the MSE
        and the R2 coefficient are returned.

    psd_method : str (default: 'welch')
        Method used for the estimation of the Power Spectral Density (PSD).
        Valid methods are: ``'welch'``, ``'multitaper'`` or ``'fft'``.

    psd_params : dict or None (default: None)
        If not None, dict with optional parameters (`welch_n_fft`,
        `welch_n_per_seg`, `welch_n_overlap`) to be passed to
        :func:`mne_features.utils.power_spectrum`. If None, default parameters
        are used (see doc for :func:`mne_features.utils.power_spectrum`).

    Returns
    -------
    output : ndarray, shape (n_channels * 4,)
        The four characteristics: intercept, slope, MSE, and R2 per channel.

    Notes
    -----
    Alias of the feature function: **spect_slope**. See [1]_
    and [2]_.

    References
    ----------
    .. [1] Demanuelle C. et al. (2007). Distinguishing low frequency
           oscillations within the 1/f spectral behaviour of electromagnetic
           brain signals. Behavioral and Brain Functions (BBF).

    .. [2] Winkler I. et al. (2011). Automatic Classification of Artifactual
           ICA-Components for Artifact Removal in EEG Signals. Behavioral and
           Brain Functions (BBF).
    """
    n_channels = data.shape[0]
    if precomputed_psd is None:
        _psd_params = _psd_params_checker(psd_params, psd_method)
        psd, freqs = power_spectrum(sfreq, data, psd_method=psd_method,
                                    **_psd_params)
    else:
        psd, freqs = precomputed_psd['psd'], precomputed_psd['freqs']

    # mask limiting to input freq_range
    if fmin == None:
        fmin = freqs[1]
    mask = np.logical_and(freqs >= fmin, freqs <= fmax)

    # freqs and psd selected over input freq_range and expressed in log scale
    freqs, psd = np.log10(freqs[mask]), np.log10(psd[:, mask])

    # linear fit
    lm = LinearRegression()
    fit_info = np.empty((n_channels, 4))
    for idx, power in enumerate(psd):
        lm.fit(freqs.reshape(-1, 1), power)
        fit_info[idx, 0] = lm.intercept_
        fit_info[idx, 1] = lm.coef_
        power_estimate = lm.predict(freqs.reshape(-1, 1))
        fit_info[idx, 2] = mean_squared_error(power, power_estimate)
        fit_info[idx, 3] = explained_variance_score(power, power_estimate)

    if not with_intercept:
        fit_info = fit_info[:, 1:]

    return fit_info

def compute_spect_edge_freq(sfreq, data, ref_freq=None, edge=None,
                            psd_method='welch', psd_params=None,precomputed_psd=None):
    """Spectal Edge Frequency (per channel).

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.

    data : ndarray, shape (n_channels, n_times)

    ref_freq : float or None (default: None)
        If not None, reference frequency for the computation of the spectral
        edge frequency. If None, `ref_freq = sfreq / 2` is used.

    edge : list of float or None (default: None)
        If not None, ``edge`` is expected to be a list of values between 0
        and 1. If None, ``edge = [0.5]`` is used.

    psd_method : str (default: 'welch')
        Method used for the estimation of the Power Spectral Density (PSD).
        Valid methods are: ``'welch'``, ``'multitaper'`` or ``'fft'``.

    psd_params : dict or None (default: None)
        If not None, dict with optional parameters (`welch_n_fft`,
        `welch_n_per_seg`, `welch_n_overlap`) to be passed to
        :func:`mne_features.utils.power_spectrum`. If None, default parameters
        are used (see doc for :func:`mne_features.utils.power_spectrum`).

    Returns
    -------
    output : ndarray, shape (n_channels * n_edge,)
        With: `n_edge = 1` if `edge` is None or `n_edge = len(edge)` otherwise.

    Notes
    -----
    Alias of the feature function: **spect_edge_freq**. See [1]_.

    References
    ----------
    .. [1] Mormann, F. et al. (2006). Seizure prediction: the long and winding
           road. Brain, 130(2), 314-333.
    """

    if precomputed_psd is None:
        _psd_params = _psd_params_checker(psd_params, psd_method)
        psd, freqs = power_spectrum(sfreq, data, psd_method=psd_method,
                                    **_psd_params)
    else:
        psd, freqs = precomputed_psd['psd'], precomputed_psd['freqs']

    if ref_freq is None:
        _ref_freq = freqs[-1]
    else:
        _ref_freq = float(ref_freq)
    if edge is None:
        _edge = [0.5]
    else:
        # Check the values in `edge`
        if not all([0 <= p <= 1 for p in edge]):
            raise ValueError('The values in ``edge``` must be floats between '
                             '0 and 1. Got {} instead.'.format(edge))
        else:
            _edge = edge
    n_edge = len(_edge)
    n_channels, n_times = data.shape
    spect_edge_freq = np.empty((n_channels, n_edge))

    out = np.cumsum(psd, 1)
    for i, p in enumerate(_edge):

        idx_ref = np.where(freqs >= _ref_freq)[0][0]
        ref_pow = np.sum(psd[:, :(idx_ref + 1)], axis=-1)
        for j in range(n_channels):
            idx = np.where(out[j, :] >= p * ref_pow[j])[0]
            if idx.size > 0:
                spect_edge_freq[j, i] = freqs[idx[0]]
            else:
                spect_edge_freq[j, i] = -1


    return spect_edge_freq
