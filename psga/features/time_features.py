"""
This is a combination of multiple function to exctract temporal features from a
mne Epochs objects. Some of these functions are a from the mne-features packages
(https://github.com/mne-tools/mne-features) with some small modifications and
all the credit goes to the authors of this package.

TODO: In the futur, we should just rely on mne_features instead of doubling up
"""

import numpy as np
import scipy.stats as stats

def compute_mean(data):
    """Mean of the data (per channel).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **mean**
    """
    return np.mean(data, axis=-1)


def compute_variance(data):
    """Variance of the data (per channel).

    Parameters
    ----------
    data : shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **variance**
    """
    return np.var(data, axis=-1, ddof=1)


def compute_std(data):
    """Standard deviation of the data.

    Parameters
    ----------
    data : shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels)

    Notes
    -----
    Alias of the feature function: **std**
    """
    return np.std(data, axis=-1, ddof=1)


def compute_ptp_amp(data):
    """Peak-to-peak (PTP) amplitude of the data (per channel).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **ptp_amp**
    """
    return np.ptp(data, axis=-1)


def compute_skewness(data):
    """Skewness of the data (per channel).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **skewness**
    """
    ndim = data.ndim
    return stats.skew(data, axis=ndim - 1)


def compute_kurtosis(data):
    """Kurtosis of the data (per channel).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **kurtosis**
    """
    ndim = data.ndim
    return stats.kurtosis(data, axis=ndim - 1, fisher=False)


def compute_maximum_value_epochs(data):
    """Maximum value of the data (per channel).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels,)
    """
    maximums = np.max(np.abs(data),axis=-1)
    return maximums

def compute_rms_value_epochs(data):
    """RMS value of the data (per channel).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels,)
    """
    rms = np.sqrt(np.mean(np.square(data),axis=-1))
    return rms


def compute_zero_crossings(data, threshold=np.finfo(np.float64).eps):
    """Number of zero-crossings (per channel).

    The ``threshold`` parameter is used to clip 'small' values to zero.
    Changing its default value is likely to affect the number of
    zero-crossings returned by the function.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    threshold : float (default: np.finfo(np.float64).eps)
        Threshold used to determine when a float should de treated as zero.

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **zero_crossings**
    """
    _data = data.copy()
    # clip 'small' values to 0
    _data[np.abs(_data) < threshold] = 0
    sgn = np.sign(_data)
    # sgn may already contain 0 values (either 'true' zeros or clipped values)
    aux = np.diff((sgn == 0).astype(np.int64), axis=-1)
    count = np.sum(aux == 1, axis=-1) + (_data[:, 0] == 0)
    # zero between two consecutive time points (data[i] * data[i + 1] < 0)
    mask_implicit_zeros = sgn[:, 1:] * sgn[:, :-1] < 0
    count += np.sum(mask_implicit_zeros, axis=-1)
    return count

def _hjorth_mobility(data):
    """Hjorth mobility (per channel).

    Hjorth mobility parameter computed in the time domain.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **hjorth_mobility**. See [1]_.

    References
    ----------
    .. [1] Paivinen, N. et al. (2005). Epileptic seizure detection: A
           nonlinear viewpoint. Computer methods and programs in biomedicine,
           79(2), 151-159.
    """
    x = np.insert(data, 0, 0, axis=-1)
    dx = np.diff(x, axis=-1)
    sx = np.std(x, ddof=1, axis=-1)
    sdx = np.std(dx, ddof=1, axis=-1)
    mobility = np.divide(sdx, sx)
    return mobility

def compute_hjorth(data):
    """Hjorth mobility (per channel).

    Hjorth mobility parameter computed in the time domain.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)

    Returns
    -------
    output : ndarray, shape (n_channels,)

    Notes
    -----
    Alias of the feature function: **hjorth_mobility**. See [1]_.

    References
    ----------
    .. [1] Paivinen, N. et al. (2005). Epileptic seizure detection: A
           nonlinear viewpoint. Computer methods and programs in biomedicine,
           79(2), 151-159.
    """


    x = np.insert(data, 0, 0, axis=-1)
    dx = np.diff(x, axis=-1)
    activity = np.var(x, ddof=1, axis=-1)
    mobility = _hjorth_mobility(data)

    m_dx = _hjorth_mobility(dx)
    complexity = np.divide(m_dx, mobility)

    return np.column_stack((activity,mobility,complexity))

def compute_time_mass(data, q=[0.5]):

    """Calculate time index where data is >= at q * total energy of the signal.


    Parameters
    ----------
    data : ndarray, shape (n_epochs, n_times)

    Returns
    -------
    output : ndarray, shape (n_epochs,)

    Notes
    -----

    References
    ----------
    ..
    """

    x = np.asarray(data)
    abs_data = np.abs(x)

    n_q = len(q)
    n_epochs, n_times = data.shape
    time_mass = np.empty((n_epochs, n_q))

    out = np.cumsum(data, 1)
    for i, p in enumerate(q):

        ref_pow = np.sum(abs_data, axis=-1)
        for j in range(n_epochs):
            idx = np.where(out[j, :] >= p * ref_pow[j])[0]
            if idx.size > 0:
                time_mass[j, i] = idx[0]/n_times
            else:
                time_mass[j, i] = -1
    return time_mass

def compute_ratio_energy_time_mass(data, q=[0.5]):

    """Calculate time index where data is >= at q * total energy of the signal.


    Parameters
    ----------
    data : ndarray, shape (n_epochs, n_times)

    Returns
    -------
    output : ndarray, shape (n_epochs,)

    Notes
    -----

    References
    ----------
    ..
    """

    x = np.asarray(data)
    abs_data = np.abs(x)

    n_q = len(q)
    n_epochs, n_times = data.shape
    ratio_mass = np.empty((n_epochs, n_q))

    out = np.cumsum(data, 1)
    for i, p in enumerate(q):

        ref_pow = np.sum(abs_data, axis=-1)
        for j in range(n_epochs):
            idx = np.where(out[j, :] >= p * ref_pow[j])[0]
            if idx.size > 0:
                ratio_mass[j, i] = np.mean(abs_data[j,:idx[0]])/np.mean(abs_data[j,idx[0]:])
            else:
                ratio_mass[j, i] = -1
    return ratio_mass