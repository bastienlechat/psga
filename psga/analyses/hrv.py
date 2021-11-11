"""
This modules performs heart-rate variability analyses. The QRS detector was
developed in-house and tested on the MIT-BIT arrhythmia database from
PhysioNet with a sensitivity of 97% (SD, 3.7%) and a specificity of 99% (SD,
2%).

This type of performance is averaged for a QRS detector, however other
available detectors (such as the one in the MNE packages) tended to perform
poorly in sleep data. My guess is that available algorithms assumes a constant
ECG amplitude whereas ECG amplitude can shift during sleep (due to lose
adherence of the sensors of the skin maybe?). The provided detectors was
inspected visually on sleep data, and seemed to perform ok. A more thorough
validation nevertheless remain to be performed.

Summary values of the overnight heart rate variability are based on the
previously published methodology in the European Heart Journal [1]. Furthemore, the
R-peaks cleaning (ecto-beats) etc, follows what has been done in the
[Multi-Ethnic Study of Atherosclerosis](https://sleepdata.org/datasets/mesa).

This module also implements event analysis for heart rate (e.g. analysis of
heart rate surges as a results of apnea events). This function reports
the 5 beats before an event onsets and 15 beats after an event onsets. It
reports both timing (in seconds) and heart rate values. This methodology was
used in a paper looking at the effect of environmental noise on sleep [2].
Tips: it's important to check that timing windows (rpeaks[-5] to rpeaks[15])
is not too big. A huge time windows suggest that there was a lot of rejected
r-peaks (due to noise), and this event maybe should be excluded from your
analysis.

[1] Malik, M., Bigger, J. T., Camm, A. J., Kleiger, R. E., Malliani, A.,
Moss, A. J., & Schwartz, P. J. (1996). Heart rate variability: Standards
of measurement, physiological interpretation, and clinical use.
European Heart Journal, 17(3), 354-381.
doi:10.1093/oxfordjournals.eurheartj.a014868

[2] TBA
"""
import os
import mne
import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as signal
import scipy.interpolate
import pywt

from psga.features.spectral_features import power_spectrum, \
    compute_absol_pow_freq_bands
from psga.features.time_features import compute_rms_value_epochs
from .base import BaseMethods
import sys
import pandas as pd

try:
    wd = sys._MEIPASS
except AttributeError:
    wd = os.path.dirname(__file__)

class HRV(BaseMethods):
    """Performs heart rate variability analysis.

    "score" function find R-peaks and calculate normal to normal beats interval (nni).

    "score_events" gives the 5 preceding and 15 following NNI from an event onset.

    Overnight metrics calculate usual heart rate variability (see [1]) markers
    over the all overnight period. We also calculate some temporal markers of
    heart rate variability for individual sleep stages
    (can not be done for frequency markers since it requires at least
    5 minutes of continuous NNI)

    Parameters
    ----------

    welch_n_fft : int
        Number of point on which to perform fft (default 2048)

    welch_n_overlap : int
        Number of overlapping points between segments (default 1024)

    frequency_bands : list
        Ultra-low frequency (0.003 to 0.04), low frequency (0.04 to 0.15) and
        high frequency (0.15 to 0.40) bounds for calculation of frequency
        characteristics of HRV. Default to [0.003,0.4,0.15,0.40]

    Notes
    -----
    Summary statistics for heart rate variability are based on the previously
    published methodology in the European Heart Journal [1]. Furthemore, the
    R-peaks cleaning (ecto-beats) etc, follows what has been done in the
    [Multi-Ethnic Study of Atherosclerosis](
    https://sleepdata.org/datasets/mesa/pages/hrv-analysis.md).

    In-house validation of the QRS detection algorithm on the MIT-BIT arrhythmia
    database from PhysioNet yielded a sensitivity of 97% (SD, 3.7%)
    and a specificity of 99% (SD, 2%). The results were published in [2].

    References
    ----------

    [1] Malik, M., Bigger, J. T., Camm, A. J., Kleiger, R. E., Malliani, A.,
    Moss, A. J., & Schwartz, P. J. (1996). Heart rate variability: Standards
    of measurement, physiological interpretation, and clinical use.
    European Heart Journal, 17(3), 354-381.
    doi:10.1093/oxfordjournals.eurheartj.a014868

    [2] TBA

    """

    def __init__(self, welch_n_fft = 2048, welch_n_overlap=1024,
                 frequency_bands = [0.003, 0.04, 0.15, 0.40]):
        self.psd_method = 'welch'
        self.psd_params = {'welch_n_fft':welch_n_fft,
                           'welch_n_overlap':welch_n_overlap}
        self.frequency_bands = frequency_bands
        self._ECG_chan = None #'ECG'
        super().__init__()

    def fit(self, raw, hypnogram, **kwargs):
        assert (self._ECG_chan is not None or len(raw.info['ch_names']) == 0)
        if self._ECG_chan is None: self._ECG_chan = raw.info['ch_names'][0]
        self._check_raw(raw)
        self._check_hypno(hypnogram)

        self.fs = raw.info['sfreq']
        _ecg = raw[self._ECG_chan,:][0].squeeze() * 10 **3
        self._ecg = mne.filter.filter_data(_ecg,self.fs,l_freq=1,
                                       h_freq=20, verbose='CRITICAL')

    def score(self,plot=False):
        """ score Rpeaks on one ECG channel"""
        self._scoring = {}
        ecg = self._ecg
        fs = self.fs
        rpeaks = QRS_detection(ecg, fs=fs)
        outliers, ecto_beat = noisy_interval(rpeaks,fs)
        rri = np.diff(rpeaks) / fs
        rpeaks = rpeaks[1:]
        if plot:
            plt.plot(ecg)
            plt.plot(rpeaks,ecg[rpeaks],'r*')
            plt.plot(rpeaks[outliers==1], ecg[rpeaks[outliers==1]], 'g^')
            plt.show()
        ecg_vals = np.hstack(ecg[rpeaks])

        self._scoring[self._ECG_chan] = {'rpeaks':rpeaks/fs,'rri':rri,
                                 'Noisy':outliers,'Ecto_beat':ecto_beat,
                                'rpeaksAmp':ecg_vals,
                                 }
        return self._scoring, None

    def overnight_metrics(self):
        noisy = self._scoring[self._ECG_chan]['Noisy']
        rp = self._scoring[self._ECG_chan]['rpeaks'][noisy==0]
        rri = self._scoring[self._ECG_chan]['rri'][noisy==0]

        vlf, lf, hf, lf_ratio, hf_ratio, total_power, _ = \
            frequency_markers_rr(rp, rri,
                                psd_method=self.psd_method,
                                psd_params= self.psd_params,
                                frequency_bands=self.frequency_bands,
                                method='all'
                                )
        vlf_5, lf_5, hf_5, lf_ratio_5, hf_ratio_5, total_power_5, noisy_epochs\
            = frequency_markers_rr(rp, rri,
                                psd_method=self.psd_method,
                                psd_params= self.psd_params,
                                frequency_bands=self.frequency_bands,
                                method='five')
        mean_nn, median_nn, nn_std, mean_hr, std_hr, rmssd, nni_50, pnni_50, \
        std_5nn, std_sd5nn = temporal_markers_rr(rri, rp)

        return {'vlf':vlf,'lf':lf,'hf':hf,'lf_ratio':lf_ratio,'hf_ratio':hf_ratio,
                'total_power':total_power,'vlf_5': vlf_5, 'lf_5': lf_5,
                'hf_5': hf_5, 'lf_ratio_5': lf_ratio_5,
                'hf_ratio_5': hf_ratio_5, 'total_power_5': total_power_5,
                'mean_nn':mean_nn,'median_nn':median_nn, 'nn_std':nn_std,
                'mean_hr':mean_hr, 'std_HR':std_hr, 'rmssd':rmssd,
                'nni_50':nni_50, 'pnni50':pnni_50,'std_5nn':std_5nn,
                'std_sd5nn':std_sd5nn, 'Noisy_5min': int(noisy_epochs),
                'Noisy_rpeaks': np.sum(self._scoring[self._ECG_chan]['Noisy']),
                'N_nni':len(self._scoring[self._ECG_chan]['Noisy'])}

    def score_from_events(self,events):
        """
        Locates the 5 preceding and 15 following Rpeaks (both location and
        respective HR values) from an event onset. This can be useful to
        study heart rate surges to any type of events, such as apneas. This
        was done recently in [1] (although they used photoplethysmography
        derived HR instead of ECG).

        Parameters
        ----------
        events : pd.DataFrame

        Returns
        -------
        rp_dataframe : pd.Dataframe
            Dataframe containing heart rate events scoring

        References
        ----------
        [1] Azarbarzin, A., Sands, S. A., Younes, M., Taranto-Montemurro,
        L., Sofer, T., Vena, D., . . . Wellman, A. (2021).
        The Sleep Apnea-specific Pulse Rate Response Predicts
        Cardiovascular Morbidity and Mortality.
        Am J Respir Crit Care Med. doi:10.1164/rccm.202010-3900OC
        """
        ch_name = self._ECG_chan
        rpeaks = self._scoring[ch_name]['rpeaks']
        hr = np.divide(60, self._scoring[ch_name]['rri'])
        events_onset = events.onset

        rpeaks_events = np.zeros((len(events_onset), 40))
        for event_count, single_event_onset in enumerate(events_onset):
            if len(np.argwhere(rpeaks > single_event_onset)) > 14:
                if len(np.argwhere(rpeaks < single_event_onset)) > 5:
                    args_baseline = np.argwhere(rpeaks < single_event_onset)[
                                    -5:]
                    args_after = np.argwhere(rpeaks > single_event_onset)[:15]
                    rp_baseline = rpeaks[args_baseline].squeeze()
                    IBI_baseline = hr[args_baseline].squeeze()
                    rp_signal = rpeaks[args_after].squeeze()
                    IBI_signal = hr[args_after].squeeze()
                    rpeaks_events[event_count, :] = np.hstack([rp_baseline,
                                                               rp_signal,
                                                               IBI_baseline,
                                                               IBI_signal])
        IBI_labels = ['IBI_' + str(k) for k in np.arange(20) - 5]
        rpeaks_labels = ['RP_' + str(k) for k in np.arange(20) - 5]

        rp_dataframe = pd.DataFrame(rpeaks_events, columns=np.hstack([
            rpeaks_labels, IBI_labels]
        ))
        rp_dataframe['label'] = events['label']
        rp_dataframe['onset'] = events['onset']
        return rp_dataframe

def temporal_markers_rr(nni, rpeaks):
    """Time based metrics of normal to normal intervals"""
    mean_nn = np.mean(nni) # mean interval
    median_nn = np.median(nni)
    nn_std = np.std(nni)
    hr = np.divide(60,nni)
    mean_hr = np.mean(hr)
    std_hr = np.std(hr)
    diff_nni = np.diff(nni)*1000
    rmssd = np.sqrt(np.mean(diff_nni ** 2))
    nni_50 = sum(np.abs(diff_nni) > 50)
    pnni_50 = 100 * nni_50 / len(nni)
    time = np.arange(rpeaks[0], rpeaks[-1], 60 * 5)
    mn = []
    sd = []
    noisy_epochs = 0
    for lb, ub in zip(time[:-1], time[1:]):
        if np.sum(np.bitwise_and(rpeaks > lb, rpeaks < ub)) < 181:
            noisy_epochs = noisy_epochs + 1
        else:
            mn.append(np.mean(nni[np.bitwise_and(rpeaks > lb, rpeaks < ub)]))
            sd.append(np.std(nni[np.bitwise_and(rpeaks > lb, rpeaks < ub)]))

    return mean_nn, median_nn, nn_std, mean_hr, std_hr, rmssd, nni_50, \
           pnni_50, np.mean(mn), np.mean(sd)

def frequency_markers_rr(rpeaks,nni, psd_method='welch', psd_params = None,
                         frequency_bands = [0.003, 0.04, 0.15, 0.40],
                         method='five'):
    """Frequency metrics of normal-to-normal interval series

    steps:
    1) Interpolate to equally spaced time series
    2) Calculate power spectrum density of nni inteprolated series
    and get absoulte power of very low (0.003, 0.04Hz)
    , low (0.04, 0.15), high freq (0.15, 0.40) frequency bands.
    3) Calculate ratios and total powers

    Parameters
    ----------
    rpeaks : list
        locations of the rpeaks
    nni : list
        normal to normal interval series
    psd_method : str
        Methods used to calculate power spectral densities (default welch)
    psd_params: dict
        Additionnal params to pass to the welch function (dict with the
        keys welch_n_fft and welch_n_overlap)
    frequency_bands : list
        Ultra-low frequency (0.003 to 0.04), low frequency (0.04 to 0.15) and
        high frequency (0.15 to 0.40) bounds for calculation of frequency
        characteristics of HRV. Default to [0.003,0.4,0.15,0.40]

    Returns
    -------

    """
    # Interpolation
    sfreq = 4
    t = rpeaks
    f = scipy.interpolate.interp1d(t, nni, kind='quadratic')
    t_inter = np.arange(t[0], t[-1], 1 / sfreq)
    nn_inter = f(t_inter)
    noisy_epochs = 0

    if method == 'all':
        absol_power = compute_absol_pow_freq_bands(sfreq, nn_inter[None,], psd_method = psd_method,
                                                   psd_params=psd_params,
                                                freq_bands=frequency_bands).squeeze()
        vlf, lf, hf = absol_power[0], absol_power[1], absol_power[2]
    elif method =='five':
        epochs_hr = []
        time = np.arange(t_inter[0],t_inter[-1],60*5)
        for lb,ub in zip(time[:-1],time[1:]):
            if np.sum(np.bitwise_and(rpeaks>lb,rpeaks<ub)) <181:
                noisy_epochs = noisy_epochs+1
            else:
                epochs_hr.append(nn_inter[np.bitwise_and(t_inter>lb,
                                                         t_inter<ub)])
        if not epochs_hr:
            print('All epochs are rejected')
        non_noisy_epochs = np.vstack(epochs_hr)
        N = np.shape(non_noisy_epochs)[1]
        absol_power = compute_absol_pow_freq_bands(sfreq, non_noisy_epochs,
                                                   psd_method=psd_method,
                                                   psd_params={'welch_n_fft':N
                                                                   },
                                                   freq_bands=frequency_bands).squeeze()
        absol_power = np.mean(absol_power,axis=0)
        vlf, lf, hf = absol_power[0], absol_power[1], absol_power[2]
    else:
        raise NotImplementedError('Method needs to be "five" or "all", '
                                  'got {} instead'.format(method))
    ratio_lf = lf/(lf+hf)
    ratio_hf = hf / (lf + hf)
    total_power = vlf+lf+hf
    return vlf, lf, hf, ratio_lf, ratio_hf, total_power, noisy_epochs

def noisy_interval(rpeaks,fs):
    """
    Find noisy rpeaks. Noisy rpeaks is defined using either 1) improbable
    heart rate values or 2) ecto-beats

    Parameters
    ----------
    rpeaks : list
        R-peaks locations
    fs : int
        Sampling frequency

    Returns
    -------
    outlier_beat : list
        Indexes of noisy beats
    ecto_beat : list
        Indexes of ecto beats
    """
    # Calculate r to r interval
    rri = np.diff(rpeaks) / fs
    ecto_beat = np.zeros((1,len(rri))).squeeze()
    outlier_beat = np.zeros((1,len(rri))).squeeze()

    #### outliers r_peaks
    idx_correct_rri, idx_noisy_rri = _outlier_rpeaks(rri, high_bpm=180,
                                                     low_bpm=20)
    noise_indexes = []
    noise_indexes.append(idx_noisy_rri)
    for x in np.arange(-3,3): #remove 3 beats before and after to be safe
        noise_indexes.append(idx_noisy_rri+x)
    noise_indexes = np.unique(np.hstack(noise_indexes))
    noise_indexes = noise_indexes[noise_indexes<len(rri)-1]
    outlier_beat[noise_indexes] = 1
    #### ecto beats
    idx_nrri, idx_ecto_beats = _ecto_beat(rri)
    ecto_beat[idx_ecto_beats] = 1

    return outlier_beat,ecto_beat

def _outlier_rpeaks(rri, high_bpm=100, low_bpm=40):
    """
    Find and return arguments of rpeaks suspected to be outliers. Outlier
    detection is based on a HR more than 100 or lower than 40.

    Parameters
    ----------
    rri : list
        list of r-r intervals
    high_bpm : int
        Upper bound of acceptable heart rate (default 100)
    low_bpm : int
        Lower bound of acceptable heart rate (default 20)

    Returns
    -------
    clean : list
        Locations of clean R-peaks
    noisy : list
        Locations of noisy R-peaks
    """
    """Remove people with a heart rate less than 40 BPM and more than 100 BPM"""
    low_interval =  60 / high_bpm
    high_interval =  60 / low_bpm
    arg_noisy = np.bitwise_or(rri > high_interval, rri < low_interval)
    clean = np.argwhere(~arg_noisy).ravel()
    noisy = np.argwhere(arg_noisy).squeeze()
    return clean, noisy

def _ecto_beat(rri):
    """Remove Ectopic heartbeats, defined as a variation of more
    than 20% from previous r-r interval"""
    ecto_beat = []
    normal_beat = []
    for count, (rri_t, rri_t1) in enumerate(zip(rri[:-1], rri[1:])):
        if abs(rri_t - rri_t1) > 0.2 * rri_t:
            ecto_beat.append(count)
        else:
            normal_beat.append(count)
    return np.hstack(normal_beat),np.hstack(ecto_beat)

def QRS_detection(sig, fs, verbose=None):
    """QRS detection algorithm.

    This is a mix between an the QRS function implemented
    in MNE and the one implemented in this paper [1]. In-house validation
    of the QRS detection algorithm on the MIT-BIT arrhythmia
    database from PhysioNet yielded a sensitivity of 97% (SD, 3.7%)
    and a specificity of 99% (SD, 2%).

    References
    ----------

    [1] Kadambe, S., Murray, R., & Boudreaux-Bartels, G. F. (1999).
    Wavelet transform-based QRS complex detector.
    IEEE Trans Biomed Eng, 46(7), 838-848. doi:10.1109/10.771194
    """
    ecg = sig
    sample_rate = fs
    swt_level = 3
    swt_ecg = pywt.swt(ecg, 'db3', start_level=0, level=swt_level,
                       trim_approx=True)
    swt_ecg = np.array(swt_ecg[1])
    squared = swt_ecg * swt_ecg
    f1 = 0.3 / sample_rate
    f2 = 10/ sample_rate
    b, a = signal.butter(3, [f1 * 2, f2 * 2], btype='bandpass')
    filtered_squared = signal.filtfilt(b, a, squared)
    uncertain_window = int(0.33 * sample_rate)
    args,peaks = signal.find_peaks(filtered_squared,height=0.001, # we need
                                   # to find better than that 0.0001 threshold
                                   distance=sample_rate/4)
    args = args[args>uncertain_window]
    rms = []
    QRS_peak = []
    for x in args:
        data_p = ecg[x-uncertain_window:x+uncertain_window]
        QRS_peak.append(np.argmax(data_p).ravel() +x-uncertain_window)
        rms.append(compute_rms_value_epochs(data_p[None,]))
    qrs = np.hstack(QRS_peak)
    return qrs