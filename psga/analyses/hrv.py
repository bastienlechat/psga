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

    "score" function find R-peaks and calculate normal to normal beats interval (nni). "score_events" gives the
    5 preceding and 15 following NNI from an event onset. Overnight metrics calculate usual heart rate variability
    (see [1]) markers over the all overnight period. We also calculate some temporal markers of heart rate
    variability for individual sleep stages (can not be done for frequency markers since it requires at least
    5 minutes of continuous NNI)

    Parameters
    ----------

    """

    #CONFIG_PATH = os.path.join(wd, 'hrv_params.yaml')

    def __init__(self, welch_n_fft = 2048, welch_n_overlap=1024,
                 frequency_bands = [0.003, 0.04, 0.15, 0.40]):
        self.psd_method = 'welch'
        self.psd_params = {'welch_n_fft':welch_n_fft,
                           'welch_n_overlap':welch_n_overlap}
        self.frequency_bands = frequency_bands
        super().__init__()

    def fit(self, raw, hypnogram, ECG_chan=None):
        assert (ECG_chan is not None or len(raw.info['ch_names']) == 0)
        if ECG_chan is None: ECG_chan = raw.info['ch_names'][0]
        self._ECG_chan = ECG_chan
        self._check_raw(raw)
        self._check_hypno(hypnogram)

        self.fs = raw.info['sfreq']
        _ecg = raw[ECG_chan,:][0].squeeze() * 10 **3
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

    def score_from_events(self,event_file):
        """

        Parameters
        ----------
        event_file

        Returns
        -------
        ch_name = self.raw.info['ch_names'][0]

        events = pd.read_excel(event_file)

        assert self.scoring, "R-peaks needs to be scored before they can be " \
                             "matched"

        rpeaks = self.scoring[ch_name]['rpeaks']
        hr = np.divide(60,self.scoring[ch_name]['rri'])
        events_onset = events.onset

        rpeaks_events = np.zeros((len(events_onset),40))
        for event_count,single_event_onset in enumerate(events_onset):
            if len(np.argwhere(rpeaks>single_event_onset))>14:
                if len(np.argwhere(rpeaks<single_event_onset)) >5:
                    args_baseline = np.argwhere(rpeaks<single_event_onset)[-5:]
                    args_after = np.argwhere(rpeaks>single_event_onset)[:15]
                    rp_baseline = rpeaks[args_baseline].squeeze()
                    IBI_baseline = hr[args_baseline].squeeze()
                    rp_signal = rpeaks[args_after].squeeze()
                    IBI_signal = hr[args_after].squeeze()
                    rpeaks_events[event_count,:] = np.hstack([rp_baseline,
                                                              rp_signal,
                                                              IBI_baseline,
                                                              IBI_signal])
        IBI_labels = ['IBI_'+str(k) for k in np.arange(20)-5]
        rpeaks_labels = ['RP_' + str(k) for k in np.arange(20)-5]

        rp_dataframe = pd.DataFrame(rpeaks_events,columns=np.hstack([
            rpeaks_labels,IBI_labels]
        ))
        rp_dataframe['label'] = events['label']
        rp_dataframe['onset'] = events['onset']
        self.scoring_events[ch_name] = rp_dataframe.to_dict(orient='list')
        self.save_dict(self.scoring_events, self.path, score_type='HRVev')
        """



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
    """frequency markers of nn series

    steps:
    1) Interpolate to equally spaced time series
    2) Calculate power spectrum density of nni inteprolated series and get absoulte power of very low (0.003, 0.04Hz)
    , low (0.04, 0.15), high freq (0.15, 0.40) frequency bands.
    3) Calculate a few ratios and total powers
    Parameters
    ----------
    rpeaks
    nni
    psd_method
    psd_params
    frequency_bands

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
    """ Calculate normal to normal interval from R-peaks.

    steps:
    1) get r to r interval
    2) remove outlier rri (more than 100 bpm and less than 40 bpm)
    3) remove ecto beats (see _ecto_beats)
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
    """Remove people with a heart rate less than 40 BPM and more than 100 BPM"""
    low_interval =  60 / high_bpm
    high_interval =  60 / low_bpm
    arg_noisy = np.bitwise_or(rri > high_interval, rri < low_interval)
    return np.argwhere(~arg_noisy).ravel(), np.argwhere(arg_noisy).squeeze()


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

    This is a mix between mne:preprocessing:qrs function and techniques used here [1]

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