import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from .base import BaseMethods
from .utils import get_rpeaks
import sys
import pandas as pd

try:
    wd = sys._MEIPASS
except AttributeError:
    wd = os.path.dirname(__file__)

class PWA(BaseMethods):
    """Performs analyses related to pulse wave amplitude signals.

    Maximum, minimum, beginning and end of each pulse wave amplitude waveform
    is found using existing R-peaks from the ECG. The corresponding pulse
    arrival time values (PAT; the delay between R-peaks and peripheral site)
    is calculated using the maximum of second derivative of PWA as the
    peripheral point, as done previously [1].

    This class can also be used to perform analysis of PWA/PAT relative to a
    given events in the score_events methods. Given an event onset,
    the methods will store the 5 preceeding and 15 following PAT/PWA maximum
    and to the  event onset. For more information, please see [2]

    Notes
    -----

    References
    -----
    [1] Kwon Y, Wiles C, Parker BE, Clark BR, Sohn MW, Mariani S, et al.
    Pulse arrival time, a novel sleep cardiovascular marker:
    the multi-ethnic study of atherosclerosis. Thorax. 2021.

    [2] TBA

    """

    def __init__(self):
        super().__init__()

    def fit(self, raw, hypnogram, ECG_chan=None, PWA_chan=None,**kwargs):
        self._check_raw(raw)
        self._check_hypno(hypnogram)
        self._ECG_chan = ECG_chan
        self._PWA_chan = PWA_chan
        self._raw = raw
        self._hypno = hypnogram
        self._ecg = raw[ECG_chan,:][0].squeeze()*10**3
        self._pwa = raw[PWA_chan,:][0].squeeze()*10**3

    def score(self, plot = False):
        """
        Maximum, minimum, beginning and end of each pulse wave amplitude waveform
        is found using existing R-peaks from the ECG. The corresponding pulse
        arrival time values (PAT; the delay between R-peaks and peripheral site)
        is calso calculated.

        Parameters
        ----------
        plot : bool
            If True, a summary plot of the detection will be plotted. Default is
            False.

        Returns
        -------
        scoring : dict
            A dictionnary containing beginning, max, min and end values of
            each PWA waveforms as well as the corresponding r-peaks locations
            used for the analysis and the corresponding pulse arrival time
            values.
        epoch_data : dict
            Empty dict, this is just for consistency across the "score" methods.
        """
        self._scoring = {}
        self._epochs_data = {}

        #Score r-peaks for PWA max detection
        rpscore = get_rpeaks(self._raw, self._hypno,
                                 ECG_chan=self._ECG_chan)
        noisy = rpscore[self._ECG_chan]['Noisy']
        rp = rpscore[self._ECG_chan]['rpeaks'][noisy == 0] #remove noisy rpeaks
        sf = self._raw.info['sfreq']
        rpeaks_loc = np.asarray(sf*rp, dtype=np.int)
        pleth = mne.filter.filter_data(self._pwa, sf, l_freq=2,
                                       h_freq=20, verbose='CRITICAL')
        # Find max/min of pulse waves and calculate PAT
        start_p,max_p,min_p,stop_p = self._get_pwa_pulse(pleth,rpeaks_loc,sf)
        rpeaks_loc = rpeaks_loc[1:-1]
        times, ptt = self._get_ptt(pleth, rpeaks_loc, sf)
        ptt = ptt[:-1]
        rpeaks_loc = rpeaks_loc[:-1]
        vals = np.vstack([start_p,max_p,pleth[max_p],min_p,pleth[min_p],stop_p,
                          ptt,
                          rpeaks_loc]).T
        vals = vals[~np.isnan(vals).any(axis=1)]

        if plot:
            pleth = mne.filter.filter_data(pleth, sf, l_freq=2,
                                           h_freq=20, verbose='CRITICAL')
            plt.plot(pleth)
            for count, row_val in enumerate(vals):
                if count % 5 == 0:
                    row_val = row_val.astype('int')
                    plt.plot(row_val[0],pleth[row_val[0]],'r*')
                    plt.plot(row_val[1], pleth[row_val[1]], 'm^')
                    plt.plot(row_val[3], pleth[row_val[3]], 'y^')
                    plt.plot(row_val[5], pleth[row_val[5]], 'g*')
            plt.show()

        self._scoring[self._PWA_chan] = {'PWA_start': vals[:,0]/sf,
                                         'PWA_max_pos': vals[:, 1]/sf,
                                         'PWA_max_amp': vals[:, 2],
                                         'PWA_min_pos': vals[:, 3]/sf,
                                         'PWA_min_amp': vals[:, 4],
                                         'PWA_stop': vals[:, 5],
                                         'PTT': vals[:,6],
                                         'Rpeaks_pos': vals[:,7]/sf,
                                         }
        return self._scoring, self._epochs_data

    def overnight_metrics(self):
        """
        Calculate overnight summary metrics of a pulse arrival time series.
        This analysis is similar to frequency analysis of heart-rate
        variability. Please see :py:psga.hrv.frequency_markers
        """
        from .hrv import frequency_markers_rr
        psd_method = 'welch'
        psd_params = {'welch_n_fft': 2048, 'welch_n_overlap': 1024}

        vlf, lf, hf, ratio_lf, ratio_hf, total_power = frequency_markers_rr(
            self._scoring['PWA_start'], self._scoring['PTT'],
            psd_method=psd_method, psd_params=psd_params
        )

        return {'vlf_ptt': vlf, 'lf_ptt': lf, 'hf_ptt': hf, 'lf_ratio_ptt':
            ratio_lf, 'hf_ratio_ptt': ratio_hf, 'total_power_ptt': total_power}

    def score_from_events(self,events):
        """
        Locates the 5 preceding and 15 following pulse wave waveform from an
        event onset. The corresponding maximum PWA and PTT values are appended.
        This can be useful to events related analysis. This
        was done recently in [1].

        Parameters
        ----------
        events : pd.DataFrame

        Returns
        -------
        rp_dataframe : pd.Dataframe
            Dataframe containing PWA/PTT events scoring

        References
        ----------
        [1] TBA
        """
        assert self._scoring, "R-peaks needs to be scored before they can be " \
                             "matched"

        events_onset = events.onset
        onsets = np.asarray(self._scoring['PWA_start'])
        ptt_values = np.asarray(self._scoring['PTT'])
        pwa_values = np.asarray(self._scoring['PWA_max_amp'])

        events_data = np.zeros((len(events_onset), 60))
        for event_count, single_event_onset in enumerate(events_onset):
            if len(np.argwhere(onsets > single_event_onset)) > 14:
                if len(np.argwhere(onsets < single_event_onset)) > 5:
                    args_baseline = np.argwhere(onsets < single_event_onset)[
                                    -5:]
                    args_after = np.argwhere(onsets > single_event_onset)[:15]
                    # Locations pre-post events
                    rp_baseline = onsets[args_baseline].squeeze()
                    rp_signal = onsets[args_after].squeeze()
                    # PTT values
                    PTT_baseline = ptt_values[args_baseline].squeeze()
                    PTT_signal = ptt_values[args_after].squeeze()
                    #PWA maximum values
                    PWA_baseline = pwa_values[args_baseline].squeeze()
                    PWA_signal = pwa_values[args_after].squeeze()

                    events_data[event_count, :] = np.hstack([rp_baseline,
                                                            rp_signal,
                                                            PTT_baseline,
                                                            PTT_signal,
                                                            PWA_baseline,
                                                            PWA_signal
                                                            ])
        PTT_labels = ['PTT_' + str(k) for k in np.arange(20) - 5]
        PWA_labels = ['PTT_' + str(k) for k in np.arange(20) - 5]
        rpeaks_labels = ['RP_' + str(k) for k in np.arange(20) - 5]

        rp_dataframe = pd.DataFrame(events_data, columns=np.hstack([
            rpeaks_labels, PTT_labels, PWA_labels]
        ))
        rp_dataframe['label'] = events['label']
        rp_dataframe['onset'] = events['onset']

        return rp_dataframe

    def _get_pwa_pulse(self, pleth, rpeaks, sf):
        """
        Find the beginning, minimum, maximum and end of a pulse wave
        amplitude waveform.

        Parameters
        ----------
        pleth : array
            Pulse wave amplitude signal
        rpeaks : list
            Locations of R-peaks in the ECG
        sf : int
            Sampling frequency

        Returns
        -------
        pleth_start, pleth_max, pleth_min, pleth_end : array (N)
            Locations of the beginning, maximum, minimum and end values of
            each pulse wave amplitude waveform. N waveform were detected.
        """
        # Get maximum of pulse wave
        max_pleth_arg = []
        for r1,r2 in zip(rpeaks[:-1],rpeaks[1:]):
            max_pleth_arg.append(np.argmax(pleth[r1:r2]) + r1)
        max_pleth_arg = np.hstack(max_pleth_arg)
        # Get minimum in pulse wave - minimum is between two maximum
        min_arg_pleth = []
        for p1, p2 in zip(max_pleth_arg[:-1], max_pleth_arg[1:]):
            min_arg_pleth.append(np.argmin(pleth[p1:p2]) + p1)
        # Find beginning and end of pulse wave using zero-crossing criteria
        pleth_start = []
        pleth_end = []
        for arg_bef, arg_curr, arg_next in zip(max_pleth_arg[:-2],
                                               max_pleth_arg[1:-1],
                                               max_pleth_arg[2:]):
            pleth_before = pleth[arg_bef:arg_curr]
            pleth_after = pleth[arg_curr:arg_next]
            zero_cross_before = np.where(np.diff(np.sign(pleth_before)))[0]  #
            zero_cross_after = np.where(np.diff(np.sign(pleth_after)))[0]
            s = np.nan
            st = np.nan
            if len(zero_cross_before) > 0: s = int(zero_cross_before[-1] + \
                                               arg_bef)
            if len(zero_cross_after) > 0: st = int(zero_cross_after[-1] +
                                                   arg_curr)
            pleth_start.append(s)
            pleth_end.append(st)

        return np.hstack(pleth_start), max_pleth_arg[1:-1], \
               min_arg_pleth[1:], np.hstack(pleth_end)

    def _get_ptt(self,pleth, rpeaks,sfreq):
        """
        Calculate pulse arrival time from a series of r-peaks and a pulse
        wave amplitude series. The methods is described here [1].

        Briefly, the first differential of the PWA signal is filtered
        using a 0.13 s moving- average filter. The differential is repeated
        on the positive values of the filtered PWA signal. Finally, the maximum
        value of the 2nd derivatives is determined, and the time between the
        R-wave peak is calculated. This time difference is called PAT.

        Parameters
        ----------
        pleth : ndarray
            Pulse wave amplitude signal
        rpeaks : list
            locations of the rpeaks
        sfreq : int
            Sampling frequency

        Returns
        -------
        times : array
            Location of pulse arrival times (in sec)
        pat : array
            Corresponding pulse arrival time values

        References
        ----------
        [1]	Kwon Y, Wiles C, Parker BE, Clark BR, Sohn MW, Mariani S, et al.
        Pulse arrival time, a novel sleep cardiovascular marker:
        the multi-ethnic study of atherosclerosis. Thorax. 2021.
        """
        from scipy.ndimage.filters import uniform_filter1d
        window_n = int(0.13*sfreq)
        pleth_der = np.diff(pleth)
        pleth_der_filtered = uniform_filter1d(pleth_der, size=window_n)
        pleth_der_filtered[pleth_der_filtered<0] = 0
        plethderder = np.diff(pleth_der_filtered)
        # Get maximum of 1st derivative of pulse wave
        max_derpleth_arg = []
        for r1, r2 in zip(rpeaks[:-1], rpeaks[1:]):
            max_derpleth_arg.append(np.argmax(plethderder[r1:r2]) + r1)
        max_derpleth_arg = np.hstack(max_derpleth_arg)
        # Calculate PAT values
        pat = []
        times = []
        for rp in rpeaks:
            if any(max_derpleth_arg>rp):
                args_pleth_point = max_derpleth_arg[
                    rp<max_derpleth_arg][0]
                pat.append((args_pleth_point-rp)/sfreq)
                times.append(rp)
            else:
                pat.append(np.nan)
                times.append(np.nan)
        return np.hstack(times),np.hstack(pat)
