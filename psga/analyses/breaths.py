import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.interpolate
import pywt

from psga.features.spectral_features import power_spectrum, \
    compute_absol_pow_freq_bands
from psga.features.time_features import compute_rms_value_epochs, \
    compute_zero_crossings, compute_hjorth
from .base import BaseMethods
import sys
import pandas as pd

try:
    wd = sys._MEIPASS
except AttributeError:
    wd = os.path.dirname(__file__)

class Breathing(BaseMethods):
    """
    This modules performs breath-by-breath analysis. We first find the end
    tidal peak (maximum of inspiration) using scipy.signal.findpeaks and
    find the beginning and end of each respiratory cycles
    using zero-crossing criteria.


    Parameters
    ----------

    """
    def __init__(self):
        super().__init__()

    def fit(self, raw, hypnogram, nasal_pressure=None, flow_chan=None):
        self._check_raw(raw)
        self._check_hypno(hypnogram)
        self.np_ch_name = nasal_pressure
        if not self.np_ch_name: raise ValueError('You need to specify nasal '
                                                 'pressure channel name')
        self._flow_chan = [nasal_pressure]
        if flow_chan: self._flow_chan = [nasal_pressure,*flow_chan]
        self._raw = raw
        self._hypno = hypnogram

    def score(self, plot = False, flow_from='pneumo'):
        raw = self._raw
        flow_chans = self._flow_chan
        sf = self._raw.info['sfreq']
        square_transform_flow=False
        if flow_from=='pneumo': square_transform_flow = True

        for flow_channel_name in flow_chans:
            _flow = raw[flow_channel_name, :][0].squeeze()
            _flow = _flow - np.mean(_flow)
            flow = mne.filter.filter_data(_flow, sf, l_freq=0.1,
                                          h_freq=6, verbose='CRITICAL')
            if square_transform_flow:
                flow[flow < 0] = -1 * np.sqrt(np.abs(flow[flow < 0]))
                flow[flow > 0] = np.sqrt(flow[flow > 0])
                flow = flow
            f2 = 1.2
            b, a = signal.butter(4, f2 * 2, btype='lowpass', fs=sf)
            data = signal.filtfilt(b, a, flow)
            dt = 1 / sf
            start_insp, start_exp, end_exp = breath_detection(data, dt)
            df = breath_features(data, dt, start_insp, start_exp, end_exp)
            df['start_breath'] = np.asarray(start_insp) * dt
            df['start_breath'] = np.asarray(start_exp) * dt
            df['end_breath'] = np.asarray(end_exp) * dt
            if plot:
                plt.plot(data)
                plt.plot(start_insp, data[start_insp], 'b*')
                plt.plot(start_exp, data[start_exp], 'k^')
                plt.plot(end_exp, data[end_exp], 'r^', alpha=0.5)
                plt.show()
            self._scoring[flow_channel_name] = df.to_dict(orient='list')

        return self._scoring, self._epochs_data


def breath_detection(flow, dt):
    """
    This function find the end tidal peak (maximum of inspiration) using
    scipy.signal.findpeaks and find the beginning and end of
    each respiratory cycles using zero-crossing criteria.

    Parameters
    ----------
    flow : np.array_like
        Flow time serie
    dt : float
        sampling time [1/Fs]

    Returns
    -------
    start_insp : np.array
        Start of inspiration (in points)
    start_exp : np.array
        Start of expiration (in points)
    end_ex:
        End of respiratory cycle (in points)
    """
    import scipy.signal
    # Find top of inspiration
    minpeakwidth = (1 / dt) * 0.3
    peakdistance = (1 / dt) * 2
    minPeak = 0.001*np.sqrt(np.mean(np.power(flow,2)))
    args, peaks = scipy.signal.find_peaks(flow, width=minpeakwidth,
                                          height=minPeak,
                                          distance=peakdistance)#,
                                          #prominence=minpeakprominence)
    arg_ma = []
    arg_mi = []
    # Only keep breaths with negative amplitude expiration
    for arg_in, arg_out in zip(args[:-1], args[1:]):
        if np.min(flow[arg_in:arg_out])<0:
            arg_min = np.argmin(flow[arg_in:arg_out]) + arg_in
            arg_ma.append(arg_in)
            arg_mi.append(arg_min)
    # Find beginning and end of breathing cycles
    breaths_start = []
    breaths_end = []
    zero_cross_exp = []
    for arg_bef, arg_curr, arg_next in zip(arg_ma[:-2], arg_ma[1:-1], arg_ma[2:]):
        flow_before = flow[arg_bef:arg_curr]
        flow_after = flow[arg_curr:arg_next]
        zero_cross_before = np.where(np.diff(np.sign(flow_before)))[0]
        zero_cross_after = np.where(np.diff(np.sign(flow_after)))[0]
        if len(zero_cross_before)>0:
            if len(zero_cross_after)>0:
                breaths_start.append(zero_cross_before[-1]+arg_bef)
                breaths_end.append(zero_cross_after[0]+arg_curr)
                zero_cross_exp.append(zero_cross_after)
    start_insp = breaths_start[:-1]
    start_exp = breaths_end[:-1]
    end_exp = breaths_start[1:]

    end_ex = []
    max_exp = 6
    for in_in,in_out,exp_out in zip(start_insp,start_exp,
                                           end_exp):
        in_out = in_out+int((1/dt)*0.1) #this is just because in_out is
        # already the pos before a sign change, so we add a little bit more
        # to be in the negative zone. This will blow up one day
        if (exp_out - in_out)*dt > max_exp:
            flow_after = flow[in_out:in_out+int(max_exp*(1/dt))]
            zero_cross_after = np.where(np.diff(np.sign(flow_after)))[0]  #
            if len(zero_cross_after) > 0:
                end_ex.append(zero_cross_after[0] + in_out)
            else:
                end_ex.append(in_out+int(max_exp*(1/dt)))
        else:
            end_ex.append(exp_out)

    return np.hstack(start_insp), np.hstack(start_exp), np.hstack(end_ex)

def breath_features(flow, dt, breath_in, breath_out, breath_end):
    """
    Calculate features of each breaths. This includes timing (beginning/end,
    max and min) of inspiration and expiration, as well as
    inspiratory/expiratory tidal volume (and minute ventilation). In addition,
    we also calculate RMS value, zero crossing rate, and hjorths parameters of
    each breaths.

    Parameters
    ----------
    flow : np.array
        Flow time series
    dt : float
        Sampling time (1/Fs)
    breath_in : np.array
        indexes (in point) of inspiration start
    breath_out : np.array
        indexes (in point) of expiration start
    breath_end : np.array
        indexes (in point) of end of respiratory cycle

    Returns
    -------
    resp : pd.DataFrame
        Dataframe containing start, middle and end of each respiratory
        cycles, as well as key features of each breaths.
    """
    volumes = np.cumsum(flow/60)*dt
    VTi = volumes[breath_out]-volumes[breath_in] #Inspiratory tidal volume
    VTe = volumes[breath_end]-volumes[breath_out] #Expiratory tidal volume
    Ti = np.subtract(breath_out, breath_in) * dt #Inspiratory time
    Te = np.subtract(breath_end, breath_out) * dt #Expiratory time
    Ttot= Ti + Te #Total breath time
    bf = (np.ones_like(Ttot)*60)/Ttot # breathing frequency
    VI = VTi * bf # inspiratory minute ventilation
    VE = VTe * bf #expiratory minute ventilaton
    feats = []
    for count,(start_cycle, end_cycle) in enumerate(zip(breath_in, breath_end)):
        resp_cycle = np.asarray(flow[start_cycle:end_cycle])
        flow_min_time = (np.argmin(resp_cycle) + start_cycle)*dt
        flow_min_val = np.min(resp_cycle)
        flow_max_time = (np.argmax(resp_cycle) + start_cycle)*dt
        flow_max_val = np.max(resp_cycle)
        rms = compute_rms_value_epochs(resp_cycle[None, :])[0]
        zcr = compute_zero_crossings(resp_cycle[None, :])[0]
        activity, mobility, complexity = compute_hjorth(resp_cycle[None, :])[0]
        feats.append([flow_min_time,flow_min_val,flow_max_time,flow_max_val,
                      rms, zcr,activity, mobility, complexity])
    feat_breath = np.vstack(feats)
    bin = np.array(breath_in)*dt
    bout = np.array(breath_out)*dt
    bend = np.array(breath_end)*dt
    other_feats = np.transpose(np.vstack([bin, bout,
                                          bend, VTi, VTe,
                                          Ti, Te, Ttot, bf, VI,
                                          VE]))
    labels = ['StartInsp','StartExp','EndCycle','VTi', 'VTe', 'Ti', 'Te',
              'Ttot', 'bf','VI','VE',
              'flow_min_time', 'flow_min_val', 'flow_max_time', 'flow_max_val',
              'rms', 'zcr', 'activity', 'mobility', 'complexity'
              ]
    resp = pd.DataFrame(data = np.hstack([other_feats, feat_breath]), columns =
    labels)
    return resp