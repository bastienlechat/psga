import mne
import os
import numpy as np
import pandas as pd
import warnings
from ..utils import check_is_fitted
from ..base import BaseMethods
from ...features.utils import power_spectrum, _psd_params_checker
#from ...hypnogram import _convert_hypno
from ...features.time_features import compute_maximum_value_epochs, \
    compute_ptp_amp, \
    compute_rms_value_epochs, compute_std, compute_zero_crossings, \
    compute_time_mass, \
    compute_hjorth, compute_ratio_energy_time_mass
from ...features.spectral_features import compute_absol_pow_freq_bands, \
    compute_relative_pow_ratios, \
    compute_hjorth_spect, compute_spect_entropy, \
    compute_spect_slope, compute_spect_edge_freq
from ...features.denoising_function import moving_average_weighted

import yaml

import joblib


def mergeDict(dict1, dict2):
    ''' Merge dictionaries and keep values of common keys in list'''
    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            dict3[key] = np.hstack([value, dict1[key]])
    return dict3


import sys

try:
    wd = sys._MEIPASS
except AttributeError:
    wd = os.path.dirname(__file__)


class qEEG(BaseMethods):
    """Performs quantitative EEG analysis on a mne raw object.

    Power spectrum analysis is computed on consecutive X ('windows_length') of raw EEG in the 'score' methods. Mean
    absolute power of a given frequency bands can then be calculated overnight and in specific sleep stage. More
    experimental metrics on the delta frequency bands [1] are also implemented. A full list of metrics
    calculated can be found in XX.

    Event type is also supported in :py: XX, in which case raw EEG data is segmented relative to an event onset.

    Parameters
    ----------

    raw : mne.Base.io.raw object

    hypnogram : hypnogram class

    path : str or None
    windows_size : int
        Length of analysis windows. Default to 5 sec.

    psd_method : str
        PSD methods, 'welch', 'multitaper' or 'fft'

    psd_params : dict or None
        Optional parameters to be passed to shai.features.utils:power_spectrum. If psd_method = 'welch', psd_params
        should contain the following keys (`welch_n_fft`, `welch_n_per_seg`, `welch_n_overlap`). If psd_method
         = 'multitaper', should contain the following ('mt_bandwidth','mt_adaptive','mt_low_bias'). If None,
         default parameters are used.
    before_event : int
        Time, in seconds relative to event onset, from which to start the analysis.
    after_event : int
        Time, in seconds relative to event onset, from which to stop the analysis.
    len_windows_event : int
        Time, in seconds, of the size of the windows analysis.
    save_results : bool
        If true, will save the results at the given "path"

    Methods
    -------

    Notes
    ------

    """

    CONFIG_PATH = os.path.join(wd, 'qeeg_params.yaml')

    def __init__(self, **kwargs):
        super().__init__()
        self.load_base_config()
        if kwargs: self.set_params(kwargs, check_has_key=True)

    def set_params(self, parameters_dict, check_has_key=False):
        for key, value in parameters_dict.items():
            if key == 'psd_params':
                value = _psd_params_checker(value,
                                            parameters_dict['psd_method'])
            if key == 'freq_bands':
                self.freq_bands = value
                value = np.hstack([value['Delta'][0], value['Delta'][1],
                                   value['Theta'][1], value['Alpha'][1],
                                   value['Sigma'][1], value['Beta'][1]])
                key = '_freq_bands'
            if check_has_key:
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    warnings.warn(key + ' key is not a valid attribute')
            else:
                setattr(self, key, value)

    def load_base_config(self):
        with open(self.CONFIG_PATH, 'r') as stream:
            cfg = yaml.load(stream)
        self.set_params(cfg)

    def fit(self, raw, hypnogram, picks=None, *args, **kwargs):
        self._check_raw(raw)
        self._check_hypno(hypnogram)
        if kwargs: self.set_params(parameters_dict=kwargs, check_has_key=True)
        if picks is not None:
            raw = raw.pick_channels(ch_names=picks)
        self._raw = raw.filter(l_freq=0.3, h_freq=35, verbose='error')
        self._hypno = _convert_hypno(hypnogram, self.windows_length)

    def score(self):
        """Calculate power spectrum based metrics for each segment of length windows_size

        Notes
        -----
        The following parameters are calculated for each segments and for each EEG channel:
        """
        self._scoring = {}
        self._epochs_data = {}
        check_is_fitted(self, ['_raw', '_hypno'])

        hypno = self._hypno
        raw = self._raw

        for channel in raw.info['ch_names']:
            self._scoring[channel] = _score_qEEG(
                raw, hypno, channel, tmin=0,
                tmax=hypno['duration'].values[0], psd_method=self.psd_method,
                psd_params=self.psd_params, freq_bands=self._freq_bands)

        return self._scoring, self._epochs_data

    def overnight_metrics(self, kdftype='lct2020'):
        """Calculate summary descriptive metrics of an overnight.

        Calculate the mean of each metrics calculated in "qEEG.score()" for individual sleep stages. More
        experimental metrics on the delta frequency bands [1] are also implemented. A full list of metrics
        calculated can be found in Notes.

        Notes
        -----
        The following parameters are calculated for each segments:

        """

        if not self._scoring:
            self.score()
        scoring = self._scoring.items()
        metrics = {}
        is_scored = True

        for channel, qeeg_dict in scoring:
            df = pd.DataFrame.from_dict(qeeg_dict)
            st = df.loc[df.Max_Val < 400, :]
            if -1 in np.unique(df.SleepStage.values):  is_scored = False

            if not is_scored: kdftype = 'log'

            if is_scored:
                # by individual sleep stage
                grouped = st.groupby(by='SleepStage').mean().drop(
                    ['SleepStageOnset', 'SleepStageDuration'], axis=1)
                v = grouped.unstack().to_frame().sort_index(level=1).T
                v.columns = [x + '_' + 'N' + str(int(y)) for (x, y) in
                             v.columns]
                # NREM
                nrem = st.drop(['SleepStageOnset', 'SleepStageDuration'],
                               axis=1).mean().to_frame().T
                nrem.columns = [x + '_NREM' for x in nrem.columns]
                grpstaged = pd.concat([v, nrem], axis=1).to_dict(
                    orient='records')[0]
            else:  # no sleep stage to sub-analyse, mean of all
                t = st.drop(['SleepStageOnset', 'SleepStageDuration'],
                            axis=1).mean().to_frame().T
                t.columns = [x + '_mean' for x in t.columns]
                grpstaged = t.to_dict(orient='records')[0]

            delta = df.absolute_delta
            if kdftype == 'lct2020':
                delta[df.SleepStage == 0] = 0
                delta[df.SleepStage == 1] = 0
                delta_smoothed = moving_average_weighted(delta)
                features = _delta_fluctuations_parameters(delta_smoothed)
            elif kdftype == 'log':
                ldelta = np.log(delta + 1)
                ldelta[df.Max_Val > 400] = np.min(ldelta[ldelta != 0])
                ldelta = ldelta - np.min(ldelta)
                ldelta_smoothed = moving_average_weighted(ldelta)
                features = _delta_fluctuations_parameters(ldelta_smoothed)
            else:
                raise NotImplementedError

            m = {**grpstaged, **features}
            metrics[channel] = {channel + k: v for k, v in m.items()}
        return metrics

    def score_from_events(self, event_file):
        """Calculate power spectrum based metrics for each segment of length windows_size

        Cut raw EEG from "before_event" to "after_event" in epochs of size "len_windows" and calculate a range
        of temporal and spectrum based metrics.

        Parameters
        ----------

        event_file : str
        excel file containing event onset (in seconds), the label and duration of each events.

        Notes
        -----
        The following parameters are calculated for each segments:

        """

        events = pd.read_excel(event_file)
        Stages = self.hypnogram.to_df(sleep_onset_offset=True,
                                      windows_size=self.windows_length)

        for channel in self.raw.info['ch_names']:

            ev_dict = {}
            for count, tmin in enumerate(
                    np.arange(self.events_lower_bound, self.events_upper_bound,
                              self.events_windows_length)):
                tmax = tmin + self.events_windows_length

                temp_dict = _score_qEEG(self.raw, events, channel, tmin=tmin,
                                        tmax=tmax, type='event',
                                        psd_method=self.psd_method,
                                        psd_params=self.psd_params,
                                        freq_bands=self._freq_bands
                                        )

                for key, val in temp_dict.items():
                    ev_dict[str(tmin) + 'W_' + key] = val

                if count == 0:
                    event_stage = add_label_to_events(events, Stages)
                    ev_dict['Event_Label'] = event_stage['label'].values
                    ev_dict['Event_Onset'] = event_stage['onset'].values
                    ev_dict['Event_Sleep_Stage'] = event_stage['stage_label']

            self.scoring_events[channel] = ev_dict
        self.save_dict(self.scoring_events, self.path, score_type='qEEGev')


def _score_qEEG(raw, Stages, channel, tmin=0, tmax=5, type='stage',
                psd_method=None, psd_params=None,
                freq_bands=None):
    ###### MNE needs an array type in points, not seconds ###########
    onset = np.asarray(Stages['onset'].values * raw.info['sfreq'], dtype='int')
    dur = np.asarray(Stages['duration'].values * raw.info['sfreq'], dtype='int')
    label_for_mne = np.ones_like(Stages['duration'].values, dtype='int')
    events = np.vstack((onset, dur, label_for_mne)).T

    ################## Get epoch data ###################
    epochs = mne.Epochs(raw, events, picks=[channel], event_id=None, tmin=tmin,
                        tmax=tmax,
                        baseline=(None, None),
                        reject=None, reject_by_annotation=False,
                        verbose='critical', flat=None)
    Stages = Stages.loc[epochs.selection, :]
    onset = np.asarray(Stages['onset'].values * raw.info['sfreq'], dtype='int')
    dur = np.asarray(Stages['duration'].values * raw.info['sfreq'], dtype='int')
    label = Stages['label'].values
    # print(np.shape(Stages))
    data = epochs.get_data().squeeze() * 10 ** 6

    # print(data[:10,:10])
    # print(np.shape(data))
    sfreq = raw.info['sfreq']

    # This check if mne rejected any epochs (and it shouldn't since we set reject = None)
    if _too_many_rejected_epoch(data, label):
        raise ValueError(
            'Sleep stage scoring has epochs finishing/starting before the recording')

    ########## Calculate Epoch Features ###########
    feat_dict = _calculate_epochs_parameters(sfreq, data, psd_method=psd_method,
                                             psd_params=psd_params,
                                             freq_bands=freq_bands)

    if type == 'stage':
        feat_dict['SleepStage'] = label
        feat_dict['SleepStageOnset'] = onset / sfreq
        feat_dict['SleepStageDuration'] = dur / sfreq

    return feat_dict


def add_label_to_events(events, Stages):
    """
    This function label a sleep stages + ascending or descending phase for each events
    :param onset_event: list of event onsets in SECONDE
    :param Stages: dataframe containing 'onset' colunum, 'label' colunum and 'AD' colunum (ascending and descending slope) in SECONDE
    :return: arguments of events contains within epochs and its label and type of slope
    """
    onset_stages = Stages['onset'].values
    stages_label = Stages['label'].values
    onsets_of_all_events = events['onset'].values

    corresponding_stage = []
    baseline_stage = []

    for single_onset_events in onsets_of_all_events:
        # find the preceding stage onset
        index_of_preceding_stage = \
        np.argwhere(onset_stages < single_onset_events)[-1]

        if int(single_onset_events - onset_stages[
            index_of_preceding_stage]) < 15:
            # print(int(single_onset_events - onset_stages[
            # index_of_preceding_stage]))
            index_of_preceding_stage = index_of_preceding_stage - 1

        corresponding_stage.append(stages_label[index_of_preceding_stage])

    events['stage_label'] = np.hstack(corresponding_stage)
    return events


def _too_many_rejected_epoch(data, label):
    if np.shape(data)[0] != len(label):
        n_epoch_dropped = len(label) - np.shape(data)[0]

        if n_epoch_dropped > 0:
            warnings.warn('{} epochs were dropped'.format(n_epoch_dropped))
            return False
        else:
            return True


def _calculate_epochs_parameters(sfreq, data, psd_method='multitaper',
                                 psd_params=None,
                                 freq_bands=None):
    #### Calculate power spectrum density of epoched data ###
    psd, freqs = power_spectrum(sfreq, data, psd_method=psd_method,
                                **psd_params, verbose=False)
    precomputed_psd = {'psd': psd.squeeze(), 'freqs': freqs}

    ### Calculate absolute frequencies of epoched data ########
    absolute_bands = compute_absol_pow_freq_bands(sfreq, data.squeeze(),
                                                  freq_bands=freq_bands,
                                                  precomputed_psd=precomputed_psd)
    freq_name = ['absolute_delta', 'absolute_theta', 'absolute_alpha',
                 'absolute_sigma', 'absolute_beta']

    ### Calculate absolute frequencies of epoched data ########
    relative_bands = compute_relative_pow_ratios(absolute_bands)
    ratio_name = ['deltaR', 'thetaR', 'alphaR', 'sigmaR', 'betaR', 'DAR',
                  'SLOWING_RATIO', 'REMR']

    ### Calculate range of spectral feature of epoched data ########
    spectral_features = np.column_stack((
        compute_hjorth_spect(sfreq, data, precomputed_psd=precomputed_psd),
        compute_spect_entropy(sfreq, data, precomputed_psd=precomputed_psd),
        # compute_spect_slope(sfreq, data,fmin=0.1,fmax=35, precomputed_psd=precomputed_psd),
        compute_spect_edge_freq(sfreq, data, precomputed_psd=precomputed_psd,
                                edge=[0.25, 0.5, 0.75, 0.85, 0.95])
    ))

    spec_feature_names = ['Hjort_Spect_mobility', 'Hjort_Spect_complexity',
                          'Spectral_Entropy',
                          'Spectral_Edge_25', 'Spectral_Edge_50',
                          'Spectral_Edge_75', 'Spectral_Edge_85',
                          'Spectral_Edge_95']

    ### Calculate range of temporal feature of epoched data ########
    temporal_features = np.column_stack((
        compute_maximum_value_epochs(data),
        compute_rms_value_epochs(data),
        compute_ptp_amp(data),
        compute_std(data),
        compute_zero_crossings(data),
        compute_hjorth(data),

    ))
    t_feature_names = ['Max_Val', 'RMS_Val', 'PTP_Amp', 'std', 'ZCR',
                       'Hjorth_Activity', 'Hjorth_Mobility',
                       'Hjorth_Complexity']

    #### Pool feature together and save to excel ####

    features = np.column_stack((absolute_bands, relative_bands,
                                spectral_features, temporal_features))
    feat_name = np.hstack([freq_name, ratio_name,
                           spec_feature_names, t_feature_names])

    feature_dict = {k: v for k, v in zip(feat_name, features.T)}
    return feature_dict


def _delta_fluctuations_parameters(overnight_fluctuations):
    data = np.expand_dims(overnight_fluctuations, axis=0)
    features_delta_band = np.column_stack((
        compute_spect_entropy(2, data, psd_method='fft'),
        compute_spect_edge_freq(2, data, psd_method='fft', edge=[0.75, 0.95]),
        compute_time_mass(data, q=[0.5, 0.75]),
        compute_ratio_energy_time_mass(data, q=[0.5, 0.75]),
        compute_spect_slope(2, data, psd_method='fft', fmin=None, fmax=1)
    ))

    names = ['Spectral_Entropy_kdf', 'Spectral_Edge_75_kdf',
             'Spectral_Edge_95_kdf',
             'TimeMass50_kdf', 'TimeMass75_kdf', 'DER_50_kdf', 'DER_75_kdf',
             'intercept_kdf', 'slope_kdf', 'MSE_kdf', 'R2_kdf']
    features_delta_dict = {k: float(v[0]) for k, v in zip(names,
                                                          features_delta_band.T)}
    return features_delta_dict