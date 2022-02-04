"""
This modules performs spectral analysis of EEG signals (sometimes
refered as quantitative EEG) on a mne object. EEG Sleep EEG is ideally suited
to frequency and time-frequency analysis, since different stages or
micro-elements (such as spindles, K-complexes, slow waves) have
specific frequency characteristics [1].

Three spectral analysis methods can be used for the analysis, Fast Fourier
transform, Welch and Multitaper spectrogram. Multitaper estimation tends to
be slightly better in reducing artefactual noise and is thus prefered. For an
in depth application of Multitaper analysis to EEG signals, please see [2].

This module can also be used to summarised spectral quantities overnight. For
example, absolute delta power can be calculated in each sleep stages. More
experimental metrics, such as spectral entropy of delta activity across the
night [2], are also implemented.

The code below has been also used to analyse event-related changes in EEG.
The following publications have used part of this code [3,4,5,6], and we refer
interested reader to this publication for further details on implementation
technicals.

[1] Lechat, B., Scott, H., Naik, G. et al (2021). New and Emerging Approaches
to Better Define Sleep Disruption and Its Consequences.
Frontiers in Neuroscience, 15. doi:10.3389/fnins.2021.751730

[2] Prerau, M. J., Brown, R. E., Bianchi, M. T., Ellenbogen, J. M., & Purdon,
P. L. (2017). Sleep Neurophysiological Dynamics Through the Lens of
Multitaper Spectral Analysis. Physiology (Bethesda), 32(1),
60-92. doi:10.1152/physiol.00062.2015

[3] Lechat, B., Hansen, K. L., Melaku, Y. A., Vakulin, A., Micic, G.,
Adams, R. J., . . . Zajamsek, B. (2021). A Novel EEG Derived Measure of
Disrupted Delta Wave Activity during Sleep Predicts All-Cause Mortality Risk.
Ann Am Thorac Soc, (in press). doi:10.1513/AnnalsATS.202103-315OC

[4] Scott, H., Lechat, B., Lovato, N., & Lack, L. (2020).
Correspondence between physiological and behavioural responses
to vibratory stimuli during the sleep onset period: A quantitative
electroencephalography analysis. J Sleep Res, e13232. doi:10.1111/jsr.13232

[5] Sweetman, A., Lechat, B., Catcheside, P. G. et al. (2021). Polysomnographic
Predictors of Treatment Response to Cognitive Behavioral Therapy for
Insomnia in Participants With Co-morbid Insomnia and Sleep Apnea:
Secondary Analysis of a Randomized Controlled Trial.
Frontiers in Psychology, 12. doi:10.3389/fpsyg.2021.676763

[6] Dunbar, C., Catcheside, P., Lechat, B., Hansen, K., Zajamske, B.,
Liebich, T., Nguyen, D.P., Scott, H., Lack, L., Decup, F., Vakulin, A.,
Micic, G., EEG power spectral responses to wind farm compared to road
traffic noise during sleep: A laboratory study. (in press) J Sleep Res,
"""

import mne
import os
import numpy as np
import pandas as pd
import warnings
from psga.analyses.utils import check_is_fitted
from psga.analyses.base import BaseMethods
from psga.features.utils import power_spectrum, _psd_params_checker
from psga.hypnogram import _convert_hypno
from psga.features.time_features import compute_maximum_value_epochs, \
    compute_ptp_amp, \
    compute_rms_value_epochs, compute_std, compute_zero_crossings, \
    compute_time_mass, \
    compute_hjorth, compute_ratio_energy_time_mass
from psga.features.spectral_features import compute_absol_pow_freq_bands, \
    compute_relative_pow_ratios, \
    compute_hjorth_spect, compute_spect_entropy, \
    compute_spect_slope, compute_spect_edge_freq
from psga.features.denoising_function import moving_average_weighted
import sys

try:
    wd = sys._MEIPASS
except AttributeError:
    wd = os.path.dirname(__file__)

FREQ_INIT = {'Delta': [0.5,4.5], 'Theta': [4.5, 7.0], 'Alpha': [7,12],
             'Sigma': [12,16], 'Beta': [16,35]}
PSD_PARAMS_INIT = {'multitaper':
                       {'mt_adaptive': True, 'mt_bandwidth': 1,
                        'mt_low_bias':True},
                   'welch':{'welch_n_fft':256,
                            'welch_n_per_seg':None,
                            'welch_n_overlap':0}}

class qEEG(BaseMethods):
    """Performs quantitative EEG analysis on a mne raw object.

    Power spectrum analysis is computed on consecutive X ('windows_length') of
    raw EEG in the 'score' methods. Mean absolute power of a given frequency
    bands can then be calculated overnight and in specific sleep stage. More
    experimental metrics on the delta frequency bands [1] are also implemented.
    A full list of metrics calculated can be found in XX.

    This class can also be used to perform analysis of qEEG relative to a
    given events in the score_events methods. Given an event dataframe,
    the methods will score qEEG relative to the event onset. For more
    information, please see [2,3].

    Parameters
    ----------
    windows_length : int
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

    Notes
    -----

    References
    -----
    [1] Lechat, B., Hansen, K. L., Melaku, Y. A., Vakulin, A., Micic, G.,
    Adams, R. J., . . . Zajamsek, B. (2021). A Novel EEG Derived Measure of
    Disrupted Delta Wave Activity during Sleep Predicts All-Cause Mortality Risk.
    Ann Am Thorac Soc, (in press). doi:10.1513/AnnalsATS.202103-315OC

    [2] Scott, H., Lechat, B., Lovato, N., & Lack, L. (2020).
    Correspondence between physiological and behavioural responses to
    vibratory stimuli during the sleep onset period: A quantitative
    electroencephalography analysis. J Sleep Res, e13232. doi:10.1111/jsr.13232

    [3] Dunbar, C., Catcheside, P., Lechat, B., Hansen, K., Zajamske, B.,
    Liebich, T., Nguyen, D.P., Scott, H., Lack, L., Decup, F., Vakulin, A.,
    Micic, G., EEG power spectral responses to wind farm compared to road
    traffic noise during sleep: A laboratory study. (in press) J Sleep Res,

    """
    def __init__(self, windows_length = 5, psd_method = 'multitaper',
                 events_windows_length = 5, events_lower_bound = -20,
                 events_upper_bound = 20,
                 ):
        self.freq_bands = np.hstack([FREQ_INIT['Delta'][0],
                                    FREQ_INIT['Delta'][1],
                                    FREQ_INIT['Theta'][1],
                                    FREQ_INIT['Alpha'][1],
                                    FREQ_INIT['Sigma'][1],
                                    FREQ_INIT['Beta'][1]])
        self.psd_method = psd_method
        self.psd_params = PSD_PARAMS_INIT[psd_method]
        self.windows_length = windows_length
        self.events_windows_length = events_windows_length
        self.events_lower_bound = events_lower_bound
        self.events_upper_bound = events_upper_bound
        self.picks = None
        super().__init__()

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
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(key + ' key is not a valid attribute')

    def fit(self, raw, hypnogram, picks=None):
        self._check_raw(raw)
        self._check_hypno(hypnogram)
        if picks is not None: self.picks = picks
        if self.picks is not None:
            raw = raw.pick_channels(ch_names=picks)
        else:
            raise ValueError('No EEG channel was selected for qEEG analysis.')
        self._raw = raw.filter(l_freq=0.3, h_freq=35, verbose='error')
        self._hypno = _convert_hypno(hypnogram, self.windows_length)

    def score(self):
        """Calculate power spectrum based metrics for each segment of length windows_size

        Notes
        -----
        The following parameters are calculated for each segments and for each EEG channel:
        - Absolute and relative power of delta,theta, alpha, sigma, and beta
        bands
        - 'Delta/alpha ratio, slowing ratio and REM ratio
        - Maximum, RMS, SD and peak-to-peak values of EEG epochs data
        - zero crossing rate of each EEG epochs
        - Spectral entropy and Spectral edges (q =0.85 and 0.95)
        """
        check_is_fitted(self, ['_raw', '_hypno'])
        hypno = self._hypno
        raw = self._raw
        for channel in raw.info['ch_names']:
            self._scoring[channel] = _score_qEEG(
                raw, hypno, channel, tmin=0,
                tmax=hypno['duration'].values[0], psd_method=self.psd_method,
                psd_params=self.psd_params, freq_bands=self.freq_bands)
        return self._scoring, self._epochs_data

    def overnight_metrics(self, kdftype='lct2020'):
        """Calculate summary descriptive metrics of an overnight.

        Calculate the mean of each metrics calculated in "qEEG.score()" for
        individual sleep stages. More experimental metrics on the delta
        frequency bands [1] are also implemented. A full list of metrics
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
            #if -1 in np.unique(df.SleepStage.values):  is_scored = False
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
            metrics = {**metrics,**{channel + k: v for k, v in m.items()}}
        return metrics

    def score_from_events(self, events):
        """Calculate power spectrum based metrics for each segment of length windows_size

        Cut raw EEG from "before_event" to "after_event" in epochs of size
        "len_windows" and calculate a range of temporal and spectrum based
        metrics.

        Parameters
        ----------
        event_file : pd.Dataframe
            Dataframe containing onset, duration and labels of events data.
        """
        hypno = self._hypno
        raw = self._raw
        metrics = {}
        for channel in raw.info['ch_names']:
            ev_dict = {}
            for count, tmin in enumerate(
                    np.arange(self.events_lower_bound, self.events_upper_bound,
                              self.events_windows_length)):
                tmax = tmin + self.events_windows_length
                temp_dict = _score_qEEG(raw, events, channel, tmin=tmin,
                                        tmax=tmax, type='event',
                                        psd_method=self.psd_method,
                                        psd_params=self.psd_params,
                                        freq_bands=self.freq_bands
                                        )
                for key, val in temp_dict.items():
                    ev_dict[str(tmin) + 'W_' + key] = val
                if count == 0:
                    event_stage = add_label_to_events(events, hypno)
                    ev_dict['Event_Label'] = event_stage['label'].values
                    ev_dict['Event_Onset'] = event_stage['onset'].values
                    ev_dict['Event_Sleep_Stage'] = event_stage['stage_label']
            metrics[channel] = ev_dict
        return metrics

def _score_qEEG(raw, Stages, channel, tmin=0, tmax=5, type='stage',
                psd_method=None, psd_params=None,
                freq_bands=None):
    ###### MNE needs an array type in points, not seconds ###########
    onset = np.asarray(Stages['onset'].values * raw.info['sfreq'], dtype='int')
    dur = np.asarray(Stages['duration'].values * raw.info['sfreq'], dtype='int')
    label = np.ones_like(Stages['duration'].values, dtype='int')
    events = np.vstack((onset, dur, label)).T

    ################## Get epoch data ###################
    epochs = mne.Epochs(raw, events, picks=[channel], event_id=None, tmin=tmin,
                        tmax=tmax,
                        baseline=(None, None),
                        reject=None, reject_by_annotation=False,
                        verbose='critical', flat=None)
    assert len(epochs.selection) == len(Stages)
    #Stages = Stages.loc[epochs.selection, :]
    #onset = np.asarray(Stages['onset'].values * raw.info['sfreq'], dtype='int')
    #dur = np.asarray(Stages['duration'].values * raw.info['sfreq'],
    # dtype='int')
    data = epochs.get_data().squeeze() * 10 ** 6
    ########## Calculate Epoch Features ###########
    feat_dict = _calculate_epochs_parameters(raw.info['sfreq'], data, psd_method=psd_method,
                                             psd_params=psd_params,
                                             freq_bands=freq_bands)
    if type == 'stage':
        feat_dict['SleepStage'] = Stages['label'].values
        feat_dict['SleepStageOnset'] = onset / raw.info['sfreq']
        feat_dict['SleepStageDuration'] = dur / raw.info['sfreq']
    return feat_dict


def add_label_to_events(events, Stages):
    """
    This function label a sleep stages + ascending or descending phase for each events
    :param onset_event: list of event onsets in sec
    :param Stages: dataframe containing 'onset' colunum, 'label' colunum and
    :return: arguments of events contains within epochs and its label and type of slope
    """
    onset_stages = Stages['onset'].values
    stages_label = Stages['label'].values
    corresponding_stage = []
    for single_onset_events in events['onset'].values:
        # find the preceding stage onset
        index_of_preceding_stage = \
        np.argwhere(onset_stages < single_onset_events)[-1]
        if int(single_onset_events - onset_stages[
            index_of_preceding_stage]) < 15:
            index_of_preceding_stage = index_of_preceding_stage - 1
        corresponding_stage.append(stages_label[index_of_preceding_stage])
    events['stage_label'] = np.hstack(corresponding_stage)
    return events

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
        #compute_hjorth_spect(sfreq, data, precomputed_psd=precomputed_psd),
        compute_spect_entropy(sfreq, data, precomputed_psd=precomputed_psd),
        # compute_spect_slope(sfreq, data,fmin=0.1,fmax=35, precomputed_psd=precomputed_psd),
        compute_spect_edge_freq(sfreq, data, precomputed_psd=precomputed_psd,
                                edge=[0.85, 0.95])
    ))
    spec_feature_names = [#'Hjort_Spect_mobility', 'Hjort_Spect_complexity',
                          'Spectral_Entropy',
                          'Spectral_Edge_85',
                          'Spectral_Edge_95']
    ### Calculate range of temporal feature of epoched data ########
    temporal_features = np.column_stack((
        compute_maximum_value_epochs(data),
        compute_rms_value_epochs(data),
        compute_ptp_amp(data),
        compute_std(data),
        compute_zero_crossings(data),
        #compute_hjorth(data),
    ))
    t_feature_names = ['Max_Val', 'RMS_Val', 'PTP_Amp', 'std', 'ZCR',
                       #'Hjorth_Activity', 'Hjorth_Mobility',
                       #'Hjorth_Complexity'
                       ]
    #### Pool feature together ####
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
