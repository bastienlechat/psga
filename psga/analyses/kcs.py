import mne
import os
import numpy as np
import pandas as pd
#from .kcmodel import scoring_algorithm_kc
from ..features.spectral_features import compute_absol_pow_freq_bands
from .base import BaseMethods
import sys
from scipy.signal import find_peaks
import pywt
import joblib

try:
    wd = sys._MEIPASS
except AttributeError:
    wd = os.path.dirname(__file__)

try:
    import torch
    import torch.jit
    #torch.jit.script_method = script_method
    #torch.jit.script = script
except ImportError:
    print(ImportError)

try:
    import gpytorch
    from gpytorch.variational import CholeskyVariationalDistribution
    from gpytorch.variational import WhitenedVariationalStrategy
except ImportError:
    print(ImportError)


class KC(BaseMethods):
    """
    ... one line comment

    ...

    Parameters
    ----------

    raw : mne.Base.io.raw object

    hypnogram : hypnogram class

    Methods
    -------

    """

    def __init__(self, include_stages = 'all', **kwargs):

        super().__init__()
        self._include_stages = include_stages
        if include_stages =='all': self._include_stages = [-1,0,1,2,3,4,5,9]
        self._epochs_data = {}
        self._metadata = {}
        self._scoring = {}

    def fit(self, raw, hypnogram, picks=None, events=None,**kwargs):
        """

        Parameters
        ----------
        raw
        hypnogram
        path
        picks
        events
        kwargs

        Returns
        -------

        """
        self._check_raw(raw)
        self._check_hypno(hypnogram)
        if kwargs: self.set_params(parameters_dict=kwargs, check_has_key=True)
        if picks is not None:
            raw = raw.pick_channels(ch_names=picks)

        self._raw = raw.filter(l_freq=0.3,h_freq=None, verbose = 'error')
        self._hypno = hypnogram

    def score(self):
        """ Score K-complexes and calculate characteristics K-complexes parameters.

        More informations about the scoring algorithm can be found in [1] and in :py:func:`~SHAI.EEG.KCscoring.model`.
        Results (scoring + metrics) is stored in kc_scoring dict. Some metrics are scored according to [2].

        Parameters
        ----------

        Returns
        -------
        output : ndarray, shape (n_channels * n_edge,)
            With: `n_edge = 1` if `edge` is None or `n_edge = len(edge)` otherwise.

        Notes
        -----
        The following parameters are calculated for each K-complexes:

         KC_onset : onset, in seconds from the beginning of the recordings, of the KC
         KC_probas : probability of the K-complex
         KC_stage : sleep stage of the K-complex
         N550 : Amplitude of the N550 components, in uv
         P900 : Amplitude of the P900 components, in uv
         PTP : Peak to peak amplitude of the KC, in uv
         Slope : K-complex slope, define as (P900-N550)/(tP900-tN550), in uv/sec
         dt_P9_N5 : Time tP900-tN550, in seconds
         baseline_delta: absoulte delta power in the 3 seconds preceeding the k-complex, in uv^2/Hz
         baseline_alpha : absoulte alpha power in the 3 seconds preceeding the k-complex, in uv^2/Hz
         after_delta : absoulte delta power in the 3 seconds after the k-complex, in uv^2/Hz
         after_alpha : absoulte alpha power in the 3 seconds after the k-complex, in uv^2/Hz
         ratio_delta : after_delta/baseline_delta,
         ratio_alpha : after_alpha/baseline_alpha


        [1] Lechat, B., et al. (2020). "Beyond K-complex binary scoring during sleep: Probabilistic
        classification using deep learning." Sleep.

        [2] Parekh A, et al. (2019) "Slow-wave activity surrounding stage N2 K-complexes and daytime
        function measured by psychomotor vigilance test in obstructive sleep apnea." Sleep.

        """
        self._scoring = {}
        self._epochs_data = {}
        hypno = self._hypno
        raw = self._raw
        include_stages = self._include_stages

        Stages = hypno
        sfreq = raw.info['sfreq']

        for channel in raw.info['ch_names']:
            ###################################
            ###### Scoring of K-complexes #####
            kc_onsets, kc_probas, kc_stages = scoring_algorithm_kc(raw, channel,
                                                                   Stages,
                                                                   score_on_stages=include_stages,
                                                                   amplitude_threshold=20e-6,
                                                                   distance=2,
                                                                   reject_epoch=400e-6,
                                                                   probability_threshold=0.5)
            # print('Detected {} K-complexes on '.format(len(kc_onsets)) + channel)
            ###################################
            ####    Calulate features      ####
            # organize event matrix for mne
            onsets_int = np.array(kc_onsets * raw.info['sfreq'], dtype='int')\
                         + self._raw.first_samp
            events = np.vstack((onsets_int, np.ones_like(onsets_int),
                                np.ones_like(onsets_int))).T
            # get epochs data
            epochs = mne.Epochs(raw, events, picks=channel, event_id=None,
                                tmin=-6, tmax=6,
                                baseline=(None, -0.5),
                                reject=None, reject_by_annotation=False,
                                verbose='critical', flat=None)
            times = epochs.times
            kc_matrix = epochs.get_data().squeeze() *-1 * 10 ** 6
            ###################################
            ###### Time-Feature calculations
            t_P900_N550, P900_timing, KC_900, KC_550, ptp_amp, slope = _temporal_features_kcs(
                kc_matrix, sfreq)
            ###################################
            ###### Frequency-Feature calculations
            delta_before, alpha_before, delta_after, alpha_after = _kc_frequency_features(
                kc_matrix, times, sfreq)

            scg = {
                'KC_onset': kc_onsets,
                'KC_probas': kc_probas,
                'KC_stage': kc_stages,
                'N550': KC_550,
                'P900': KC_900,
                'PTP': ptp_amp,
                'Slope': slope,
                'dt_P9_N5': t_P900_N550,
                'baseline_delta': delta_before,
                'baseline_alpha': alpha_before,
                'after_delta': delta_after,
                'after_alpha': alpha_after,
                'ratio_delta': (delta_after - delta_before) / delta_before,
                'ratio_alpha': (alpha_after - alpha_before) / alpha_before
            }

            self._scoring[channel] = scg
            self._epochs_data[channel] =  (kc_matrix, times, kc_probas)

        return self._scoring, self._epochs_data

    def score_from_events(self, events):

        event_onset = events.onset.values
        scoring = self._scoring
        for channel in list(scoring.keys()):
            sc = []
            d = pd.DataFrame.from_dict(scoring[channel])
            kcs_onset = d['KC_onset'].values
            for event_count, single_event_onset in enumerate(event_onset):
                args = np.argwhere(kcs_onset>single_event_onset)
                if len(args) !=0:
                    dkc = d.loc[args[0],:]
                    dkc['noise_count'] = event_count
                    dkc['delta_t'] = dkc['KC_onset'] - single_event_onset
                    sc.append(dkc)

            dch = pd.concat(sc)
            dch = dch.set_index('noise_count')
            dch.columns = [col+'_'+channel for col in dch.columns]
            events = events.merge(dch, how='left',left_index=True,
                                right_index=True)
        return events

    def overnight_metrics(self,probability_thresholds = 0.5):
        """ Calculate summary k-complex metrics

        Summary K-complexes metrics (see Notes for a detailed list) are calculated for each channels and individual
        sleep stages.

        Notes
        -----
        Parameters are calculated for each channels. Furthermore, parameters are calculated for stage 2, 3
        and NREM. For example, K-complexes densities (dKC) are returned as follows:

        dKC : KC density (#/min) in NREM sleep
        dKC_N1 : KC density (#/min) in N2
        dKC_N2 : KC density (#/min) in N2
        dKC_N3 : KC density (#/min) in N3

        Full list of parameters:
        dKC : KC density (#/min) in NREM sleep
        N550 : Amplitude of the N550 components, in uv
        P900 : Amplitude of the P900 components, in uv
        PTP : Peak to peak amplitude of the KC, in uv
        Slope : K-complex slope, define as (P900-N550)/(tP900-tN550), in uv/sec
        dt_P9_N5 : Time tP900-tN550, in seconds
        baseline_delta: absoulte delta power in the 3 seconds preceeding the k-complex, in uv^2/Hz
        baseline_alpha : absoulte alpha power in the 3 seconds preceeding the k-complex, in uv^2/Hz
        after_delta : absoulte delta power in the 3 seconds after the k-complex, in uv^2/Hz
        after_alpha : absoulte alpha power in the 3 seconds after the k-complex, in uv^2/Hz
        ratio_delta : after_delta/baseline_delta,
        ratio_alpha : after_alpha/baseline_alpha

        density_function markers ?

        [1] Lechat, B., et al. (2020). "Beyond K-complex binary scoring during sleep: Probabilistic
        classification using deep learning." Sleep.

        [2] Parekh A, et al. (2019) "Slow-wave activity surrounding stage N2 K-complexes and daytime
        function measured by psychomotor vigilance test in obstructive sleep apnea." Sleep.

        """

        if not self._scoring:
            scoring, metadata = self.score()
        else:
            scoring, metadata = self._scoring, self._epochs_data

        if any([probability_thresholds < 0, probability_thresholds > 1]):
            raise ValueError('K-complex ``probability_thresholds`` must be a float between 0 and 1.')

        """ Calculate KCs metrics"""

        metrics = {}

        for channel, kc_dict in scoring.items():
            m = kc_metrics_by_sleep_stage(kc_dict,
                                          hypnogram=self._hypno,
                                          pth = probability_thresholds)
            m = m.to_dict(orient='list')
            for key, val in m.items():
                metrics[channel + key] = float(val[0])
        return metrics

    def _plot_average(self):
        pass
        #if not self._scoring:
        #    raise RuntimeError('You need to score K-complex before plotting')
        #from .plotting import KC_from_probas, KC_from_electrodes,
        # KC_from_electrodes_all
        #KC_from_probas(self._epochs_data, np.asarray(self._scoring[
        #                                                 self._raw.info[
        #                                                     'ch_names'][
        #                                                     0]]['KC_probas']))
        #KC_from_electrodes(self._epochs_data)
        #KC_from_electrodes_all(self._epochs_data)

def kc_metrics_by_sleep_stage(kc_dict, hypnogram, pth):
    df = pd.DataFrame.from_dict(kc_dict)
    df = df.loc[df.KC_probas > pth, :]
    ## NREM
    nrem = df.mean().to_frame().T
    nrem.columns = [x + '_mean' for x in nrem.columns]
    nrem = nrem.drop(['KC_onset_mean'], axis=1)
    t = nrem

    kc_stage = df['KC_stage'].values
    if -1 in np.unique(hypnogram.label.values): # Hypnogram is unscored
        t['n_KC'] = len(kc_stage)
        t['dKC'] = float(
            len(kc_stage) * 2 / len(hypnogram.label.values))
    else:
        t['n_KC'] = float(len(kc_stage))
        t['dKC'] = float(
            np.sum(np.isin(kc_stage, [1, 2, 3, 4])) * 2 / np.sum(
                np.isin(hypnogram.label.values, [1, 2, 3, 4])))
        t['dKC_N1'] = float(np.sum(np.isin(kc_stage, [1])) * 2 / np.sum(
            np.isin(hypnogram.label.values, [1])))
        t['dKC_N2'] = float(np.sum(np.isin(kc_stage, [2])) * 2 / np.sum(
            np.isin(hypnogram.label.values, [2])))
        t['dKC_N3'] = float(np.sum(np.isin(kc_stage, [3])) * 2 / np.sum(
            np.isin(hypnogram.label.values, [3])))
    return t

def _temporal_features_kcs(time_data,Fs):
    """ Calculate characteristics time points of K-complexes
    TODO: I'm sure it's possible to do a function that can handle 1d and 2d arrays
    """
    if time_data.ndim == 1:
        return _kc_temporal_features_1d(time_data, Fs)
    else:
        return _kc_temporal_features_2d(time_data,Fs)

def _kc_frequency_features(time_data, times, sfreq):
    """ Calculate absolute power of delta and alpha band before (on a 3 seconds
     windows) and after K-complexes"""
    exp = [('before', -2.5, -0.5), ('after', 1, 3)]
    res = {}
    for m in exp:
        kc_matrix_temp = time_data[:, np.bitwise_and(times > m[1], times < m[2])]

        absol_power = compute_absol_pow_freq_bands(sfreq, kc_matrix_temp, psd_method='multitaper',
                                                   psd_params={'mt_adaptive': True, 'mt_bandwidth': 3,
                                                               'mt_low_bias': True},
                                                   freq_bands=[0.5, 4, 8, 12])
        delta = absol_power[:, 0]
        alpha = absol_power[:, 2]
        res[m[0]] = (delta, alpha)
    delta_before, alpha_before, delta_after, alpha_after = res['before'][0], res['before'][1],\
                                                           res['after'][0], res['after'][1]
    return delta_before, alpha_before, delta_after, alpha_after

def _kc_temporal_features_1d(time_data, Fs):
    """Calculate kc features for 1d array"""
    half_index = int(len(time_data) / 2)
    #epochs are centered around N550 components
    N550_index = np.argmax(time_data[half_index - int(0.2 * Fs):half_index + int(0.2 * Fs)]) + \
                 half_index - int(0.2 * Fs)
    P900_index = np.argmax(-1 * time_data[half_index + int(0.2 * Fs):half_index + int(0.750 * Fs)]) + \
                 half_index + int(0.2 * Fs)
    t_P900_N550 = (P900_index - N550_index) / Fs
    P900_timing = (P900_index - half_index) / Fs
    KC_900 = -1 * time_data[P900_index]
    KC_550 = time_data[N550_index]
    ptp_amp = abs(KC_900) + KC_550
    slope = ptp_amp / t_P900_N550
    return t_P900_N550, P900_timing, KC_900, KC_550, ptp_amp, slope

def _kc_temporal_features_2d(time_data, Fs):
    """Calculate kc features for 2d array"""
    half_index = int(np.shape(time_data)[1] / 2)
    N550_index = np.argmax(-1*
        time_data[np.arange(np.shape(time_data)[0]), half_index - int(0.2 * Fs):half_index + int(0.2 * Fs)],
        axis=1) + half_index - int(
        0.2 * Fs)
    P900_index = np.argmax(
        time_data[np.arange(np.shape(time_data)[0]), half_index + int(0.2 * Fs):half_index + int(0.750 * Fs)],
        axis=1) + half_index + int(
        0.2 * Fs)
    t_P900_N550 = (P900_index - N550_index) / Fs
    P900_timing = (P900_index - half_index) / Fs
    KC_900 = time_data[np.arange(np.shape(time_data)[0]), P900_index]
    KC_550 = -1*time_data[np.arange(np.shape(time_data)[0]), N550_index]
    ptp_amp = abs(KC_900) + KC_550
    slope = ptp_amp / t_P900_N550
    return t_P900_N550, P900_timing, KC_900, KC_550, ptp_amp, slope

##########################################################################
##                      K-complex scoring functions                     ##
##########################################################################

def scoring_algorithm_kc(raw, channel, stages, score_on_stages = [1,2,3], amplitude_threshold = 20e-6, distance = 2,
                reject_epoch = 500e-6, probability_threshold = 0.5):
    """
    Score K-complexes according to [1]. Briefly, peaks superior to
    "amplitude_threshold" in the raw EEG are found, and  then classified
    using deep kernel learning. Deep kernel learning is a mix between neural
    network and gaussian processes; and it attributes each waveform a
    "probability" (probability threshold) of being a K-complex. The higher
    the probability, the more "confident" is the algorithm; which is generally
    seen in very large and well defined K-complexes.


    Parameters
    ----------
    raw : :py:class:`mne.io.BaseRaw`
        Raw data
    channel : str
        Channel on which socre K-complexes
    stages : pd.DataFrame
        Dataframe containing the following keys: "onset" (sleep stage scoring onset), "dur" (duration of the scored
        stage) and "label" (sleep stage label)
    score_on_stages : list
        Valid sleep stages to score K-complexes.
    amplitude_threshold : float or int
        Minimum amplitude for a peak to be considered as possible K-complexes
    distance: float or int
        Minimum between two consecutive peaks to be classified as K-complexes
    reject_epoch: float or int
        Reject candidate K-complexes if their maximum values (positive or negative) is superior to this value
    probability_threshold: float
        Reject waveform scored as K-complexes if their probability is inferior to this threshold.

    Returns
    -------

    onsets: K-complexes onsets (in seconds)
    probas: Probability of the K-complex
    stage_peaks: sleep stage of the k-complex

    Notes
    -----

    Lechat, B., et al. (2020). "Beyond K-complex binary scoring during sleep: Probabilistic
    classification using deep learning." Sleep.
    """

    C3 = np.asarray(
        [raw[count, :][0] for count, k in enumerate(raw.info['ch_names']) if
         k == channel]).ravel()
    Fs = raw.info['sfreq']

    st = stages.loc[stages['label'].isin(score_on_stages),:]
    length_of_stages = int(st['duration'].values[0]*Fs)
    onset_of_stages = np.round(st['onset'].values[1:-1]* Fs).astype('int')
    stage_label = st['label'].values[1:-1]

    ###### preprocessing ###########
    peaks,stage_peaks = _find_peaks_staged(C3, Fs, sonset=onset_of_stages,sdur=length_of_stages, slabel=stage_label,
                    min = amplitude_threshold, distance=distance)
    d, args = Epochdata(C3, Fs, peaks, detrend=True, reject_max = reject_epoch)
    peaks = peaks[args]
    stage_peaks = stage_peaks[args]

    d_pad = pad_nextpow2(d)


    ######## Wavelet decomposition #########
    wavelet = pywt.Wavelet('sym3')
    coefs = pywt.wavedec(d_pad, wavelet=wavelet, mode='periodization', level=pywt.dwt_max_level(d.shape[-1], wavelet.dec_len))
    X = np.hstack(coefs[:5])

    ########### Model prediction #############

    model, likelihood = get_model()
    data_scaled = scale_input(X)

    probas, _ = predict(model, likelihood, torch.from_numpy(data_scaled))

    #######################################################################
    stage_peaks = stage_peaks[probas > probability_threshold]
    onsets = peaks[probas > probability_threshold] / Fs
    probas = probas[probas > probability_threshold]

    return onsets, probas, stage_peaks

##########################################################################
##                      pre-processing functions                        ##
##########################################################################
def scale_input(X, scaler = True):
    scaler_filename = os.path.join(wd, 'model/scaler_final_A2.save')
    scaler = joblib.load(scaler_filename)
    X_scaled = scaler.transform(X)
    return X_scaled

def pad_nextpow2(dat):
    """
    return an array pad with zero to the next power of 2 of the input
    """
    g = np.ceil(np.log2(np.shape(dat)[1]))
    ze = np.zeros((np.shape(dat)[0],np.array(np.power(2, g) - np.shape(dat)[1], dtype='int')))
    data = np.hstack([dat, ze])
    return data

def _find_peaks_staged(data, Fs, sonset,sdur, slabel,
                    min, distance):
    """Find peaks of at least "min" amplitude the given sleep stages
    """
    p = []
    stages = []
    for j,(low,up,sstage) in enumerate(zip(sonset, sonset+sdur,slabel)):
        data_for_peak = data[low:up] - np.mean(data[low:up])
        temp, _ = find_peaks(data_for_peak, height=min, distance=distance * Fs)
        p.append(temp + low)
        stages.append(np.ones(len(temp))*sstage)
    return np.hstack(p), np.hstack(stages)

def Epochdata(data, Fs, peaks, post_peak=3, pre_peak=3, detrend=True, reject_max = None):
    """ Epochs raw data for each peak in peaks.
    """
    max_peaks_locs = len(data) - int(post_peak*Fs)
    min_peaks_locs = int(pre_peak*Fs)
    peaks = peaks[np.bitwise_and(peaks>min_peaks_locs,peaks<max_peaks_locs)]
    epochs = np.vstack([data[up:low] for up,low in zip(peaks-int(pre_peak * Fs), peaks+int(post_peak * Fs))])
    if detrend:
        epochs = epochs - np.mean(epochs,axis=1, keepdims=True)
    if reject_max is not None:
        args = np.argwhere(~(np.max(np.abs(epochs),axis=1)>reject_max)).squeeze() #print(np.max(np.abs(epochs),axis=1))
        epochs = epochs[args,:]
        return epochs, args
    else:
        return epochs

##########################################################################
##                      Predictions models/functions                    ##
##########################################################################
class LargeFeatureExtractor(torch.nn.Sequential):
    """ Neural network used for feature extraction"""
    def __init__(self, input_dim, output_dim,drop_out =0.5):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(input_dim, 1000, bias=False))
        self.add_module('bn1', torch.nn.BatchNorm1d(1000))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('dropout1', torch.nn.Dropout(p=drop_out, inplace=False))


        self.add_module('linear2', torch.nn.Linear(1000, 1000,bias=False))
        self.add_module('bn2', torch.nn.BatchNorm1d(1000))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('dropout2', torch.nn.Dropout(p=drop_out, inplace=False))


        self.add_module('linear3', torch.nn.Linear(1000, 500,bias=False))
        self.add_module('bn3', torch.nn.BatchNorm1d(500))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('dropout3', torch.nn.Dropout(p=drop_out, inplace=False))


        self.add_module('linear4', torch.nn.Linear(500, 256,bias=False))
        self.add_module('bn4', torch.nn.BatchNorm1d(256))
        self.add_module('relu4', torch.nn.ReLU())
        self.add_module('dropout4', torch.nn.Dropout(p=drop_out, inplace=False))


        self.add_module('linear6', torch.nn.Linear(256, output_dim,bias=False))

class GaussianProcessLayer(gpytorch.models.AbstractVariationalGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = WhitenedVariationalStrategy(self, inducing_points, variational_distribution,
                                                           learn_inducing_locations=True)
        super(GaussianProcessLayer, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class DKLModel(gpytorch.Module):
    """ Deep kernel learning model as gaussian processes on top of neural network"""
    def __init__(self, inducing_points, feature_extractor, num_features):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(inducing_points)
        self.num_features = num_features

    def forward(self, x):
        #print(x.type())
        projected_x = self.feature_extractor(x.float())

        res = self.gp_layer(projected_x)
        return res

def predict(model, likelihood, X):
    """prediction """
    model.eval()
    likelihood.eval()
    correct = 0
    with torch.no_grad():
        output = likelihood(model(X))  #
        pred_labels = output.mean.ge(0.5).float().cpu().numpy()
        probas = output.mean.cpu().numpy()
    return probas, pred_labels

def get_model():
    """ convenience function to load the model with its parameters """
    inducing_filename = os.path.join(wd, 'model/inducing_points_A2.npy')
    model_file = os.path.join(wd, 'model/finaldkl_final_model_epoch50.dat')
    data_dim = 128
    num_features = 16
    drop_out_rate = 0.8
    feature_extractor = LargeFeatureExtractor(input_dim=data_dim,
                                              output_dim=num_features,
                                              drop_out=drop_out_rate)
    X_induced = torch.from_numpy(np.load(inducing_filename))
    model = DKLModel(inducing_points=X_induced, feature_extractor=feature_extractor,
                     num_features=num_features)

    # Bernouilli likelihood because only 2 classes
    likelihood = gpytorch.likelihoods.BernoulliLikelihood()
    model.load_state_dict(torch.load(model_file,map_location=torch.device('cpu'))['model'])
    likelihood.load_state_dict(torch.load(model_file,map_location=torch.device('cpu'))['likelihood'])

    return model, likelihood