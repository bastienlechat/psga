import numpy as np
import os, glob
import mne

from .breaths import Breathing
from .hrv import HRV
from .qeeg import qEEG

ECG_PARAMS = {}
qEEG_PARAMS = {'picks':['C3', 'C4','F3','F4','O1','O2'],
               'windows_length':5,
               'psd_method':'multitaper',
               'events_windows_length':5,
               'events_lower_bound':-20,
               'events_upper_bound':20,
}

KC_PARAMS = {}
BB_PARAMS = {}
HRV_PARAMS = {}

class Pipeline(object):
    """Create/run a pipeline of pre-defined sleep analysis steps.

    Each analysis in the pipeline is a child of BaseMethods class (all
    grouped in GMOOP.analysis).

    The pipeline is constructed using a dict. Keys of the dict are the name
    of the analysis step ('qeeg' for qEEG analysis) and a Step dataclass
    container.

    Step dataclass contains on which channels to perform the analysis (chs),
    whether or not perform the analysis (analyse) and the actual analysis
    methods.

    #TODO: User input analysis step ?

    """

    DEFAULT_PIPELINE = {
        'hrv': (HRV(),HRV_PARAMS, ['ECG']),
        'qeeg':(qEEG(),qEEG_PARAMS, ),
        #'kc':  (None, KC_PARAMS,['C3','C4']),
                        }

    def __init__(self, steps=None):
        if steps is None:
            self.steps = self.DEFAULT_PIPELINE
        self.analysis_list = list(self.steps.keys())

    def _set_channel_picks(self, analysis_type):
        """map required channels"""
        if analysis_type =='hrv':
            supported_ecg_channels = ['ECG']
            ecg_chan = [ch for ch in supported_ecg_channels if ch in
                         self.raw.info['ch_names']]
            return ecg_chan
        elif analysis_type =='qeeg':
            supported_qeeg_channels = ['C3', 'C4','F3','F4']
            qeeg_chan = [ch for ch in supported_qeeg_channels if ch in
                       self.raw.info['ch_names']]
            return qeeg_chan
        elif analysis_type =='kc':
            supported_kc_channels = ['C3','C4']
            kc_chan = [ch for ch in supported_kc_channels if ch in
                       self.raw.info['ch_names']]
            return kc_chan
        else:
            raise ValueError('No channels for this analysis')

    def get_params(self):
        """iterates over analysis to get default params"""
        params = {}
        for name,cls in self.steps.items():
            params[name] = cls.get_params()
        return params

    def fit_params(self,params_dict):
        """ iterates over analysis to set parameters"""
        for keys,params_analysis in params_dict.items():
            if keys in self.analysis_list:
                self.steps[keys].set_params(params_dict[keys],
                                                        check_has_key=True)
            else:
                raise AttributeError('{} is not a valid analysis'.format(keys))

    def run(self, edf_file, hypnogram_file, config, event_file=None,
             params_dict=None,path=None,
            ):
        """

        Parameters
        ----------
        edf_file
        hypnogram_file
        event_file
        params_dict
        path

        Returns
        -------

        """

        if params_dict is None:
            params_dict = {k:{} for k in self.analysis_list}

        px_id = os.path.basename(os.path.dirname(edf_file))
        if path is None: path = os.path.join(os.path.dirname(edf_file), px_id)

        metrics = OvernightMetric()
        metrics['id'] = px_id

        self.config_tmp = config
        self.ch_mapping = config['channel']
        self._load_raw(edf_file) # load/init self.raw
        self._load_hypno(edf_file,hypnogram_file) #load/init self.hypnogram

        metrics.add_dict_to_metrics(self.hypnogram.hypnogram_features())

        for name, cls in self.steps.items():
            param_analysis = params_dict[name]

            try:
                if config['analyze'][name]:
                    channels = self._set_channel_picks(name)
                    if channels:
                        temp_metric = cls.fit_participant(self.raw.copy(),
                                                        hypnogram=self.hypnogram,
                                                        picks= channels,
                                                        path = path,
                                                        events=event_file,
                                                        **param_analysis
                                                                   )
                        metrics.add_dict_to_metrics(temp_metric)

            except:
                print('Analysing ' +name+' did not work; for ' + px_id)

        return metrics.get_metrics()

    def _load_raw(self, filename):
        raw = mne.io.read_raw_edf(filename, verbose='CRITICAL',
                                  preload=False)

        eeg_channels = np.hstack([self.ch_mapping['eeg'],self.ch_mapping[
            'eegref']]) if self.ch_mapping['eegref'] else self.ch_mapping['eeg']
        ch_list = np.hstack([self.ch_mapping['eeg'],
                             self.ch_mapping['eegref'],
                             self.ch_mapping['ecg']
                             ])

        raw = raw.pick_channels(ch_list).load_data()

        # map eeg channels to eeg "types" for mne to do referencing
        ch_type_mapping = {chan:('eeg' if chan in eeg_channels
                                 else 'misc') for chan in ch_list
                           }

        raw = raw.resample(128).set_channel_types(ch_type_mapping,
                                                         verbose=None)

        if self.ch_mapping['eegref']:
            self.raw, _ = mne.set_eeg_reference(raw, self.ch_mapping['eegref'],
                                                #ch_type='eeg',
                                                verbose='CRITICAL')
        else:  # if no ref is provided assume data are already referenced
            self.raw, _ = mne.set_eeg_reference(raw, [],
                                                    verbose='CRITICAL')

        name_map = {}
        for ch_name in self.raw.info['ch_names']:
            for key,item in self.config_tmp['ch_map'].items():
                if ch_name in item:
                    name_map[ch_name] = key

        self.raw.rename_channels(mapping = name_map)