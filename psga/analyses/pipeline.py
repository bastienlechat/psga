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
        'hrv': (HRV(), HRV_PARAMS),
        'qeeg':(qEEG(),qEEG_PARAMS),
        #'kc':  (None, KC_PARAMS,['C3','C4']),
                        }

    def __init__(self, steps=None):
        if steps is None:
            self.pipeline = self.DEFAULT_PIPELINE
        self.analysis_list = list(self.pipeline.keys())

    def get_params(self):
        """iterates over analysis to get default params"""
        params = {}
        for name,cls in self.pipeline.items():
            params[name] = cls[0].get_params()
        return params

    def fit_params(self,params_dict):
        """ iterates over analysis to set parameters"""
        for keys,params_analysis in params_dict.items():
            if keys in self.analysis_list:
                self.pipeline[keys][0].set_params(params_dict[keys],
                                                        check_has_key=True)
            else:
                raise AttributeError('{} is not a valid analysis'.format(keys))

    def run(self, raw, hypno, events=None,
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
        metrics = {}

        return metrics

