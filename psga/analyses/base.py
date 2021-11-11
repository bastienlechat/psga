import mne
import pandas as pd
import warnings
import yaml

class BaseMethods(object):
    """ Base class for each PSG analyses"""

    def __init__(self):
        self._scoring = {}
        self._epochs_data = {}

    def fit(self, raw, hypnogram, picks=None,**kwargs):
        raise NotImplementedError

    def score(self, *args, **kwargs):
        raise NotImplementedError

    def score_from_events(self,event_file):
        raise NotImplementedError

    def overnight_metrics(self, *args, **kwargs):
       raise NotImplementedError

    def _check_raw(self, raw):
        if not isinstance(raw,mne.io.BaseRaw):
            raise ValueError('raw must be a mne.io.BaseRaw object, got a {}'.format(type(raw)))

    def _check_hypno(self, hyp):
        if not isinstance(hyp,pd.DataFrame):
            raise ValueError('raw must be a pd.DataFrame object, '
                             'got a {}'.format(type(hyp)))

    def set_params(self, parameters_dict, check_has_key=False):
        for key, value in parameters_dict.items():
            if value:
                if check_has_key:
                    if hasattr(self,key):
                        setattr(self, key, value)
                    else:
                        warnings.warn(key + ' key is not a valid attribute')
                else:
                    setattr(self,key,value)

    def get_params(self):
        attr = vars(self) #.__dict__

        attr_public = {k:v for k,v in attr.items() if not k.startswith(
            '_') }
        return attr_public