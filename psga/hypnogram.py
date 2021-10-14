import pandas as pd
import numpy as np

class Hypnogram(object):

    def __init__(self, df,meas_date=None,
                 sleep_onset_offset=(None,None)):
        onset, label, duration = df['onset'].values, df['label'].values, \
                                 df['duration'].values
        self.sleep_stage_onset = onset
        self.sleep_stage_duration = duration
        self.sleep_stage_label = label
        sleep_epochs = self.sleep_stage_onset[np.isin(
            self.sleep_stage_label,[1,2,3,4,5])]
        self.sleep_onset = sleep_epochs[0] if sleep_onset_offset[0] is None \
            else sleep_onset_offset[0]
        self.sleep_offset = sleep_epochs[-1] if sleep_onset_offset[1] is None \
            else sleep_onset_offset[1]

    def to_df(self, sleep_onset_offset=True, windows_size=None):
        Stages = pd.DataFrame(data={'onset': self.sleep_stage_onset, 'dur': self.sleep_stage_duration,
                                    'label': self.sleep_stage_label})
        if windows_size is not None:
            Stages = _convert_hypno(Stages, windows_size)
        if sleep_onset_offset:
            val_stages = np.nonzero(np.isin(Stages['label'].values, [1, 2, 3, 4, 5]))[0]
            sleep_onset = val_stages[0]
            sleep_offset = val_stages[-1]
            Stages = Stages.loc[sleep_onset:sleep_offset, :]
        return Stages

    def hypnogram_features(self):
        feature = {}
        feature['TST'] = np.sum(self.sleep_stage_duration[np.isin(self.sleep_stage_label, [1, 2, 3, 4, 5])]) / 3600
        feature['TRT'] = np.sum(self.sleep_stage_duration)/3600
        feature['N1_min'] = np.sum(self.sleep_stage_duration[np.isin(self.sleep_stage_label, [1])]) / 60
        feature['N2_min'] = np.sum(self.sleep_stage_duration[np.isin(self.sleep_stage_label, [2])]) / 60
        feature['N3_min'] = np.sum(self.sleep_stage_duration[np.isin(self.sleep_stage_label, [3])]) / 60
        feature['REM_min'] = np.sum(self.sleep_stage_duration[np.isin(self.sleep_stage_label, [5])]) / 60
        feature['Unscored_min'] = np.sum(self.sleep_stage_duration[np.isin(self.sleep_stage_label, [9])]) / 60
        feature['SOL'] = float(np.argwhere(np.isin(self.sleep_stage_label, [1, 2, 3, 4, 5]))[0] * 30 / 60)
        feature['SE'] =  (feature['TST']*3600) / np.sum(self.sleep_stage_duration[np.isin(
            self.sleep_stage_label, [0, 1, 2, 3, 4, 5])])
        return feature

def _convert_hypno(stages, windows_length):
    if stages['duration'].values[0] != windows_length:
        labels = stages['label'].values
        if stages['duration'].values[0] % windows_length == 0:
            repeat_number = stages['duration'].values[0] / windows_length
        else:
            raise ValueError('Analysis windows length needs to be a multiple of sleep stage length')
        St = pd.DataFrame([])
        St['label'] = np.repeat(labels, repeat_number)
        dur = np.ones_like(St['label'].values) * windows_length
        St['duration'] = dur
        St['onset'] = np.cumsum(St['duration'].values) + stages['onset'].values[0]\
                      - windows_length
    else:
        St = stages
    return St