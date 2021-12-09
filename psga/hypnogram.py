import pandas as pd
import numpy as np
import mne

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


class AutomatedSleepStaging(object):
    def __init__(self, raw, type='yasa'):
        self.type = type
        assert isinstance(raw, mne.io.BaseRaw)
        self.raw = raw
        self._automated_hyp = None

    def score(self, **kwargs):
        if self.type=='yasa':
            assert 'EEG' in kwargs.keys()
            assert 'EOG' in kwargs.keys()
            assert 'EMG' in kwargs.keys()
            hyp = self._yasa_score(EEG = kwargs['EEG'],
                             EOG = kwargs['EOG'],
                             EMG = kwargs['EMG'])
            self._automated_hyp = hyp
            return hyp
        else:
            raise NotImplementedError

    def validate(self, true_stage):
        """

        Parameters
        ----------
        true_stage : np.array
            Sleep staging to compare automatic scoring against (e.g. manual)

        Returns
        -------

        """
        pred = self._automated_hyp
        true = true_stage

        if len(true)>len(pred): # assume that something went wrong with the
            # last epoch
            true = true[:-1]

        assert len(pred) == len(true)

        from sklearn.metrics import classification_report, confusion_matrix, \
                                ConfusionMatrixDisplay
        import matplotlib.pyplot as plt
        report = classification_report(true, pred, target_names=['Wake','N1',
                                                              'N2','N3','REM'])

        cm = confusion_matrix(true, pred)

        ConfusionMatrixDisplay.from_predictions(true,pred, display_labels=[
            'Wake','N1','N2','N3','REM'])
        plt.show()
        return report, cm


    def _yasa_score(self, EEG, EOG, EMG, plot = True):
        """
        Small wrapper around yasa.SleepStaging class to automatically score
        sleep stages. All credit goes to the original authors. See [1] for
        more informations on the algorithm.

        Parameters
        ----------
        EEG : str
            Name of the EEG channel. Central (C3 or C4) recommended.
        EOG : str
            Name of the EOG location.
        EMG : str
            Name of the EMG location
        plot : bool
            If True, probability of each sleep stages will
            be plotted as an hypnogram using
            yasa.SleepStaging.plot_predict_proba(). Default set to False

        Returns
        -------
        hypno : np.array
            Array of automatically scored sleep stages*

        References
        ----------
        [1] Vallat, R. & Walker, M. P. (2021). An open-source, high-performance
        tool for automated sleep staging. Elife, 10.

        Notes
        -----
        In YASA, sleep stages are labelled as W, N1, N2, N3 and R. We
        replaced this label to 0,1,2,3,5 for compatibility with others
        function in this package.
        """

        try:
            import yasa
        except:
            raise ImportError('YASA must be installed')
        sls = yasa.SleepStaging(self.raw, eeg_name=EEG, eog_name=EOG,
                                emg_name=EMG,
                                metadata=None)
        hypno = sls.predict()
        print(len(hypno))
        hypno[hypno == 'W'] = 0
        hypno[hypno == 'N1'] = 1
        hypno[hypno == 'N2'] = 2
        hypno[hypno == 'N3'] = 3
        hypno[hypno == 'R'] = 5

        if plot:
            import matplotlib.pyplot as plt
            sls.plot_predict_proba()
            plt.show()
        return np.asarray(hypno, dtype='int')