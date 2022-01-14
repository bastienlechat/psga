
class PSGbase:
    """Base class for polysomnography"""

    def __init__(self, folder):
        self._folder = folder

    def hypnogram(self):
        raise NotImplementedError

    def raw_data(self):
        raise NotImplementedError

    def events(self):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError

    def automatic_scoring(self, raw,
                          EEG ='C3',EEG_REF='M2', EOG='EOG',
                          EMG='EMG', algorithm='yasa'):
        pass
