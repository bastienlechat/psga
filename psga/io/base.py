
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