
class PSGbase:

    def __init__(self, folder):
        """
        base class for polysomnography object
        """
        self._folder = folder

    def hypnogram(self):
        raise NotImplementedError

    def raw_data(self):
        raise NotImplementedError

    def events(self):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError