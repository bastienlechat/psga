import warnings
import os
from ..base import PSGbase
import numpy as np
import pandas as pd
from .utils import _read_event, _read_technician_note, _read_sleep_stage, \
    _read_data_segments,_read_header,_read_montage, _read_epoch_data,\
    read_dat_data, read_d16_data, read_dat_discrete_data
import scipy.signal
import mne
import time
import scipy.signal

def read_psg_compumedics(folder,include = 'all',mne_output = True,
                         resample = True, sf = 128
                         ):
    """
    Read compumedics raw files. This function was only tested for files
    recorded with Grael. Furthermore, data types may be different for older
    file. If an error occurs, consider exporting your data to .edf using
    profusion and use our :py:

    Parameters
    ----------
    folder : str
        path to the folder containing compumedics files
    include : list or 'all'
        The list of channel to be loaded
    mne_output : bool
        If True (default), will output raw data as a :py:class:`mne.io.BaseRaw`
        object. If False, will return a dict.
    resample : bool
        If True, all channels will be resampled to "sf" frequency.
    sf : int
        Sampling frequency to resample channels to (default 128).

    Returns
    -------

    raw : :py:class:`mne.io.BaseRaw` or 'dict
        An MNE Raw instance if mne_output is True, else a dict containing
        informations about each channels + the data
    hypnogram : pd.DataFrame
        A dataframe with sleep staging (label, duration and onset) informations.
    events : pd.DataFrame
        A dataframe containing events (e.g. apneas, arousals) informations.
        Each line corresponds to a different events. Dataframe keys are
        "EVT_LABEL", "EVT_TIME" and "EVT_LENGTH" and represents label,
        onset and duration of the onsets.
    """
    for file in ['STUDYCFG.xml', 'DATASEGMENTS.xml']:
        if not os.path.exists(os.path.join(folder, 'STUDYCFG.xml')):
            raise IOError('{} folder does not contain the {} file'.format(
                folder, file))

    pg = PSGcompumedics(folder)
    dtcdata = pg.raw_data(include=include)
    hypnogram = pg.hypnogram()
    events = pg.events()
    if mne_output:
        ch_name = []
        ch_data = []
        ch_list = list(dtcdata.keys())
        for ch in ch_list:
            temp = dtcdata[ch]
            if len(temp['data']) != 0:
                if 'DiscreteData' not in list(temp.keys()):
                    ch_name.append(ch)
                    data = temp['data']
                    datafs = int(temp['Rate'])
                    if temp['Type'] == '4': datafs = datafs/2 #it seems
                    # sampling frequency is wrong for these channels
                    if resample:
                        resampled_data = scipy.signal.resample_poly(\
                            data, sf, datafs) * -1
                        ch_data.append(resampled_data)
                    else:
                        if datafs == 512:
                            sf=512
                            ch_data.append(data)
        info = mne.create_info(ch_name, sfreq=sf)
        info['meas_date'] = pg.start_date()
        raw = mne.io.RawArray(np.vstack(ch_data), info, verbose='error')
    else:
        raw = dtcdata

    return raw,hypnogram,events

def cpm_what_channels(folder):
    """Helper functions to print available channels given a folder with
    polysomnography files"""
    pg = PSGcompumedics(folder)
    return pg.available_channel

class PSGcompumedics(PSGbase):
    """
    Class to read compumedics files
    """

    def __init__(self,folder, lights_on_off = None):
        super().__init__(folder)
        self.multiple_data_segment = False
        self._folder = folder

        # POLYSOMNOGRAPHY INFO
        self.montage = _read_montage(folder)
        self.available_channel = list(self.montage.keys())
        self.header = _read_header(folder)
        self.data_segments = _read_data_segments(folder)
        self._tech_notes = _read_technician_note(folder)
        self._epoch_data = _read_epoch_data(folder)
        self._epoch_length = int(self.header['EpochLength'])
        self._stages = _read_sleep_stage(folder)

        if lights_on_off is None:
            self.lights_off, self.lights_on = self._find_lights()

    def raw_data(self, include = 'all', detrend = True):
        """
        Reads PSG raw data and returns a dict.

        Parameters
        ----------
        include : list or 'all'
            The list of channel to be loaded
        detrend : bool
            If true, linear trend (offset values) will be removed using
            :py:`scipy.signal.detrends`

        Returns
        -------
        raw : dict
            Data. Keys and values will depend on the channels (discrete vs
            continuous)
        """
        montage = self.montage
        if include is 'all': include = list(montage.keys())
        for ch_name in list(montage.keys()):
            if ch_name in include:
                ch = montage[ch_name]
                ch_file = os.path.join(self._folder, ch['Filename'])
                ch_fs = int(ch['Rate'])
                ext =  os.path.splitext(ch_file)[1]
                if ext =='.D16':
                    data = read_d16_data(ch_file, ch_fs)
                elif ext=='.DAT':
                    if 'DiscreteData' in list(ch.keys()):
                        t, data= read_dat_discrete_data(ch_file, ch_fs)
                    else:
                        data = read_dat_data(ch_file)
                else:
                    raise ValueError('Weird channel format: ' + ext)

                if int(ch['Type']) == 1:
                    data = scipy.signal.detrend(data)
                    data = mne.filter.filter_data(np.asarray(data,
                                                            dtype='float'),
                                                            ch_fs,l_freq=0.05,

                                                  h_freq=None, verbose='error')

                if 'DiscreteData' not in list(ch.keys()):
                    if len(self.data_segments)>1:
                        if int(ch['Type']) == 4: ch_fs = ch_fs/2
                        starts = np.asarray([int(seg['Start']) for seg in \
                                self.data_segments])
                        durs = np.asarray([int(seg['Duration']) for seg in \
                                self.data_segments])
                        total_duration = starts[-1] + durs[-1]
                        array = np.zeros((int(ch_fs*(total_duration)),1)).squeeze()
                        prev = 0
                        for (start,stop,du) in zip(starts,starts+durs,
                                                   durs):
                            end_data = prev + int(du*ch_fs)
                            array[int(start*ch_fs):int(stop*ch_fs)] = data[
                                prev:end_data]
                            prev = prev + int(du*ch_fs)
                        data = array
                    montage[ch_name]['data'] = data
                else:
                    montage[ch_name]['data'] = (t, data)
            else:
                montage[ch_name]['data'] = []
        return montage

    def posture(self):
        """
        Reads posture information from compumedics files.

        Returns
        -------
        posture : pd.Dataframe
            Dataframe containing posture informations for each epochs.
        """
        posture = self._epoch_data[['Posture']]
        posture['onset'] = np.cumsum(np.ones_like(posture['Posture'].values)*30)
        posture['duration'] = np.ones_like(posture['onset'].values) * 30
        return posture

    def hypnogram(self, trim = False):
        """
        Reads sleep staging informations.

        Returns
        -------
        hypnogram : pd.DataFrame
            A dataframe with sleep staging (label, duration and onset)
            informations.
        """
        hypno = np.asarray(self._stages, dtype='int')
        onsets = np.cumsum(np.ones_like(hypno)*self._epoch_length) - 30
        if trim:
            index_hypn_in_loff_lon = np.bitwise_and(onsets>self.lights_off,
                                                    onsets<self.lights_on)
            onsets = onsets[index_hypn_in_loff_lon]
            hypno = hypno[index_hypn_in_loff_lon]
        stage_duration = np.ones_like(hypno) * 30
        Stages = pd.DataFrame(data={'label': hypno,
                                    'duration': stage_duration,
                                    'onset': onsets,
                                    })
        return Stages

    def events(self):
        """
        Reads manual scoring of events (e.g. apneas and arousals).

        Returns
        -------
        events : pd.DataFrame
            A dataframe containing events (e.g. apneas, arousals) informations.
            Each line corresponds to a different events. Dataframe keys are
            "EVT_LABEL", "EVT_TIME" and "EVT_LENGTH" and represents label,
            onset and duration of the onsets.
        """
        _events = _read_event(self._folder)
        return _events

    def _find_lights(self):
        from ..utils import lights_is_wrong
        lights_off, lights_on = self._find_lights_from_tech_notes()
        if lights_is_wrong(lights_off, lights_on):
            if len(self._stages) > 0:
                lights_on = len(self._stages) * int(self._epoch_length)
                lights_off = 0
        return (lights_off, lights_on)

    def _find_lights_from_tech_notes(self):
        lights_off, lights_on = (None,None)
        if self._tech_notes:
            labels = self._tech_notes['event']
            if all(['LIGHTS OFF' in labels, 'LIGHTS ON' in labels]):
                for event in ['LIGHTS OFF','LIGHTS ON']:
                    index_lo = [idx for idx, s in enumerate(labels) if s == event]
                    if len(index_lo) > 1:
                        warnings.warn("Found {} " + event +" tech notes, setting "
                                                           "lights "
                                      "on/off to the last".format(len(index_lo)))
                        index_lo = index_lo[-1]
                    if index_lo:
                        if event=='LIGHTS OFF':
                            lights_off = int(index_lo[0])
                        else:
                            lights_on = int(index_lo[0])
            else:
                warnings.warn('Tech notes does not contain LIGHTS OFF and '
                              'LIGHTS ON.')
        else:
            warnings.warn('No tech form to infer lights on/off from.')
        return (lights_off, lights_on)

    def start_date(self):
        import datetime
        date = self.header['StartDate'] + ' ' + self.header['StartTime']
        datetime_object = datetime.datetime.strptime(date,
                                            '%d/%m/%Y %H:%M:%S')
        datetime_object = datetime_object.replace(tzinfo=datetime.timezone.utc)
        return datetime_object

    def info(self):
        print('-----------Compumedics Polysomnography ------------')
        print('- Date : {}                  '
              'Time: {}'.format(self.header['StartDate'],self.header['StartTime']))
        print('- Lights on time : {}          '
              'Lights off time: {}'.format(self.lights_on,self.lights_off))