import xml.etree.ElementTree as ET
import warnings
import os
import glob
from ..base import PSGbase
import pandas as pd
import numpy as np
import mne

def read_psg_edf(folder, include = 'all', preload = False):
    """
    Read compumedics polysomnography files exported to .edf. This function was
    only tested for .edf exported from compumedics profusion software and
    requires a .xml file with scoring informations.

    Parameters
    ----------
    folder : str
        path to the folder containing the .edf file and the .xml scoring file
    include : list or 'all'
        The list of channel to be loaded
    preload : bool
        Whether or not data is loaded in memory.

    Returns
    -------
    raw : :py:class:`mne.io.BaseRaw`
        An MNE Raw instance.
    hypnogram : pd.DataFrame
        A dataframe with sleep staging (label, duration and onset) informations.
    events : pd.DataFrame
        A dataframe containing events (e.g. apneas, arousals) informations.
        Each line corresponds to a different events. Dataframe keys are
        "EVT_LABEL", "EVT_TIME" and "EVT_LENGTH" and represents label,
        onset and duration of the onsets.

    Note
    ----
    Onset of sleep stages and events are in seconds elapsed since the
    beginning of the PSG recording.
    """
    if len(glob.glob(folder + '/*.edf'))==0:
        raise ValueError("No edf file there: " + folder)
    if len(glob.glob(folder + '/*.xml'))==0:
        raise ValueError("No xml file there: " + folder)
    pg = PSGedf(folder)
    hypnogram = pg.hypnogram()
    events = pg.events()
    raw = pg.raw_data(include=include, preload = preload)

    return raw, hypnogram, events

def edf_what_channels(folder):
    """Helper functions to print available channels given a folder with
    polysomnography files"""
    pg = PSGedf(folder)
    print(pg.available_channel)

class PSGedf(PSGbase):
    """
    Class to read edf files.
    """

    def __init__(self, folder):
        super().__init__(folder)
        self.edf_file = glob.glob(folder + '/*.edf')[0]
        self.xml_file = glob.glob(folder + '/*.xml')[0]
        r = mne.io.read_raw_edf(self.edf_file, preload=False, verbose='error')
        self.available_channel = r.info['ch_names']

    def hypnogram(self):
        """
        Reads sleep staging informations.

        Returns
        -------
        hypnogram : pd.DataFrame
            A dataframe with sleep staging (label, duration and onset)
            informations.
        """
        stage = import_stages_from_xml(self.xml_file)
        return stage

    def raw_data(self, include = 'all', preload=False):
        """
        Reads PSG raw data and returns a :py:class:`mne.io.BaseRaw`.

        Returns
        -------
        raw : :py:class:`mne.io.BaseRaw`
            An MNE Raw instance.
        """
        raw = mne.io.read_raw_edf(self.edf_file, preload=False, verbose='error')
        if include=='all': include = raw.info['ch_names']
        raw = raw.pick_channels(include)
        if preload:
            raw = raw.load_data()
        return raw

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
        events = import_events_from_xml(self.xml_file)
        return events

    def summary(self):
        raise NotImplementedError

def import_stages_from_xml(xml_file):
    """
    Read compumedics .xml and extract sleep staging informations

    Parameters
    ----------
    xml_file : str
        path fo the xml file to read

    Returns
    -------
    stages : pd.DataFrame
        A dataframe with sleep staging (label, duration and onset)
        informations.

    Notes
    -------
    By default compumedics labels 'Unsure" epochs as 9. This was changed to -1.
    If any stages is labelled as 4 (sleep stage 4), we re-label it as sleep
    stage 3 (R&K conversion to AASM).
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    stages = pd.DataFrame()

    label = np.asarray([child.text for child in root.findall("./SleepStages/SleepStage")], dtype='int')
    if np.sum(label==4)>0:
        label[label==4] = 3
    label[label==9] =-1
    stages['label'] = label
    stages['duration'] = np.ones_like(label) * \
                         np.asarray([child.text for child in
                                     root.findall("./EpochLength")],dtype='int')
    stages['onset']  = np.cumsum(stages['duration'].values) - \
                       np.asarray([child.text for child in
                                   root.findall("./EpochLength")],dtype='int')
    return stages

def import_events_from_xml(xml_file):
    """
    Read compumedics .xml and extract events informations

    Parameters
    ----------
    xml_file : str
        path fo the xml file to read

    Returns
    -------
    stages : pd.DataFrame
        A dataframe with events (label, duration and onset)
        informations.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    events = pd.DataFrame()

    events['EVT_LABEL'] = \
        [child.text for child in root.findall("./ScoredEvents/ScoredEvent/Name")]
    events['EVT_TIME'] = \
        np.asarray([child.text for child in
                    root.findall("./ScoredEvents/ScoredEvent/Start")],
                                 dtype='float')
    events['EVT_LENGTH'] = np.asarray([child.text for child in root.findall("./ScoredEvents/ScoredEvent/Duration")],
                              dtype='float')
    return events