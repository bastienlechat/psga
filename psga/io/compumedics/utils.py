import xml.etree.ElementTree as ET
import warnings
import pandas as pd
import sys, os
import configparser
import numpy as np
import glob

from ..utils import XmlDictConfig, XmlListConfig
pd.set_option('mode.chained_assignment', None)

MAPPING_EVENT = {
    '0':'Unknown',
    '1': 'central_apnea',
    '2': 'obstructive_apnea',
    '3': 'mixed_apnea',
    '4': 'desat',
    '5': 'respiratory_artefact',
    '6': 'spo2_artefact',
    '7': 'arousal_t1',
    '8': 'arousal_t2',
    '9': 'arousal_t3',
    '10': 'arousal_t4',
    '11': 'arousal_t5',
    '12': 'limb_left',
    '13':'limb_right',
    '14':'bradycardia',
    '15':'tachycardia',
    '16': 'tco2_artifact',
    '17': 'etco2_artifact',
    '18': 'distal_ph_artifact',
    '19': 'distal_ph_event',
    '20': 'proximal_ph_artifact',
    '21': 'proximal_ph_event',
    '22': 'blood_pressure_artifact',
    '23': 'body_temp_artifact',
    '24': 'unsure_resp_event',
    '25': 'resp_paradox',
    '26': 'periodic_breathing',
    '27': 'PLM_episode',
    '28': 'heart_rate_artifact',
    '29': 'obstructive_hypopnea',
    '30': 'central_hypopnea',
    '31': 'mixed_hypopnea',
    '32': 'RERA',
    '33': 'snore_event',
    '34': 'user_event_1',
    '35': 'user_event_2',
    '36': 'user_event_3',
    '37': 'user_event_4',
    '38': 'user_event_5',
    '39': 'user_resp_event_1',
    '40': 'user_resp_event_2',
    '41': 'user_resp_event_3',
    '42': 'user_resp_event_4',
    '43': 'user_resp_event_5',
    '44': 'delta_wave',
    '45': 'spindles',
    '46': 'left_eye_movement',
    '47': 'left_eye_movement_anti_phase',
    '48': 'left_eye_movement_phase',
    '49': 'right_eye_movement',
    '50': 'right_eye_movement_anti_phase',
    '51': 'right_eye_movement_phase',
    '52': 'PTT_event',
    '53': 'PTT_artifact',
    '54': 'asystole',
    '55': 'wide_complex_tachycardia',
    '56': 'narrow_complex_tachycardia',
    '57': 'atrial_fibrilation',
    '58': 'bruxism',
    '59': 'SMA',
    '60': 'TMA',
    '61': 'rythmic_movement',
    '62': 'ECG_artifact',
    '63': 'CAP_A1',
    '64': 'CAP_A2',
    '65': 'CAP_A3',
    '66': 'PES_artifact',
    '67': 'CPAP_artifact',
    '68': 'user_event_6',
    '69': 'user_event_7',
    '70': 'user_event_8',
    '71': 'user_event_9',
    '72': 'user_event_10',
    '73': 'user_event_11',
    '74': 'user_event_12',
    '75': 'user_event_13',
    '76': 'user_event_14',
    '77': 'user_event_15',
    '78':'transient_muscle_activity',
    '79':'hypnagogic_foot_tremor',
    '80': 'hypnagogic_foot_tremor_burst_left',
    '81': 'hypnagogic_foot_tremor_burst_right',
    '82': 'excessive_fragmentary_myolonus',
    '83': 'alternating_leg_muscle_activation',
    '84': 'rythmic_movement_burst',
    '85': 'hyperventilation',
    '86': 'excessive_fragment_myolonus_burst_left',
    '87': 'excessive_fragment_myolonus_burst_right',
    '88': 'hypoventilation',
}


def cpm_list_event():
    event_list  = [val for _,val in MAPPING_EVENT.items()]
    return event_list


def _read_epoch_data(folder):
    """
    Read PROCESS.ADV file in the compumedics folder.

    Compumedics automatically run some epochs level analysis when recording (or
    maybe when closing file?) such as spindles detection etc.. which is then
    saved in Process.ADV file.

    Returns
    -------
    epochdata : pd.DataFrame
        30-seconds epoch-level of summary data (e.g. heart rate)

    Notes
    ------
    Some parameters (e.g. U15) have not yet been figured out.
    """

    process_file = os.path.join(folder, 'PROCESS.ADV')
    n = np.fromfile(process_file, dtype=np.int16)
    number_of_epochs = n[0]
    other_data = np.reshape(n[1:], (number_of_epochs, -1))

    columns_names = ['Artifact', 'DeltaL', 'DeltaM', 'DeltaH', 'ThetaD',
                     'ThetaA',
                     'Alpha', 'Sigma', 'Beta', 'U10', 'Spindles', 'MeanSAO2',
                     'MinSAO2',
                     'MaxSAO2', 'U15', 'U16', 'Sound', 'REM', 'EMGamp', 'U20',
                     'CPAP',
                     'U22', 'HR', 'U24', 'U25', 'Posture', 'U27', 'U28', 'U29',
                     'U30',
                     'KC', 'U32', 'U33']
    epochdata = pd.DataFrame(data=other_data, columns=columns_names)

    return epochdata

def _read_header(folder):
    """Reads compumedics header"""
    config_tree = ET.parse(os.path.join(folder, 'STUDYCFG.xml'))
    root = config_tree.getroot()
    header = {}
    for elem in root:
        if elem.tag not in ['Channels','ChannelGroups']:
            if elem.text is not None:
                header[elem.tag] = elem.text
    return header

def _read_montage(folder):
    """Reads the mapping between channel name and file name"""
    config_tree = ET.parse(os.path.join(folder, 'STUDYCFG.xml'))
    root = config_tree.getroot()
    montage = {}
    for ch in root.findall("./Channels/Channel"):
        param_dict = {}
        ch_label = 0
        for elem in ch:
            if elem.tag == 'Label':
                ch_label = elem.text
            else:
                param_dict[elem.tag] = elem.text
        if isinstance(ch_label,int):
            raise ValueError('Could not find channel name')
        else:
            montage[ch_label] = param_dict

    return montage

def _read_data_segments(folder):
    """
    DATASEGMENTS.xml file contains the beginning and duration of each
    recording block. More than 1 recording block can happen if recording is
    discontinued during the night (e.g. grael box is removed from recording
    system).
    Start and duration of each recording blocks is useful to 0-pad signals
    when recording was discontinued (gap between segments).

    Returns
    -------
    raw : list of dict
        For each block, a dict is created with the following
        keys: "Start", "Duration", "VideoServer", "VideoID"
    """
    ds_tree = ET.parse(os.path.join(folder, 'DATASEGMENTS.xml'))
    root = ds_tree.getroot()
    ds = root.findall("./DataSegment")
    segments = XmlListConfig(ds)
    if len(segments)==0:
        raise ValueError('No data segment found')
    elif len(segments) in [1,2]:
        pass
    else:
        warnings.warn('Recording contains more than 2 data segments, '
                      'not yet tested')
    return segments

def _read_event(folder):
    """
    Read events file. Sometime in 2020 compumedics shifted the storage file
    from .MDB to .EDB (a binary file). We haven't found a way of reading .MDB
    file easily in python yet so only .EDB is supported.

    For file structure of .EDB, see "read_EDB_event" function.
    """

    if len(glob.glob(os.path.join(folder, '*.EDB')))>0:
        events = read_EDB_event(glob.glob(os.path.join(folder, '*.EDB'))[-1])
        code = events['EVT_CODE'].astype(int).astype("string")
        code = [MAPPING_EVENT[code_num] for code_num in code.values]
        events['EVT_LABEL'] = code
    else:
        raise NotImplementedError('Only .EDB are supported for event file')

    return events

def _read_technician_note(folder):
    """
    Read the "EventsSx.dat" file in a given compumedic folder.  EventsSx.dat
    is a binary file that should be structured in block of 600 bits.

    Current guesses regarding block structure:
        - First 4 (1-4 assuming index=1) bytes might be padding/zeros –
        perhaps a long integer? (seems always to be 0)
        - Next 4 (5-9) maybe also a long integer with various numbers like 1,
         15, 6, 26 etc (it seems 15 is for trace property and 1 for tech notes)
        - Next 8 (9-16) is the time stamp (double precision)
        - Next 8 (17-24) unknown.
        - from 24 to 600: These bytes are reserved for text

    Parameters
    ----------

    folder : str
        Path of the compumedics folder

    Returns
    -------
    tech_notes : dict
        * onset : array
            Time (in seconds from the beginning) of the technician
            notes
        * event: str
            Corresponding event label

    Notes
    -----
    The structure described above does not work for some byte blocks. I think
    there is different file structure when techs are checking impedances. All
    technician notes (including user defined one) are coded according to the
    structure described above (this has been tested).
    """
    tech_notes = None
    if os.path.exists(os.path.join(folder, 'EventsSx.dat')):
        try:
            file = os.path.join(folder,'EventsSx.dat')
            n = np.fromfile(file, dtype=np.uint8).reshape(-1, 600)
            x = n[:, 0:8].copy().view(np.int64) #this always seems to be empty
            time_onset = n[:, 8:16].copy().view(np.double).squeeze()
            string_val = n[:, 24:].copy().view(np.byte)
            string_list = []
            for x in string_val:
                test = [chr(bin_val) for bin_val in x if bin_val > 0]
                string_val = ''.join([str(elem) for elem in test])
                string_list.append(string_val)
            #check that we have as many labels that onset
            assert len(string_list) == len(time_onset)

            tech_notes = {'onset': time_onset, 'event': string_list}
        except:
            raise ValueError('Could not read technician notes, '
                             'setting to None')
    else:
        raise NotImplementedError('Only EventsSx.dat are supported for event '
                                  'file, setting technician note to None')
    return tech_notes

def _read_sleep_stage(folder):
    """
    Read staging file from compumedics (uint8). It seems compumedics
    attributes uses the following mapping:
        [Stage]
        0= Unscored file
        1=1
        2=2
        3=3
        4=4
        5=REM
        10=Wake
        128=Unsure?
        138=Unscored?

    To be consistent with the way compumedics export their hypnogram in .xml
    (during .edf export), the following code are modified within this function:
        0 is changed to -1 (Unscored to -1)
        10 is changed to 0 (Wake = 0)
        128 is changed to 0 (Unsure is changed to 9, which is the code
        used to signify unscored in the .xml I think)
        138 is changed to 9 (Unscored? is changed to wake*)


    Parameters
    ----------
    folder : str
        path to compumedics folder

    Returns
    -------
    hypno : array (n)
        sleep stage hypnogram

    Notes
    -------
    * This seems like a strange choice, but this was deduce based on comparison
    between hypnogram in native compumedics format (read here) and the
    corresponding .xml exported using profusion 4.
    """
    list_staging = sorted(glob.glob(os.path.join(folder, 'SLP*.DAT')),
                           key=os.path.getmtime)

    if len(list_staging)==0:
        raise NotImplementedError('Could not find sleep staging file')
    elif len(list_staging)>1:
        warnings.warn('Multiple sleep stage scoring file were found, getting '
                      'the latest one.')
        fname = list_staging[-1]
    else:
        fname = list_staging[0]

    hypno = np.asarray(np.fromfile(fname, np.uint8), dtype='int')
    hypno[hypno == 0] = -1
    hypno[hypno == 10] = 0
    hypno[hypno == 128] = 9
    hypno[hypno == 138] = 0
    return hypno


def read_EDB_event(file):
    """
    Read the .EDB event file in a given compumedic folder.  Events are stored
    in a binary file that should be structured in block of 40 bits.

    Current guesses regarding block structure:
        - First 4 (1-4 assuming index=1) unsigned integers representing the
        events code
        - 4 to 8 is unknown
        - next 8 is a float representing event onset time
        - next 8 is a float representing event duration time
        - After that we are unsure of the structure. It seems the last bits
        is a integer that is a binary values for manual vs automatic scoring

    Parameters
    ----------

    folder : str
        Path of the compumedics folder

    Returns
    -------
    eventsframe : pd.DataFrame
        Dataframe containing events scoring
    """
    name_key = ['EVT_CODE','UNKNOWN1', 'EVT_TIME', 'EVT_LENGTH',
                'PARAM3','PARAM2', 'PARAM1','MAN_SCORED']
    dt = np.dtype({'names': name_key,
                   'formats': [np.uint32,np.uint32, np.float64,
                               np.float64,np.float64,np.int16,
                               np.int16,np.uint8],
                   'offsets': [0, 4, 8, 16,
                               24, 32, 34, 36]})

    with open(file, 'rb') as fid:
        b = fid.read()

    list_event = []
    for x in np.arange(0, len(b), 40):
        val = np.frombuffer(b, dt, count=1, offset=x)
        list_event.append(np.hstack(val[0]))
    events = np.vstack(list_event)
    eventsframe = pd.DataFrame(data=events,columns=name_key)

    return eventsframe

def read_dat_data(fname):
    """
    .DAT files are raw data stored without any compression. This was how most
    of the data was stored before introducing the .d16 format.

    Parameters
    ----------
    fname : str
        file to read

    Returns
    -------
    signal : np.array
        the data
    """
    signal = np.fromfile(fname, dtype=np.float32)
    return signal.squeeze()

def read_dat_discrete_data(fname, sf):
    """
    .DAT discrete files are used to store data such as heart rate, which are
    not necessarily continuous in time. These are stored in block of 8 where
    the first 4 bits represents the time (in points) of the data and the next
    8 represents the actual values.

    Parameters
    ----------
    fname : str
        file to read
    sf : int
        Sampling frequency

    Returns
    -------
    t : np.array
        Timing of the data (in seconds)
    v : np.array
        Values of the data
    """
    n = np.fromfile(fname, dtype=np.uint8).reshape(-1, 8)
    t = n[:, 0:4].copy().view(np.int) / sf
    v = n[:, 4:8].copy().view(np.float32)

    return t.squeeze(), v.squeeze()

def read_d16_data(fname, sf):
    """
    Read the .D16 raw data file in a given compumedic folder. ".D16" files are
    coded using some kind of lossy compression that uses the first bits to
    store a scale/offset as floats and subsequent datapoints as integer.
    Integers are then rescale using the scale/offset.

    The block size depends on the sampling frequency of the signal. and
    corresponds to a 1 seconds block size (of int16) + 8 bits (4 for scale
    and 4 for offset)

    Parameters
    ----------
    fname : str
        file to read
    sf : int
        Sampling frequency

    Returns
    -------
    signal : np.array
        the signal

    Notes
    -----
    For discontinuous recordings, one will have to pad with zeros the signal
    using the different data segments. See _read_data_segments for more
    information.
    """
    next_shape = 4 + 4 + 2 * sf
    n = np.fromfile(fname, dtype=np.uint8).reshape(-1, next_shape)
    OFFSET = n[:, 0:4].copy().view(np.float32) * -1
    SCALE = n[:, 4:8].copy().view(np.float32)
    LEVEL = n[:, 8:].copy().view(np.int16) * -1
    cumsum_level = np.cumsum(LEVEL, axis=1)
    mat_signal = np.add(OFFSET, np.multiply(SCALE, cumsum_level))
    signal = np.reshape(mat_signal, (-1, 1))
    return signal.squeeze()