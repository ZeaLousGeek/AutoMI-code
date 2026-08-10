import os
import platform

import mne
import numpy as np
import scipy

from src.main.train.utils.files import find_directory


def label2events(label, label_pos):
    events = [[label_pos[i], 0, int(label[i])] for i in range(len(label))]
    events = np.array(events)
    event_dict = {'1': 1, '2': 2}
    type_set = {str(item[2]) for item in events}
    new_event_dict = {k: v for k, v in event_dict.items() if k in type_set}

    return events, new_event_dict


def load_session_data(
        data_raw_mat,
        split_time=[0, 4],
        resample_rate=250):
    mne.set_log_level('warning')
    data_mat = data_raw_mat['x']
    srate = data_raw_mat['fs'][0, 0]
    cue_pos = data_raw_mat['t'].squeeze()
    y_class = data_raw_mat['y_class']
    trial_label = data_raw_mat['y_dec'].squeeze()

    info = mne.create_info(
        ch_names=[item[0] for item in data_raw_mat['chan'].squeeze().tolist()],
        ch_types="eeg",

        sfreq=srate
    )
    data_raw = mne.io.RawArray(data_mat.T, info)
    data_raw.resample(resample_rate)
    for i in range(len(cue_pos)):
        cue_pos[i] = int(cue_pos[i] / (srate / resample_rate))

    events, event_id = label2events(trial_label, cue_pos)
    tmin, tmax = 0, 4

    picks = mne.pick_types(data_raw.info, meg=False, eeg=True, stim=False, eog=False,
                           exclude='bads')
    epochs = mne.Epochs(data_raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks, baseline=(0, 0), preload=True)

    data = epochs.get_data()

    first_pos = 1 if split_time[0] == 0 else 0
    start = int(resample_rate * split_time[0]) + first_pos
    end = int(resample_rate * split_time[1]) + first_pos
    data = data[:, :, start:end]

    event_labels = epochs.events[:, 2]
    label = (event_labels == 1).astype(int)[:, np.newaxis]

    return data, label

def load_data(path=None,
              subject=1,
              session=1,
              dataset_type="T",
              split_time=[0, 4],
              resample_rate=250):
    if path is None:
        root = os.environ.get("AUTOMI_DATA_ROOT")
        if root:
            path = os.path.join(root, 'OpenBMI', 'mat')
        else:
            base = find_directory(target_folder='OpenBMI')
            if base is None:
                raise FileNotFoundError("Set AUTOMI_DATA_ROOT or place data at <scanned_root>/OpenBMI/mat")
            path = os.path.join(base, 'OpenBMI', 'mat')
    path = path + '/sess' + str(session).zfill(2) + '_subj' + str(subject).zfill(2) + '_EEG_MI.mat'
    data_raw = scipy.io.loadmat(path)

    EEG_MI_train = data_raw['EEG_MI_train'][0, 0]
    EEG_MI_test = data_raw['EEG_MI_test'][0, 0]

    if dataset_type == "T":
        data, label = load_session_data(EEG_MI_train, split_time, resample_rate)
    elif dataset_type == "E":
        data, label = load_session_data(EEG_MI_test, split_time, resample_rate)
    elif dataset_type == "All":
        train_data, train_label = load_session_data(EEG_MI_train, split_time, resample_rate)
        test_data, test_label = load_session_data(EEG_MI_test, split_time, resample_rate)
        data = np.concatenate((train_data, test_data), 0)
        label = np.concatenate((train_label, test_label), 0)

    return data, label


def load_single_subject_data(path=r"data\OpenBMI\mat",
                             subject=1,
                             dataset_type='T',
                             split_time=[0, 4],
                             resample_rate=250
                             ):
    data = []
    label = []
    for i in range(1, 3):
        cur_data, cur_label = load_data(path, subject, i, dataset_type, split_time, resample_rate)
        if i == 1:
            data = cur_data
            label = cur_label
        else:
            data = np.concatenate((data, cur_data), 0)
            label = np.concatenate((label, cur_label), 0)

    return data, label


class TEST(object):

    @staticmethod
    def test_load_data():
        for subject in range(1, 55):
            for session in range(1, 3):
                print(subject, session)
                load_data(subject=subject, session=session)


if __name__ == '__main__':
    for i in range(1,55):
        print(i)
        data, label = load_data(subject=i, session=2, dataset_type="All")
        print("test")


