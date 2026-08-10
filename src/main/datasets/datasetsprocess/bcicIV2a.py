import os.path

import mne
import numpy as np
import scipy.io
import platform

from src.main.datasets.dataprocess.convert import events_class_T, events_class_E
from src.main.train.utils.files import find_directory



def load_data(path=None,
              subject=1,
              dataset_type="E",
              split_time=[0.5, 3.5],
              resample_rate=250
              ):
    if path is None:
        root = os.environ.get("AUTOMI_DATA_ROOT")
        if root:
            path = os.path.join(root, 'bcicIV2a', 'gdf')
        else:
            base = find_directory(target_folder='bcicIV2a')
            if base is None:
                raise FileNotFoundError("Set AUTOMI_DATA_ROOT or place data at <scanned_root>/bcicIV2a/gdf")
            path = os.path.join(base, 'bcicIV2a', 'gdf')
    data_raw_path = path + "/A0" + str(subject) + str(dataset_type) + ".gdf"
    mne.set_log_level('warning')
    data_raw = mne.io.read_raw_gdf(data_raw_path)
    data_raw.resample(resample_rate)
    events_from_annot, event_dict = mne.events_from_annotations(data_raw)

    if dataset_type == 'T':
        events, event_id = events_class_T(events_from_annot, event_dict)
    elif dataset_type == 'E':
        data_raw_mat_path = path + "/IV2a_true_labels/A0" + str(subject) + dataset_type + ".mat"
        data_raw_mat = scipy.io.loadmat(data_raw_mat_path)
        label = data_raw_mat['classlabel']
        events = events_class_E(events_from_annot, event_dict, label)
        event_id = {'769': 1, '770': 2, '771': 3, '772': 4}
    tmin, tmax = 0, 4

    picks = mne.pick_types(data_raw.info, meg=False, eeg=True, stim=False, eog=False,
                           exclude='bads')
    epochs = mne.Epochs(data_raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks, baseline=(0, 0), preload=True)
    epochs_num = len(epochs)

    epochs_data_frame = epochs.to_data_frame().drop(columns=['EOG-left', 'EOG-central', 'EOG-right'])

    data = []
    label = []
    for i in range(epochs_num):
        cur_data = epochs_data_frame.loc[epochs_data_frame['epoch'] == i, :]
        cur_label = cur_data.iat[len(cur_data) - 1, 1]
        cur_data = cur_data.drop(columns=['time', 'condition', 'epoch'])

        cur_data_array = np.array(cur_data).T
        first_pos = 1 if split_time[0] == 0 else 0
        cur_data_array = cur_data_array[np.newaxis, :,
                         (int(resample_rate * split_time[0]) + first_pos):(
                                     int(resample_rate * split_time[1]) + first_pos)]
        cur_label_array = np.asarray([int(cur_label) - 769])
        cur_label_array = cur_label_array[np.newaxis, :]

        if i == 0:
            data = cur_data_array
            label = cur_label_array
        else:
            data = np.concatenate((data, cur_data_array), 0)
            label = np.concatenate((label, cur_label_array), 0)

    return data, label


def load_data_Cl_FS(path, subject, **kwargs):
    X_train, y_train = load_data(path=path, subject=subject, dataset_type="T")
    X_test, y_test = load_data(path=path, subject=subject, dataset_type="E")

    if len(y_train.shape) == 2 and y_train.shape[1] == 1:
        y_train = np.squeeze(y_train)
    if len(y_test.shape) == 2 and y_test.shape[1] == 1:
        y_test = np.squeeze(y_test)

    return [(X_train, y_train, X_test, y_test)]
