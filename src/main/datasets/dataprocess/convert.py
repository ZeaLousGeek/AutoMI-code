import numpy as np
import torch
from torch.utils.data import TensorDataset


def array_to_tensordataset(train_data, train_label, test_data, test_label):
    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32),
                                  torch.tensor(train_label, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32),
                                 torch.tensor(test_label, dtype=torch.long))

    return train_dataset, test_dataset


def events_class(events_from_annot, event_dict, label, types_to_check=['769', '770', '771', '772'], dataset_type="T"):
    type = [event_dict[t] for t in types_to_check if t in event_dict]

    events_from_annot = events_from_annot[np.isin(events_from_annot[:, 2], type)]
    if dataset_type == "T":
        new_event_dict = {k: v for k, v in event_dict.items() if v in type}
        return events_from_annot, new_event_dict
    elif dataset_type == "E":
        events_from_annot[:, 2] = label[:len(events_from_annot)].flatten()
        return events_from_annot


def events_class_T(events_from_annot, event_dict, types_to_check=['769', '770', '771', '772']):
    type = [event_dict[t] for t in types_to_check if t in event_dict]

    events_from_annot = events_from_annot[np.isin(events_from_annot[:, 2], type)]
    new_event_dict = {k: v for k, v in event_dict.items() if v in type}

    return events_from_annot, new_event_dict


def events_class_E(events_from_annot, event_dict, label, types_to_check=['783']):
    type = [event_dict[t] for t in types_to_check if t in event_dict]

    events_from_annot = events_from_annot[np.isin(events_from_annot[:, 2], type)]
    events_from_annot[:, 2] = label[:len(events_from_annot)].flatten()

    return events_from_annot

def sessions_num_to_text(session, session_mapping):

    if session in session_mapping:
        session_id = session_mapping[session]
        return session_id
    else:
        raise ValueError("The session id is error, please check dataset-sessions value in yaml")

def test():
    pass


if __name__ == '__main__':
    test()
