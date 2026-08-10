import importlib.util
import logging
import logging.config
import os
import random
import re
import shutil
import sys
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*deprecated.*")

import torch
import torch.backends.cudnn as cudnn
import yaml
from torch import nn

PYNVML_AVAILABLE = importlib.util.find_spec("pynvml") is not None
if PYNVML_AVAILABLE:
    import pynvml

from src.main.datasets.dataprocess.convert import sessions_num_to_text
from src.main.datasets.datasetsprocess import bcicIV2a, OpenBMI
from src.main.models.ADFCNN import ADFCNN
from src.main.models.EEGConformer import Conformer
from src.main.models.EEGNet import EEGNet
from src.main.models.FBMSNet import FBMSNet
from src.main.models.IFNet import IFNet
from src.main.models.ShallowConvNet import ShallowConvNet
from src.main.train.utils.files import replace_first_zero

'''
#############################################################################
# 路径操作
#############################################################################
'''


def get_project_rootpath():
    path = os.path.realpath(os.curdir)
    while True:
        entries = os.listdir(path)
        if '.git' in entries or 'requirements.txt' in entries:
            return path
        if path == os.path.dirname(path):
            return path
        path = os.path.dirname(path)


def get_gpu_model_number():
    if not PYNVML_AVAILABLE:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        pynvml.nvmlShutdown()

        match = re.search(r'\d+', gpu_name)
        if match:
            return match.group()
        return None
    except Exception:
        return None


def add_single_config_path(cur_path, training_config_paths):
    root_path = get_project_rootpath()
    single_config_path = os.path.join(root_path, cur_path)
    training_config_paths.append(single_config_path)
    return training_config_paths


def config_paths(dataset_name, model_name, mode_name, file_name):
    trainging_config_paths = []
    trainging_config_paths = add_single_config_path("configs/identity_config.yaml", trainging_config_paths)
    trainging_config_paths = add_single_config_path(f"configs/datasets/{dataset_name}.yaml", trainging_config_paths)
    trainging_config_paths = add_single_config_path(f"configs/models/{model_name}.yaml", trainging_config_paths)
    trainging_config_paths = add_single_config_path(f"configs/modes/{mode_name}.yaml", trainging_config_paths)
    trainging_config_paths = add_single_config_path(f"configs/files/{file_name}.yaml", trainging_config_paths)
    return trainging_config_paths


def read_yaml(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file at path {path} does not exist.")

    with open(path, 'r', encoding='utf-8') as file:
        configs = yaml.safe_load(file)

    return configs


def write_yaml(training_configs):
    path = (training_configs['train']['logs']['save_path'] + '/'
            + training_configs['train']['logs']['name'] + '.yaml')
    with open(path, 'w') as yaml_file:
        yaml.dump(training_configs, yaml_file, default_flow_style=False)


def updata_configs(configs, subconfig):
    for key, value in subconfig.items():
        if isinstance(value, dict) and key in configs:
            updata_configs(configs[key], value)
        else:
            configs[key] = value
    return configs


def read_yamls(training_config_paths):
    training_configs = {}
    for config_path in training_config_paths:
        training_configs = updata_configs(training_configs, read_yaml(config_path))

    train_dataset_network_cl_type_name = (training_configs['dataset']['name'] + '_' +
                                          training_configs['model']['name'] + '_' +
                                          training_configs['train']['mode'])
    training_configs['train']['logs']['name'] = train_dataset_network_cl_type_name
    training_configs['train']['logs']['log_name'] = train_dataset_network_cl_type_name + '.log'
    training_configs['train']['logs']['model_name'] = train_dataset_network_cl_type_name + '.pth'
    training_configs['train']['logs']['result_name'] = (
            '00_log_results_of_' + train_dataset_network_cl_type_name + '.xlsx')
    training_configs['train']['logs']['k_fold_acc_result_name'] = (
            'k_fold_mean_max_acc_of_' + train_dataset_network_cl_type_name + '.xlsx')
    training_configs['train']['logs']['session_acc_result_name'] = (
            'session_mean_max_acc_of_' + train_dataset_network_cl_type_name + '.xlsx')
    training_configs['train']['logs']['subject_acc_result_name'] = (
            'subject_mean_max_acc_of_' + train_dataset_network_cl_type_name + '.xlsx')

    return training_configs


def extract_folder_number(folder_name):
    match = re.search(r'_(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return None


def find_max_folder_number_in_directory(directory):
    today = datetime.now().strftime("%Y%m%d")
    max_number = -1
    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)):
            if today in item:
                folder_number = extract_folder_number(item)
                if folder_number is not None:
                    max_number = max(max_number, folder_number)
    return max_number


def create_log_folder(path, model_name='', dataset_name='', subject_id=None):
    base_path = get_project_rootpath() + path
    month = datetime.now().strftime("%Y%m")
    base_path = os.path.join(base_path, month)
    today = datetime.now().strftime("%Y%m%d")
    base_path = os.path.join(base_path, today)
    os.makedirs(base_path, exist_ok=True)

    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    parts = [timestamp]
    if model_name:
        parts.append(model_name)
    if dataset_name:
        parts.append(dataset_name)
    if subject_id is not None:
        parts.append(str(subject_id))
    folder_name = "_".join(parts)
    folder_path = os.path.join(base_path, folder_name)

    os.makedirs(folder_path, exist_ok=True)
    save_path = path + "/" + month + "/" + today + "/" + folder_name
    return save_path


def copy_file(source_path, target_dir, file_name):
    os.makedirs(target_dir, exist_ok=True)

    destination_path = os.path.join(target_dir, file_name)

    shutil.copyfile(source_path, destination_path)

    print(f"文件已复制到 {destination_path}")


def get_formatted_current_time():
    current_datetime = datetime.now()
    formatted_time = current_datetime.strftime('%Y%m%d%H%M%S')
    return formatted_time


def create_log(training_configs):
    root_path = get_project_rootpath()
    with open(root_path + r'/src/main/configs/log_config.yaml', 'r') as f:
        config = yaml.safe_load(f.read())
        log_path = training_configs['train']['logs'][
                       'save_path'] + '/' + get_formatted_current_time() + '_' + training_configs['train']['logs'][
                       'log_name']
        config['handlers']['fileHandler']['filename'] = log_path
        logging.config.dictConfig(config)

    logger = logging.getLogger('sampleLogger')

    return logger, log_path


def add_suffix_to_folder(folder_path, suffix):

    directory, original_name = os.path.split(folder_path)

    new_name = original_name + suffix

    new_folder_path = os.path.join(directory, new_name)

    shutil.move(folder_path, new_folder_path)


def get_gpu_info(logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(device)
        total_memory = props.total_memory / 1073741824
        logger.info("Current GPU is {}, and the total memery is {} GB".format(gpu_name, total_memory))
    else:
        logger.info('No gpu device available')
        sys.exit(1)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def check_and_create_yaml(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(file_path):
        data = {'Info': 'AutoCreate'}
        with open(file_path, 'w') as file:
            yaml.dump(data, file)
        print(f"文件 {file_path} 已创建并写入内容: {data}")
    else:
        print(f"文件 {file_path} 已存在")


def load_configs(training_config_paths, subject_id=None, log_save_dir=None,
                 copy_model=True):
    training_configs = read_yamls(training_config_paths)

    if log_save_dir:
        os.makedirs(log_save_dir, exist_ok=True)
        training_configs['train']['logs']['save_path'] = log_save_dir
    else:
        model_name = training_configs.get('model', {}).get('name', '')
        dataset_name = training_configs.get('dataset', {}).get('name', '')
        training_configs['train']['logs']['save_path'] = create_log_folder(
            training_configs['train']['logs']['save_path'],
            model_name=model_name,
            dataset_name=dataset_name,
            subject_id=subject_id,
        )
        training_configs['train']['logs']['save_path'] = get_project_rootpath() + training_configs['train']['logs'][
            'save_path']

    write_yaml(training_configs)
    if copy_model:
        model_path = os.path.join(get_project_rootpath(), 'src', 'models',
                                  replace_first_zero(training_configs['model']['name']) + '.py')
        model_name = training_configs['model']['name'] + '.py'
        copy_file(model_path, training_configs['train']['logs']['save_path'], model_name)

    logger, log_path = create_log(training_configs)
    training_configs['train']['logs']['log_path'] = log_path

    get_gpu_info(logger)

    if training_configs['train']['random']['flag']:
        setup_seed(training_configs['train']['random']['seed'])

    cudnn.benchmark = training_configs['train']['accelerate']['benchmark']
    cudnn.enabled = training_configs['train']['accelerate']['CuDNN_enabled']


    return training_configs, logger


def load_data(training_configs, logger):
    root_path = get_project_rootpath()
    path = os.path.join(root_path, training_configs['dataset']['path'])
    subject = training_configs['subject_id']
    split_time = training_configs['preprocess']['time_window']
    resample_rate = training_configs['preprocess']['resample_rate']

    if training_configs['train']['mode'] != 'Cl_FS':
        raise ValueError(f"Unsupported train mode: {training_configs['train']['mode']}. "
                         f"Only 'Cl_FS' is supported.")

    if training_configs['dataset']['name'] == 'bcicIV2a':
        dataset_type = sessions_num_to_text(training_configs['session'],
                                            training_configs['dataset']['session_mapping'])
        training_configs['dataset_type'] = dataset_type
        data, label = bcicIV2a.load_data(path, subject, dataset_type, split_time, resample_rate)
    elif training_configs['dataset']['name'] == "OpenBMI":
        dataset_type = training_configs['dataset_type']
        data, label = OpenBMI.load_single_subject_data(path, subject, dataset_type, split_time, resample_rate)


    if training_configs.get('train', {}).get('datapercent') is not None and training_configs['dataset_type'] == "E":
        if training_configs['train']['datapercent'] > 0:
            print(f"数据已经被截取，截取比例为{training_configs['train']['datapercent']*100}%")
            data = data[:int(len(data) * training_configs['train']['datapercent'])]
            label = label[:int(len(label) * training_configs['train']['datapercent'])]
        else:
            print(f"数据已经被截取，截取阈值前10个")
            data = data[:10]
            label = label[:10]
    return data, label


def load_model(training_configs):
    num_classes = training_configs['dataset']['classes']
    num_channels = training_configs['dataset']['channels'] * len(training_configs['preprocess']['filter_bank'])
    num_samples = training_configs['preprocess']['time_point']

    if training_configs['model']['name'] == 'ShallowConvNet':
        model = ShallowConvNet(
            n_classes=num_classes,
            ch_nums=num_channels,
            F1=training_configs['model']['F1'],
            T1=training_configs['model']['T1'],
            F2=training_configs['model']['F2'],
            P1_T=training_configs['model']['P1_T'],
            P1_S=training_configs['model']['P1_S'],
            drop_out=training_configs['model']['drop_out'],
            pool_mode=training_configs['model']['pool_mode'],
            weight_init_method=training_configs['model']['weight_init_method'],
            last_dim=training_configs['model']['last_dim']
        )
    elif training_configs['model']['name'] == 'EEGNet':
        model = EEGNet(
            num_classes=num_classes,
            num_channels=num_channels,
            num_samples=num_samples,
            F1=training_configs['model']['F1'],
            F2=training_configs['model']['F2'],
            D=training_configs['model']['D'],
            kernel_length_1=training_configs['model']['kernel_length_1'],
            kernel_length_2=training_configs['model']['kernel_length_2'],
            dropout_rate=training_configs['model']['dropout_rate'],
            last_dim=training_configs['model']['last_dim'],
        )
    elif training_configs['model']['name'] == 'EEGConformer':
        model = Conformer(
            emb_size=training_configs['model']['emb_size'],
            depth=training_configs['model']['depth'],
            n_classes=num_classes,
            channel_size=num_channels,
            last_dim=training_configs['model']['last_dim']
        )
    elif training_configs['model']['name'] == 'FBMSNet':
        model = FBMSNet(
            nChan=training_configs['dataset']['channels'],
            nTime=num_samples,
            nClass=num_classes,
            nBands=training_configs['model']['nBands'],
        )
    elif training_configs['model']['name'] == 'IFNet':
        model = IFNet(
            num_channels=num_channels,
            num_classes=num_classes,
            out_channels=training_configs['model']['out_channels'],
            radix=training_configs['model']['radix'],
            kernel_size=training_configs['model']['kernel_size'],
            patch_size=training_configs['model']['patch_size'],
            Lineat_input=training_configs['model']['Lineat_input']
        )
    elif training_configs['model']['name'] == 'ADFCNN':
        model = ADFCNN(
            num_channels=num_channels,
            nClass=num_classes,
            last_dim=training_configs['model']['last_dim'],
        )
    elif training_configs['model']['name'] == 'CTNet':
        from models.CTNet import CTNet
        model = CTNet(
            eeg1_number_channel=num_channels, 
            number_class=num_classes, 
            flatten_eeg1=training_configs['model']['emb_size'] * num_samples
        )
    return model


def load_optimizer(training_configs, model):
    opt_name = training_configs['train']['optimizer']['name']
    opt_cfg = training_configs['train']['optimizer']
    lr = opt_cfg['lr']

    if opt_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=opt_cfg.get('weight_decay', 0.01),
        )
    elif opt_name == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=opt_cfg.get('momentum', 0.9),
            weight_decay=opt_cfg.get('weight_decay', 0),
        )
    else:
        raise ValueError(f"Unsupported optimizer: '{opt_name}'. Supported: Adam, AdamW, SGD")

    return optimizer


def load_loss(training_configs):
    loss_name = training_configs['train']['loss']['name']
    if loss_name == 'CrossEntropyLoss':
        loss_func = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: '{loss_name}'. Supported: CrossEntropyLoss")

    return loss_func


if __name__ == '__main__':
    training_config_paths = config_paths("bcicIV2a", "EEGNet", "Cl_FS", "bcicIV2a/bcicIV2a_EEGNet_Cl_FS")
    training_configs, logger = load_configs(training_config_paths)
    print("test")
