import logging
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.main.datasets.dataprocess.convert import array_to_tensordataset
from src.main.datasets.dataprocess.filters import filter_bank
from src.main.train.utils.configs import get_project_rootpath, load_data, load_model, load_optimizer, load_loss, load_configs, \
    add_suffix_to_folder
from src.main.train.utils.files import wait_until_folder_available
from src.main.train.utils.results import save_results
from src.main.train.utils.training import model_train, model_test


def save_peak_para_and_net_of_model(training_configs, model):
    if not training_configs.get('train', {}).get('save_weights', False):
        return
    save_model_path = training_configs['train']['logs']['save_path'] \
                      + '/P_' \
                      + '_' + 'Sub' + str(training_configs['subject_id']) \
                      + '_' + training_configs['train']['logs']['model_name']
    torch.save(model.state_dict(), save_model_path)


def save_final_para_and_net_of_model(training_configs, model):
    if not training_configs.get('train', {}).get('save_weights', False):
        return
    save_model_path = training_configs['train']['logs']['save_path'] \
                      + '/' + 'Sub' + str(training_configs['subject_id']) \
                      + '_' + training_configs['train']['logs']['model_name']
    torch.save(model.state_dict(), save_model_path)

def check_saved_model_type(file_path):
    try:
        obj = torch.load(file_path)
        if isinstance(obj, nn.Module):
            return "model"
        elif isinstance(obj, dict) and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in obj.items()):
            return "state_dict"
        return "unknown"
    except:
        return "error"

def train_single_subject_Cl_FS(training_configs, logger):
    training_configs['session'] = 0
    training_configs['dataset_type'] = "T"
    print("Loading training dataset")
    train_data, train_label = load_data(training_configs, logger)
    training_configs['session'] = 1
    training_configs['dataset_type'] = "E"
    print("Loading test dataset")
    test_data, test_label = load_data(training_configs, logger)

    train_data = filter_bank(train_data, filter_banks=training_configs['preprocess']['filter_bank'])
    test_data = filter_bank(test_data, filter_banks=training_configs['preprocess']['filter_bank'])


    train_dataset, test_dataset = array_to_tensordataset(train_data, train_label, test_data, test_label)
    train_loader = DataLoader(dataset=train_dataset, batch_size=training_configs['train']['training_batchsize'],
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=training_configs['train']['test_batchsize'],
                             shuffle=True)

    model = load_model(training_configs)
    optimizer = load_optimizer(training_configs, model)
    loss_func = load_loss(training_configs)

    max_epoch = training_configs['train']['max_epochs']

    if torch.cuda.is_available():
        try:
            model.cuda()
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"Warning: Failed to move model to GPU: {e}")
            print("Falling back to CPU")
    else:
        print("CUDA not available, using CPU")

    all_acc = []
    max_acc = 0.

    if training_configs.get('train', {}).get('resume', {}).get('mode') == 'test':
        resume_path = training_configs['train']['resume']['path'][training_configs['subject_id']]

        file_type = check_saved_model_type(resume_path)

        if file_type == "state_dict":
            model.load_state_dict(torch.load(resume_path))
        elif file_type == "model":
            model = torch.load(resume_path).to(next(model.parameters()).device)
        else:
            raise RuntimeError(f"无法加载模型文件: {resume_path} (类型: {file_type})")

        cur_acc = model_test(training_configs, test_loader, model, 0, logger)
        all_acc.append(cur_acc)
        save_peak_para_and_net_of_model(training_configs, model)
        return

    for epoch in range(max_epoch):
        model_train(training_configs, train_loader, model, loss_func, optimizer, epoch, logger)
        cur_acc = model_test(training_configs, test_loader, model, epoch, logger)
        all_acc.append(cur_acc)
        if cur_acc >= max_acc:
            max_acc = cur_acc
            save_peak_para_and_net_of_model(training_configs, model)
    save_final_para_and_net_of_model(training_configs, model)


def train_Cl_FS(training_config_path):
    training_configs, logger = load_configs(training_config_path)
    for subject_id in range(1, training_configs['dataset']['subjects'] + 1):
        if training_configs.get('dataset', {}).get('choosen_subs'):
            if subject_id not in training_configs['dataset']['choosen_subs']:
                continue
        print(training_configs['train']['logs']['name'])
        logger.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        logger.info(f"The {subject_id}th subject")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"The {subject_id}th subject")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        logger.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        logger.info("============================================================================")
        logger.info(f"The 0th session of the {subject_id}th subject")
        logger.info("============================================================================")
        logger.info("============================================================================")
        logger.info(f"The 0-fold cross-validation, and the 0th session of the {subject_id}th subject")
        logger.info("============================================================================")
        training_configs['subject_id'] = subject_id

        start = time.perf_counter()
        train_single_subject_Cl_FS(training_configs, logger)
        end = time.perf_counter()
        execution_time_ms = end - start
        print("-" * 60)
        print(f"print 执行时间: {execution_time_ms:.2f} 秒")
        print("-" * 60)

    save_results(training_configs)

    logger.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    logger.info("The trainging is finish")
    logger.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    logging.shutdown()

    wait_until_folder_available(training_configs['train']['logs']['save_path'])
    suffix = '_' + training_configs['model']['name'] + '_' + training_configs['dataset']['name'] + '_' + \
             training_configs['train']['mode'] + '_finish'
    add_suffix_to_folder(training_configs['train']['logs']['save_path'], suffix)


class TEST(object):

    @staticmethod
    def test_train_Cl_FS():
        pass


if __name__ == '__main__':
    TEST.test_train_Cl_FS()
