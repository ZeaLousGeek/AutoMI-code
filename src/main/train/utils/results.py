import os
import shutil
import numpy as np
import pandas as pd
import torch
from scipy import stats
import logging
import logging.config
import re
import yaml
from scipy.stats import ttest_rel
from statsmodels.stats.weightstats import ztest
from thop import profile
from tqdm import tqdm
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis
'''
#############################################################################
# log文件操作
#############################################################################
'''

def extract_info_from_log(training_configs):
    log_path = training_configs['train']['logs']['log_path']

    pattern = r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+ - sampleLogger - INFO\] The (\d+)-fold cross-validation, and the (\d+)th session of the (\d+)th subject'
    pattern_loss = r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+ - sampleLogger - INFO\] Epoch:\[\d+/\d+\]\t\| train loss: ([-+]?\d*\.?\d+)'
    pattern_accuracy = (r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+ - sampleLogger - INFO\] Epoch:\[\d+/\d+\]\t\| Test Accuracy: ([-+]?\d*\.?\d+)'
                        r'\t\| Precision: ([-+]?\d*\.?\d+)\t\| Recall: ([-+]?\d*\.?\d+)\t\| F1 Score: ([-+]?\d*\.?\d+)\t\| Kappa: ([-+]?\d*\.?\d+)')

    results = {}
    current_subject = None
    current_session = None
    current_fold = None

    with open(log_path, 'r') as file:
        for line in file:
            match_pattern = re.search(pattern, line)
            match_loss = re.search(pattern_loss, line)
            match_accuracy = re.search(pattern_accuracy, line)
            if match_pattern:
                fold = match_pattern.group(1)
                session = match_pattern.group(2)
                subject = match_pattern.group(3)
                current_subject = subject
                current_session = session
                current_fold = fold

                if subject not in results:
                    results[subject] = {}

                if session not in results[subject]:
                    results[subject][session] = {}

                if fold not in results[subject][session]:
                    results[subject][session][fold] = {
                        'loss': [],
                        'accuracy': [],
                        'precision': [],
                        'recall': [],
                        'f1_score': [],
                        'kappa': []
                    }


            elif match_loss and current_subject and current_session and current_fold:
                loss = float(match_loss.group(1))
                results[current_subject][current_session][current_fold]['loss'].append(loss)

            elif match_accuracy and current_subject and current_session and current_fold:
                accuracy = float(match_accuracy.group(1))
                precision = float(match_accuracy.group(2))
                recall = float(match_accuracy.group(3))
                f1_score = float(match_accuracy.group(4))
                kappa = float(match_accuracy.group(5))

                results[current_subject][current_session][current_fold]['accuracy'].append(accuracy)
                results[current_subject][current_session][current_fold]['precision'].append(precision)
                results[current_subject][current_session][current_fold]['recall'].append(recall)
                results[current_subject][current_session][current_fold]['f1_score'].append(f1_score)
                results[current_subject][current_session][current_fold]['kappa'].append(kappa)

    return results

def extract_mean_max_acc_from_info(training_configs, results):
    simpl_results = {}
    for subject in results.keys():
        simpl_results[subject] = {}
        for session in results[subject].keys():
            simpl_results[subject][session] = {}
            for fold in results[subject][session].keys():
                cur_accuracy = results[subject][session][fold]['accuracy']
                cur_precision = results[subject][session][fold]['precision']
                cur_recall = results[subject][session][fold]['recall']
                cur_f1_score = results[subject][session][fold]['f1_score']
                cur_kappa = results[subject][session][fold]['kappa']

                if not cur_accuracy:
                    continue

                last_num = -training_configs['train']['results']['final_epoch_nums']

                cur_acc_last = cur_accuracy[last_num:]
                cur_precision_last = cur_precision[last_num:]
                cur_recall_last = cur_recall[last_num:]
                cur_f1_score_last = cur_f1_score[last_num:]
                cur_kappa_last = cur_kappa[last_num:]

                mean_accuracy = sum(cur_acc_last) / len(cur_acc_last)
                mean_precision = sum(cur_precision_last) / len(cur_precision_last)
                mean_recall = sum(cur_recall_last) / len(cur_recall_last)
                mean_f1_score = sum(cur_f1_score_last) / len(cur_f1_score_last)
                mean_kappa = sum(cur_kappa_last) / len(cur_kappa_last)

                max_accuracy = max(cur_accuracy)
                max_acc_index = cur_accuracy.index(max_accuracy)

                simpl_results[subject][session][fold] = {
                    'mean_accuracy': mean_accuracy,
                    'max_accuracy': max_accuracy,
                    'mean_precision': mean_precision,
                    'acc_precision': cur_precision[max_acc_index],
                    'mean_recall': mean_recall,
                    'acc_recall': cur_recall[max_acc_index],
                    'mean_f1_score': mean_f1_score,
                    'acc_f1_score': cur_f1_score[max_acc_index],
                    'mean_kappa': mean_kappa,
                    'acc_kappa': cur_kappa[max_acc_index],
                }

    return simpl_results

'''
#############################################################################
# excel文件操作
#############################################################################
'''
def save_loss_and_acc_to_excel(training_configs, results):
    log_path = training_configs['train']['logs']['log_path']

    rows = []
    for subject in results.keys():
        for session in results[subject].keys():
            for fold in results[subject][session].keys():
                fold_data = results[subject][session][fold]
                num_epochs = min(len(fold_data['loss']), len(fold_data['accuracy']),
                                 len(fold_data['precision']), len(fold_data['recall']),
                                 len(fold_data['f1_score']), len(fold_data['kappa']))
                for i in range(num_epochs):
                    loss = fold_data['loss'][i]
                    accuracy = fold_data['accuracy'][i]
                    precision = fold_data['precision'][i]
                    recall = fold_data['recall'][i]
                    f1_score = fold_data['f1_score'][i]
                    kappa = fold_data['kappa'][i]

                    rows.append({
                        'Subject': subject,
                        'Session': session,
                        'fold': fold,
                        'Epoch': i + 1,
                        'Loss': loss,
                        'Acc': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1 Score': f1_score,
                        'Kappa': kappa
                    })

    df = pd.DataFrame(rows)

    log_dir = os.path.dirname(log_path)
    excel_file_path = os.path.join(log_dir, training_configs['train']['logs']['result_name'])

    df.to_excel(excel_file_path, index=False)

def save_fold_mean_max_acc_to_excel(training_configs, results):
    data = []

    for subject, sessions in results.items():
        for session, folds in sessions.items():
            for fold, metrics in folds.items():
                if not isinstance(metrics.get('mean_accuracy'), (int, float)):
                    continue
                row = {
                    'subject': int(subject),
                    'session': int(session),
                    'fold': int(fold),
                    'mean_accuracy': metrics['mean_accuracy'],
                    'mean_precision': metrics['mean_precision'],
                    'mean_recall': metrics['mean_recall'],
                    'mean_f1_score': metrics['mean_f1_score'],
                    'mean_kappa': metrics['mean_kappa'],
                    'max_accuracy': metrics['max_accuracy'],
                    'acc_precision': metrics['acc_precision'],
                    'acc_recall': metrics['acc_recall'],
                    'acc_f1_score': metrics['acc_f1_score'],
                    'acc_kappa': metrics['acc_kappa']
                }
                data.append(row)

    df = pd.DataFrame(data)

    log_dir = os.path.dirname(training_configs['train']['logs']['log_path'] )
    last_folder = os.path.basename(os.path.normpath(log_dir))
    save_file_name = '01_' + last_folder + '_' + training_configs['train']['logs']['k_fold_acc_result_name']
    excel_file_path = os.path.join(log_dir, save_file_name)
    df.to_excel(excel_file_path)

    return df

def save_session_mean_max_acc_to_excel(training_configs, df):
    if 'fold' in df.columns:
        df = df.drop(columns=['fold'])
    else:
        print(f"[警告] save_session_mean_max_acc_to_excel: DataFrame 中不存在 'fold' 列")
        print(f"[警告] 可用列: {list(df.columns)}")

    mean_df = df.groupby(['subject', 'session'], as_index=False).mean()

    log_dir = os.path.dirname(training_configs['train']['logs']['log_path'])
    last_folder = os.path.basename(os.path.normpath(log_dir))
    save_file_name = '02_' + last_folder + '_'+ training_configs['train']['logs']['session_acc_result_name']
    excel_file_path = os.path.join(log_dir, save_file_name)
    mean_df.to_excel(excel_file_path)

    return mean_df

def save_subject_mean_max_acc_to_excel(training_configs, df):
    if 'session' in df.columns:
        df = df.drop(columns=['session'])
    else:
        print(f"[警告] save_subject_mean_max_acc_to_excel: DataFrame 中不存在 'session' 列")
        print(f"[警告] 可用列: {list(df.columns)}")

    mean_df = df.groupby('subject', as_index=False).mean()

    log_dir = os.path.dirname(training_configs['train']['logs']['log_path'])
    last_folder = os.path.basename(os.path.normpath(log_dir))
    save_file_name = '03_' + last_folder + '_' + training_configs['train']['logs']['subject_acc_result_name']
    excel_file_path = os.path.join(log_dir, save_file_name)
    mean_df.to_excel(excel_file_path, index=False)

    return mean_df

def save_results(training_configs):
    results = extract_info_from_log(training_configs)
    save_loss_and_acc_to_excel(training_configs, results)
    mid_results = extract_mean_max_acc_from_info(training_configs, results)
    mid_results_fold_df = save_fold_mean_max_acc_to_excel(training_configs, mid_results)

    if mid_results_fold_df is None or mid_results_fold_df.empty:
        print("[警告] fold级别DataFrame为空，跳过session和subject级别聚合")
        return

    mid_results_session_df = save_session_mean_max_acc_to_excel(training_configs, mid_results_fold_df)

    if mid_results_session_df is None or mid_results_session_df.empty:
        print("[警告] session级别DataFrame为空，跳过subject级别聚合")
        return

    save_subject_mean_max_acc_to_excel(training_configs, mid_results_session_df)


def read_excel(file_path):
    try:
        df = pd.read_excel(file_path, 'Sheet1')
        return df
    except Exception as e:
        print(f"读取 Excel 文件时出错: {e}")
        return None


def matching_files(base_path, search_pattern,
                   start_date='2024062401', end_date='2024062610',
                   datasets=['bcicIV2a'], modes=['Cl_FS'], models=['ShallowConvNet'],
                   file_type=".xlsx"):
    excel_paths = []

    start_date = int(start_date)
    end_date = int(end_date)

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.startswith('~$'):
                continue

            if search_pattern in file and file.endswith(file_type):
                path_parts = root.split(os.sep)
                folder_name = path_parts[-1]

                if not folder_name.endswith("finish"):
                    continue

                if any(mode in folder_name for mode in modes) and \
                        any(model in folder_name.split('_') for model in models) and \
                        any(dataset in folder_name for dataset in datasets):

                    part_date = int(folder_name[:8] + folder_name[9:11])
                    if start_date <= part_date <= end_date:
                        excel_paths.append(os.path.join(root, file))

    return excel_paths


def is_list(para):
    if isinstance(para, str):
        para = [para]
    return para


def clean_dataframe(df, target_value):
    if df.columns[0] != target_value:
        df = df.dropna(how='all')

        df = df.drop(df.columns[0], axis=1)

        df = df.iloc[1:].reset_index(drop=True)

        df.columns = ['subject', 'mean_acc', 'max_acc']

        mean_row = df[['mean_acc', 'max_acc']].mean().to_frame().T

        df = pd.concat([df, mean_row], ignore_index=True)

        df.loc[len(df) - 1, 'subject'] = 'mean'

    return df


def extract_classification_results_from_excels(find_path="logs",
                                               start_date='2022090101',
                                               end_date='2026063099',
                                               datasets=['bcicIV2a', 'OpenBMI'],
                                               modes=['Cl_FS'],
                                               models=['SKLightNet']):
    from src.main.train.utils.configs import get_project_rootpath
    find_path = os.path.join(get_project_rootpath(), find_path)

    matching_excel_paths = matching_files(find_path, 'subject_mean_max_acc_of_', start_date, end_date, datasets, modes,
                                          models)

    results = {dataset: {mode: {model: {} for model in models} for mode in modes} for dataset in datasets}

    for file_path in matching_excel_paths:
        file_name = os.path.basename(file_path)
        parts = file_name.split("_")
        dataset_name = parts[-4]
        model_name = parts[-3]
        mode_name = '_'.join(file_name.split(".")[0].split("_")[10:])

        if dataset_name in datasets and mode_name in modes and model_name in models:
            print(f"当前进行是{dataset_name}数据集中{mode_name}类型中的{model_name}文件：{file_name}")

            cur_data_df = pd.read_excel(file_path)
            cur_data_df = clean_dataframe(cur_data_df, target_value='subject')

            cur_data = cur_data_df.loc[0:len(cur_data_df) - 1,
                       ['max_accuracy', 'acc_precision', 'acc_recall', 'acc_f1_score', 'acc_kappa']].astype(
                float)

            if not results[dataset_name][mode_name][model_name]:
                results[dataset_name][mode_name][model_name] = {'max_accuracy': cur_data['max_accuracy'].tolist(),
                                                               'acc_precision': cur_data['acc_precision'].tolist(),
                                                               'acc_recall': cur_data['acc_recall'].tolist(),
                                                               'acc_f1_score': cur_data['acc_f1_score'].tolist(),
                                                               'acc_kappa': cur_data['acc_kappa'].tolist()}
            else:
                pre_data = results[dataset_name][mode_name][model_name]['max_accuracy']

                if model_name in ["MSDCNN0MSDCNN",
                                  "MSDCNN0D2All63",
                                  'MSDCNN0D2Begin31',
                                  'MSDCNN0D2Begin127',
                                  "MSDCNN0D3CK3Tmp30",
                                  "MSDCNN0D3CK5Tmp30",
                                  "MSDCNN0D14t12t20t32t40t100",
                                  "MSDCNN0D42PS250",
                                  "MSACNN0MSACNN",
                                  "LightMIEEGNet0LightMIEEGNet", "ResDSNet0ResDSNet"]:
                    cur_data['max_accuracy'] = np.maximum(pre_data, cur_data['max_accuracy'])
                    replaced_positions = cur_data['max_accuracy'] > pre_data
                else:
                    cur_data['max_accuracy'] = np.minimum(pre_data, cur_data['max_accuracy'])
                    replaced_positions = cur_data['max_accuracy'] < pre_data

                cur_data['acc_precision'] = np.where(replaced_positions, cur_data['acc_precision'],
                                                     results[dataset_name][mode_name][model_name]['acc_precision'])
                cur_data['acc_recall'] = np.where(replaced_positions, cur_data['acc_recall'],
                                                  results[dataset_name][mode_name][model_name]['acc_recall'])
                cur_data['acc_f1_score'] = np.where(replaced_positions, cur_data['acc_f1_score'],
                                                    results[dataset_name][mode_name][model_name]['acc_f1_score'])
                cur_data['acc_kappa'] = np.where(replaced_positions, cur_data['acc_kappa'],
                                                 results[dataset_name][mode_name][model_name]['acc_kappa'])

                results[dataset_name][mode_name][model_name] = {
                    'max_accuracy': cur_data['max_accuracy'].tolist(),
                    'acc_precision': cur_data['acc_precision'].tolist(),
                    'acc_recall': cur_data['acc_recall'].tolist(),
                    'acc_f1_score': cur_data['acc_f1_score'].tolist(),
                    'acc_kappa': cur_data['acc_kappa'].tolist()
                }

    return results


def remove_empty_dicts(d):
    if isinstance(d, dict):
        return {k: remove_empty_dicts(v) for k, v in d.items() if v != {}}
    return d


def is_dict_empty(d):
    if isinstance(d, dict):
        return all(is_dict_empty(v) for v in d.values())
    return False


def append_last_row(df_list1, df_list2):
    if isinstance(df_list1, pd.DataFrame):
        df_list1 = [df_list1]
    if isinstance(df_list2, pd.DataFrame):
        df_list2 = [df_list2]

    df1 = pd.concat(df_list1, ignore_index=True)

    last_row = df_list2[-1].iloc[[-1]]

    df1 = pd.concat([df1, last_row], ignore_index=True)

    return df1


def calc_t_test(row, last_row_subjects, subject_cols):
    other_subjects = pd.to_numeric(row[subject_cols], errors='coerce').values
    last_row_subjects = pd.to_numeric(last_row_subjects, errors='coerce')

    mask = ~pd.isna(other_subjects) & ~pd.isna(last_row_subjects)
    other_subjects = other_subjects[mask]
    last_row_subjects = last_row_subjects[mask]

    if len(other_subjects) == 0 or len(last_row_subjects) == 0:
        return float('nan')

    t_stat, p_value = ttest_rel(other_subjects, last_row_subjects)
    return p_value


def significance_stars(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return str(p_value)

def t_test_with_last_row(df, subject_prefix='Subject_'):
    subject_cols = [col for col in df.columns if col.startswith(subject_prefix)]

    last_row_subjects = pd.to_numeric(df.iloc[-1][subject_cols], errors='coerce').values

    df['p_value'] = df.apply(lambda row: calc_t_test(row, last_row_subjects, subject_cols), axis=1)

    df['sign'] = df['p_value'].apply(significance_stars)

    return df


def calc_z_test(row, last_row_subjects, subject_cols):
    other_subjects = pd.to_numeric(row[subject_cols], errors='coerce').values
    last_row_subjects = pd.to_numeric(last_row_subjects, errors='coerce')

    mask = ~pd.isna(other_subjects) & ~pd.isna(last_row_subjects)
    other_subjects = other_subjects[mask]
    last_row_subjects = last_row_subjects[mask]

    if len(other_subjects) == 0 or len(last_row_subjects) == 0:
        return float('nan')

    z_stat, p_value = ztest(other_subjects, last_row_subjects)

    return p_value


def z_test_with_last_row(df, subject_prefix='Subject_'):
    subject_cols = [col for col in df.columns if col.startswith(subject_prefix)]

    last_row_subjects = pd.to_numeric(df.iloc[-1][subject_cols], errors='coerce').values

    df['p_value'] = df.apply(lambda row: calc_z_test(row, last_row_subjects, subject_cols), axis=1)

    df['sign'] = df['p_value'].apply(significance_stars)

    return df


def extract_acc_table(data, dataset, class_method, metric):
    if not data:
        raise ValueError("字典完全为空")

    networks = data.get(dataset, {}).get(class_method, {})
    if not networks:
        raise ValueError(f"数据集 '{dataset}' 或分类方法 '{class_method}' 不存在或为空")

    non_empty_networks = {k: v for k, v in networks.items() if v}

    if not non_empty_networks:
        raise ValueError(f"数据集 '{dataset}' 或分类方法 '{class_method}' 下没有有效的网络数据")

    num_subjects = len(next(iter(non_empty_networks.values()))[metric])

    subjects_header = [f'Subject_{i + 1}' for i in range(num_subjects)]

    table_data = []
    for network_name, network_data in non_empty_networks.items():
        metric_values = network_data[metric]
        mean_metric = np.mean(metric_values)
        std_metric = np.std(metric_values)
        row = [network_name] + metric_values + [mean_metric, std_metric]
        table_data.append(row)

    df = pd.DataFrame(table_data, columns=['Network'] + subjects_header + ['mean', 'std'])

    return df


def save_dfs_to_excel(dfs, file_path='', file_name='results.xlsx', sheet_names=None):
    if not file_path:
        file_path = os.getcwd()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified path {file_path} does not exist.")

    full_file_name = os.path.join(file_path, file_name)

    if sheet_names is None:
        sheet_names = [f'Sheet{i + 1}' for i in range(len(dfs))]

    if len(sheet_names) < len(dfs):
        sheet_names += [f'Sheet{i + 1}' for i in range(len(sheet_names), len(dfs))]

    with pd.ExcelWriter(full_file_name, engine='openpyxl') as writer:
        for i, df in enumerate(dfs):
            sheet_name = sheet_names[i]
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"DataFrames saved successfully to {full_file_name}")

def create_target_directory_structure(target_dir, datasets, models):
    for dataset in datasets:
        dataset_dir = os.path.join(target_dir, dataset)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        for model in models:
            model_dir = os.path.join(dataset_dir, model)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

def get_folders_to_move(source_dir, pattern, models, datasets):
    folders_to_move = []
    for root, dirs, files in os.walk(source_dir):
        for folder_name in dirs:
            folder_path = os.path.join(root, folder_name)

            if os.path.isdir(folder_path):
                match = re.match(pattern, folder_name)
                if match:
                    model = match.group(1)
                    dataset = match.group(2)

                    if model in models and dataset in datasets:
                        folders_to_move.append((folder_path, folder_name, dataset, model))
    return folders_to_move

def move_folders(folders_to_move, target_dir):
    for folder_path, folder_name, dataset, model in tqdm(folders_to_move, desc="移动文件夹", unit="folder"):
        target_path = os.path.join(target_dir, dataset, model, folder_name)
        target_folder = os.path.join(target_dir, dataset, model)
        tqdm.write(f"移动文件夹 {folder_name} 到 {target_folder}")
        try:
            shutil.move(folder_path, target_path)
        except Exception as e:
            tqdm.write(f"移动 {folder_name} 时出错: {str(e)}")

def organize_folders(source_dir, target_dir, datasets, models):
    create_target_directory_structure(target_dir, datasets, models)

    pattern = r'\d{8}_\d+_([^_]+)_([^_]+)_[^_]+'

    folders_to_move = get_folders_to_move(source_dir, pattern, models, datasets)

    move_folders(folders_to_move, target_dir)

    print("文件整理完成！")


def print_model_summary(model, input):
    device = next(model.parameters()).device
    input = input.to(device)

    summary_stats = summary(
        model,
        input_data=input,
        verbose=1,
    )


    flops = FlopCountAnalysis(model, input,)
    print(f"FLOPs: {flops.total()} / {flops.total() / 1e6:.2f} M")
    return summary_stats


if __name__ == '__main__':
    results = extract_classification_results_from_excels(
        find_path=r"G:\Results\P2\00Classification",
        start_date="2024092416",
        end_date="2026100199",
        datasets=['bcicIV2a', 'OpenBMI'],
        modes=['Cl_FS'],
        models=['ShallowConvNet', 'DeepConvNet', 'EEGNet', 'FBCNet', 'EEGConformer', 'FBMSNet', 'IFNet', 'TSFCNet', 'EEGCDILNet', 'ADFCNN', 'EEGProgress', 'LightMIEEGNet0LightMIEEGNet']
    )
    df = extract_acc_table(results, 'bcicIV2a', 'Cl_FS', 'max_accuracy')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(df)

