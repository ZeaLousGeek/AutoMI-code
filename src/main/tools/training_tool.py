
import os
import sys
import logging
import time
import warnings
import importlib
import importlib.util
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_ROOT = PROJECT_ROOT / 'src'
CONFIGS_DIR = SRC_ROOT / 'main' / 'configs'


def _build_training_config_paths(dataset_name, model_name, mode='Cl_FS'):
    from src.main.train.utils.files import replace_first_zero
    from src.main.train.utils.configs import get_project_rootpath, check_and_create_yaml

    root_path = get_project_rootpath()

    dataset_config = os.path.join(root_path, f"src/main/configs/datasets/{dataset_name}.yaml")
    model_config = os.path.join(root_path, f"src/main/configs/models/{replace_first_zero(model_name)}.yaml")
    if not Path(model_config).exists():
        model_config = os.path.join(root_path, f"src/main/configs/models/{model_name}/{model_name}.yaml")
    train_config = os.path.join(root_path, "src/main/configs/identity_config.yaml")
    file_config = os.path.join(
        root_path,
        f"src/main/configs/files/{dataset_name}/{dataset_name}_{model_name}_{mode}.yaml"
    )
    mode_config = os.path.join(root_path, f"src/main/configs/modes/{mode}.yaml")

    check_and_create_yaml(file_config)

    return [dataset_config, model_config, train_config, file_config, mode_config]


def _get_model_module_key(model_name):
    subdir = SRC_ROOT / 'models' / model_name
    if subdir.is_dir() and (subdir / f'{model_name}.py').exists():
        return f'models.{model_name}.{model_name}'
    return f'models.{model_name}'


def _inject_model_code(model_name, model_code, model_save_dir):
    save_dir = Path(model_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_file = save_dir / 'model_improved.py'
    model_file.write_text(model_code, encoding='utf-8')

    module_key = _get_model_module_key(model_name)

    modules_to_remove = [
        key for key in list(sys.modules.keys())
        if 'models' in key and model_name in key
    ]
    for key in modules_to_remove:
        del sys.modules[key]

    spec = importlib.util.spec_from_file_location(module_key, str(model_file))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_key] = module
    spec.loader.exec_module(module)

    parent_parts = module_key.split('.')
    if len(parent_parts) > 1:
        parent_key = '.'.join(parent_parts[:-1])
        if parent_key not in sys.modules:
            parent_spec = importlib.util.spec_from_file_location(parent_key, str(model_file))
            parent_module = importlib.util.module_from_spec(parent_spec)
            sys.modules[parent_key] = parent_module
        setattr(sys.modules[parent_key], parent_parts[-1], module)

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(
        f"[{ts}] 已从 output 动态加载改进模型: {model_file}\n"
        f"[{ts}] 注入 sys.modules['{module_key}']"
    )
    return model_file


def _run_single_dataset_training(dataset_name, model_name, mode='Cl_FS',
                                 config_overrides=None, test_mode=False,
                                 subject_id=None, log_save_dir=None):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    tag = f"[{dataset_name}/Sub{subject_id}] " if subject_id else ""
    print(f"[{ts}] {tag}========================================")
    print(f"[{ts}] {tag}训练流程开始: {dataset_name} / {model_name} / {mode}")

    config_paths = _build_training_config_paths(dataset_name, model_name, mode)

    missing = [p for p in config_paths if not Path(p).exists()]
    if missing:
        for p in missing:
            print(f"[{ts}] {tag}配置文件缺失: {p}")
        return None, None

    print(f"[{ts}] {tag}配置文件全部验证通过，启动训练...")

    from src.main.train.utils.configs import load_configs, add_suffix_to_folder
    from src.main.train.utils.results import (
        save_results, extract_info_from_log, extract_mean_max_acc_from_info,
    )
    from src.main.train.utils.files import wait_until_folder_available
    from src.main.train.temp.Cl_FS import train_single_subject_Cl_FS

    training_configs, logger = load_configs(config_paths, subject_id=subject_id,
                                               log_save_dir=log_save_dir,
                                               copy_model=False)

    warnings.filterwarnings("ignore", message="Channel names are not unique")

    if config_overrides:
        _apply_config_overrides(training_configs, config_overrides)

    if subject_id is not None:
        subjects_to_train = [subject_id]
        total_subjects = 1
    else:
        total_subjects = training_configs['dataset']['subjects']
        if test_mode:
            total_subjects = 1
        chosen_subs = training_configs.get('dataset', {}).get('choosen_subs')
        subjects_to_train = [
            sid for sid in range(1, total_subjects + 1)
            if not chosen_subs or sid in chosen_subs
        ]

    for sid in subjects_to_train:
        training_configs['subject_id'] = sid

        logger.info("+" * 78)
        logger.info(f"The {sid}th subject")
        logger.info("=" * 76)
        logger.info(f"The 0th session of the {sid}th subject")
        logger.info("=" * 76)
        logger.info(
            f"The 0-fold cross-validation, and the 0th session "
            f"of the {sid}th subject"
        )
        logger.info("=" * 76)

        ts_sub = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts_sub}] {tag}训练受试者 {sid}/{total_subjects}")

        start = time.perf_counter()
        train_single_subject_Cl_FS(training_configs, logger)
        elapsed = time.perf_counter() - start

        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"{tag}受试者 {sid} 完成，耗时 {elapsed:.2f}秒"
        )

    save_results(training_configs)

    results = extract_info_from_log(training_configs)
    simpl_results = extract_mean_max_acc_from_info(training_configs, results)

    total_acc = 0.0
    count = 0
    for subject, sessions in simpl_results.items():
        for session, folds in sessions.items():
            for fold, metrics in folds.items():
                max_acc = metrics.get('max_accuracy')
                if isinstance(max_acc, (int, float)):
                    total_acc += max_acc
                    count += 1

    mean_accuracy = total_acc / count if count > 0 else 0.0

    logger.info("+" * 78)
    logger.info("The training is finished")
    logger.info("+" * 78)
    logging.shutdown()

    if not log_save_dir:
        wait_until_folder_available(training_configs['train']['logs']['save_path'])
        suffix = (
            '_' + training_configs['model']['name']
            + '_' + training_configs['dataset']['name']
            + '_' + training_configs['train']['mode']
            + '_finish'
        )
        add_suffix_to_folder(training_configs['train']['logs']['save_path'], suffix)

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {tag}训练完成，平均最大准确率: {mean_accuracy * 100:.2f}%")
    print(f"[{ts}] {tag}========================================")
    return mean_accuracy, simpl_results


def _apply_config_overrides(training_configs, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and key in training_configs:
            if isinstance(training_configs[key], dict):
                _apply_config_overrides(training_configs[key], value)
            else:
                training_configs[key] = value
        else:
            training_configs[key] = value


def run_single_subject_training(dataset_name, model_name, subject_id,
                                model_code=None, model_save_dir=None,
                                config_overrides=None):
    if model_code is not None:
        if model_save_dir is None:
            raise ValueError(
                "当 model_code 不为空时，必须提供 model_save_dir"
            )
        _inject_model_code(model_name, model_code, model_save_dir)

    return _run_single_dataset_training(
        dataset_name, model_name,
        config_overrides=config_overrides,
        subject_id=subject_id,
        log_save_dir=model_save_dir,
    )


def run_model_training(dataset_name, model_name, model_code=None,
                       model_save_dir=None, config_overrides=None,
                       test_mode=False):
    if model_code is not None:
        if model_save_dir is None:
            raise ValueError(
                "当 model_code 不为空时，必须提供 model_save_dir "
                "（output 迭代文件夹路径）"
            )
        _inject_model_code(model_name, model_code, model_save_dir)

    return _run_single_dataset_training(
        dataset_name, model_name,
        config_overrides=config_overrides,
        test_mode=test_mode,
        log_save_dir=model_save_dir,
    )


def run_model_training_all_datasets(model_name, datasets=None, model_code=None,
                                    model_save_dir=None, config_overrides=None,
                                    test_mode=False):
    if datasets is None:
        datasets = ['bcicIV2a', 'OpenBMI']

    if model_code is not None:
        if model_save_dir is None:
            raise ValueError(
                "当 model_code 不为空时，必须提供 model_save_dir "
                "（output 迭代文件夹路径）"
            )
        _inject_model_code(model_name, model_code, model_save_dir)

    results = {}
    detailed_results = {}
    for dataset in datasets:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] 在数据集 {dataset} 上训练模型 {model_name}")

        accuracy, detail = _run_single_dataset_training(
            dataset, model_name,
            config_overrides=config_overrides,
            test_mode=test_mode,
            log_save_dir=model_save_dir,
        )
        if accuracy is not None:
            results[dataset] = accuracy
            detailed_results[dataset] = detail
        else:
            print(f"[{ts}] 数据集 {dataset} 训练失败")
            results[dataset] = 0.0

    return results, detailed_results
