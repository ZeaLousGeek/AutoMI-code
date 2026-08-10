
import argparse
import sys
import importlib.util
import warnings
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*deprecated.*")


PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'main'))


def check_dependencies():
    required_packages = [
        'torch', 'torchvision', 'numpy', 'scipy', 'yaml',
        'langchain', 'langgraph', 'langchain_openai',
        'mne', 'h5py', 'tqdm', 'dotenv'
    ]

    missing_packages = []

    for package in required_packages:
        if package == 'yaml':
            import_found = importlib.util.find_spec('yaml') is not None
        elif package == 'dotenv':
            import_found = importlib.util.find_spec('dotenv') is not None
        else:
            import_found = importlib.util.find_spec(package) is not None

        if not import_found:
            missing_packages.append(package)

    if missing_packages:
        print("错误: 缺少必需的依赖包:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\n请运行以下命令安装所有依赖:")
        print("  pip install -r requirements.txt")
        sys.exit(1)


def discover_models():
    models_dir = PROJECT_ROOT / 'src' / 'main' / 'models'
    names = []
    for item in sorted(models_dir.iterdir()):
        if item.is_dir():
            if (item / f"{item.name}.py").exists():
                names.append(item.name)
        elif item.is_file() and item.suffix == '.py' and item.stem != '__init__':
            names.append(item.stem)
    return names


def resolve_model_path(model_name):
    models_dir = PROJECT_ROOT / 'src' / 'main' / 'models'
    candidates = [
        models_dir / model_name / f'{model_name}.py',
        models_dir / f'{model_name}.py',
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到模型文件: {model_name}")


def _get_subject_count(dataset_name):
    config_path = PROJECT_ROOT / 'src' / 'main' / 'configs' / 'datasets' / f'{dataset_name}.yaml'
    if not config_path.exists():
        return 9
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg.get('dataset', {}).get('subjects', 9)


def main():
    from src.main.utils.config import AVAILABLE_MODELS, DEFAULT_MODEL

    available_eeg_models = discover_models()

    parser = argparse.ArgumentParser(
        description='AutoMI v1.0.0 - 运动想象脑电信号分类模型自动迭代系统'
    )

    parser.add_argument(
        '--model', '-m',
        nargs='+',
        choices=available_eeg_models,
        default=['EEGNet'],
        help='选择 EEG 模型 (默认: EEGNet)，可多选'
    )

    parser.add_argument(
        '--llm',
        nargs='+',
        choices=AVAILABLE_MODELS,
        default=[DEFAULT_MODEL],
        help=f'选择大模型 (默认: {DEFAULT_MODEL})，可多选'
    )

    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=26,
        help='最大迭代次数 (默认: 26)'
    )

    parser.add_argument(
        '--conda-env',
        type=str,
        default='automi',
        help='Conda环境名称 (默认: automi)'
    )

    parser.add_argument(
        '--datasets', '-d',
        nargs='+',
        choices=['bcicIV2a', 'OpenBMI'],
        default=['bcicIV2a'],
        help='选择使用的数据集 (默认: bcicIV2a)，可多选，如: -d bcicIV2a OpenBMI'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='测试模式，每个数据集仅使用1个受试者'
    )

    parser.add_argument(
        '--max-param-failures',
        type=int,
        default=5,
        help='parameter_evolution连续未改进多少次后强制切换为structure_update (默认: 5)'
    )

    parser.add_argument(
        '--max-workers', '-w',
        type=int,
        default=5,
        help='最大并行受试者进程数 (默认: 5)'
    )

    parser.add_argument(
        '--gpu', '-g',
        type=str,
        default='0',
        help='指定使用的GPU编号，多GPU用逗号分隔 (默认: 0)，如: --gpu 0 或 --gpu 0,1,2'
    )

    parser.add_argument(
        '--ablation',
        type=str,
        choices=['no-structure-update', 'random-action', 'no-experience', 'no-literature'],
        default=None,
        help='消融实验模式 (可选)：no-structure-update (禁用结构更新), random-action (随机动作选择), no-experience (禁用经验追踪), no-literature (禁用文献检索)'
    )

    args = parser.parse_args()
    gpu_list = [g.strip() for g in args.gpu.split(',')]

    print("=" * 70)
    print("AutoMI v1.0.0 - 运动想象脑电信号分类模型自动迭代系统")
    print("=" * 70)

    check_dependencies()

    subject_tasks = []
    for ds in args.datasets:
        n_subjects = _get_subject_count(ds)
        if args.test:
            n_subjects = 1
        for sid in range(1, n_subjects + 1):
            subject_tasks.append((ds, sid))

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    from src.main.workflow.subject_runner import run_subject_workflow
    from src.main.tools.summary_tool import generate_aggregated_summary

    for model_name in args.model:
        model_path = resolve_model_path(model_name)

        for llm_name in args.llm:
            if args.ablation:
                output_root = OUTPUT_DIR.parent / 'output_ablation' / args.ablation
                output_root.mkdir(parents=True, exist_ok=True)
            else:
                output_root = OUTPUT_DIR

            base_output_dir = str(
                output_root / f'{model_name}_{llm_name}_{timestamp}'
            )
            Path(base_output_dir).mkdir(parents=True, exist_ok=True)

            print("=" * 70)
            print(f"配置:")
            print(f"  最大迭代次数: {args.iterations}")
            print(f"  Conda环境: {args.conda_env}")
            print(f"  EEG 模型: {model_name}")
            print(f"  大模型 (LLM): {llm_name}")
            print(f"  选择数据集: {', '.join(args.datasets)}")
            print(f"  参数优化最大连续失败次数: {args.max_param_failures}")
            print(f"  最大并行进程数: {args.max_workers}")
            print(f"  GPU: {args.gpu}")
            print(f"  总受试者任务数: {len(subject_tasks)}")
            print(f"  输出目录: {base_output_dir}")
            if args.test:
                print(f"  测试模式: 每个数据集仅1个受试者")
            if args.ablation:
                print(f"  消融实验模式: {args.ablation}")
            print("=" * 70)

            all_results = []
            with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
                future_map = {}
                for task_idx, (ds, sid) in enumerate(subject_tasks):
                    assigned_gpu = gpu_list[task_idx % len(gpu_list)]
                    future = pool.submit(
                        run_subject_workflow,
                        model_name=model_name,
                        model_path=str(model_path),
                        dataset_name=ds,
                        subject_id=sid,
                        base_output_dir=base_output_dir,
                        max_iterations=args.iterations,
                        max_consecutive_param_failures=args.max_param_failures,
                        llm_model=llm_name,
                        gpu_id=assigned_gpu,
                        ablation_mode=args.ablation,
                    )
                    future_map[future] = (ds, sid)

                for future in as_completed(future_map):
                    ds, sid = future_map[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                        status = "成功" if result['success'] else "失败"
                        acc = result['best_accuracy'] * 100
                        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print(
                            f"[{ts}] [{ds}/Sub{sid}] 完成 ({status}) "
                            f"最佳准确率={acc:.2f}%"
                        )
                    except Exception as exc:
                        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print(f"[{ts}] [{ds}/Sub{sid}] 进程异常: {exc}")
                        all_results.append({
                            'dataset': ds,
                            'subject_id': sid,
                            'best_accuracy': 0.0,
                            'iteration_history': [],
                            'success': False,
                            'error': str(exc),
                        })

            print("\n" + "=" * 70)
            print(f"[{model_name} + {llm_name}] 所有受试者流程完成，生成聚合汇总...")
            print("=" * 70)

            generate_aggregated_summary(base_output_dir, all_results)

            successful = [r for r in all_results if r['success']]
            if successful:
                avg_best = sum(r['best_accuracy'] for r in successful) / len(successful)
                print(f"\n总计 {len(successful)}/{len(all_results)} 个受试者成功")
                print(f"全局平均最佳准确率: {avg_best * 100:.2f}%")
            else:
                print("\n所有受试者均失败，请检查日志")

    print("\n系统执行完成!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
