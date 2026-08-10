
import os
import sys
import traceback
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent


def run_subject_workflow(
    model_name: str,
    model_path,
    dataset_name: str,
    subject_id: int,
    base_output_dir: str,
    max_iterations: int = 10,
    max_consecutive_param_failures: int = 3,
    llm_model: str = "qwen3-max",
    gpu_id: str = "0",
    ablation_mode: str = None,
):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    src_root = PROJECT_ROOT / 'src'
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from src.main.utils import config
    config.DEFAULT_MODEL = llm_model

    tag = f"[{dataset_name}/Sub{subject_id}]"
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{tag} [{ts}] 子进程启动，模型={model_name}，LLM={llm_model}，GPU={gpu_id}")

    try:
        from src.main.workflow.graph import create_workflow
        from src.main.workflow.state import initialize_state

        state = initialize_state(
            max_iterations=max_iterations,
            selected_model_name=model_name,
            selected_model_path=model_path,
            dataset_name=dataset_name,
            subject_id=subject_id,
            base_output_dir=base_output_dir,
            max_consecutive_param_failures=max_consecutive_param_failures,
            ablation_mode=ablation_mode,
        )

        app = create_workflow()
        result = app.invoke(state)

        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(
            f"{tag} [{ts}] 子进程完成，"
            f"最佳准确率={result['best_accuracy'] * 100:.2f}%"
        )

        return {
            'dataset': dataset_name,
            'subject_id': subject_id,
            'best_accuracy': result['best_accuracy'],
            'iteration_history': [
                {
                    'iteration': e.get('iteration'),
                    'accuracy': e.get('accuracy', 0.0),
                    'improved': e.get('improved', False),
                    'action': e.get('plan', {}).get('action', ''),
                }
                for e in result.get('iteration_history', [])
            ],
            'success': True,
            'error': None,
        }

    except Exception:
        tb = traceback.format_exc()
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{tag} [{ts}] 子进程异常:\n{tb}")
        return {
            'dataset': dataset_name,
            'subject_id': subject_id,
            'best_accuracy': 0.0,
            'iteration_history': [],
            'success': False,
            'error': tb,
        }
