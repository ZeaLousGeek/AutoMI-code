import sys
from pathlib import Path
from typing import TypedDict
from src.main.agents.planning_agent import PlanningAgent
from src.main.agents.execution_agent import ExecutionAgent
from src.main.agents.output_agent import OutputAgent
from src.main.tools.experience_tracker import ExperienceTracker

PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_ROOT = PROJECT_ROOT / 'src'
RL_CHECKPOINTS_DIR = PROJECT_ROOT / 'checkpoints'


def _rl_save_path(dataset_name: str | None = None,
                  subject_id: int | None = None) -> Path:
    if dataset_name is not None and subject_id is not None:
        return RL_CHECKPOINTS_DIR / f'rl_{dataset_name}_sub{subject_id}.json'
    return RL_CHECKPOINTS_DIR / 'rl_agent.json'


class TeeWriter:
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log_file.write(message)
            self.log_file.flush()
        except (ValueError, OSError):
            pass

    def flush(self):
        self.terminal.flush()
        try:
            self.log_file.flush()
        except (ValueError, OSError):
            pass


def _read_model_source(model_name: str) -> str | None:
    candidates = [
        SRC_ROOT / 'models' / model_name / f'{model_name}.py',
        SRC_ROOT / 'models' / f'{model_name}.py',
    ]
    for path in candidates:
        if path.exists():
            return path.read_text(encoding='utf-8')
    return None


class AgentState(TypedDict):
    iteration: int
    max_iterations: int
    planning_agent: PlanningAgent
    execution_agent: ExecutionAgent
    output_agent: OutputAgent
    current_plan: dict
    execution_result: dict
    test_results: dict
    current_accuracy: float
    old_accuracy: float
    best_accuracy: float
    best_results: dict
    iteration_history: list
    done: bool
    selected_model_name: str
    selected_model_path: Path
    test_mode: bool
    selected_datasets: list
    current_model_code: str
    initial_model_code: str
    best_model_code: str
    model_code_before_planning: str
    model_code_before_structure_update: str
    in_structure_tuning_phase: bool
    structure_tuning_remaining: int
    force_structure_update: bool
    iteration_log_parts: list
    experience_tracker: ExperienceTracker
    last_execution_error: str
    tee_log_file: object
    dataset_name: str
    subject_id: int
    rl_save_path: str
    ablation_mode: str


def initialize_state(max_iterations: int = 10, selected_model_name=None,
                     selected_model_path=None, test_mode=False,
                     selected_datasets=None,
                     max_consecutive_param_failures=3,
                     dataset_name: str | None = None,
                     subject_id: int | None = None,
                     base_output_dir: str | None = None,
                     ablation_mode: str | None = None) -> AgentState:
    RL_CHECKPOINTS_DIR.mkdir(exist_ok=True)

    per_subject = dataset_name is not None and subject_id is not None

    if selected_datasets is None:
        selected_datasets = [dataset_name] if dataset_name else ['bcicIV2a']

    planning_agent = PlanningAgent(
        selected_datasets=selected_datasets,
        max_consecutive_param_failures=max_consecutive_param_failures,
        dataset_name=dataset_name,
        subject_id=subject_id,
        ablation_mode=ablation_mode,
    )

    execution_agent = ExecutionAgent(
        test_mode=test_mode,
        selected_datasets=selected_datasets,
        dataset_name=dataset_name,
        subject_id=subject_id,
    )

    if per_subject and base_output_dir:
        output_agent = OutputAgent(
            selected_model_name, selected_model_path,
            selected_datasets=selected_datasets,
            base_output_dir=base_output_dir,
            dataset_name=dataset_name,
            subject_id=subject_id,
        )
    else:
        output_agent = OutputAgent(
            selected_model_name, selected_model_path,
            selected_datasets=selected_datasets,
        )

    tag = f"[{dataset_name}/Sub{subject_id}] " if per_subject else ""

    if selected_model_name:
        planning_agent.current_model = selected_model_name
        print(f"{tag}使用用户选择的模型: {selected_model_name}")
    print(f"{tag}使用数据集: {', '.join(selected_datasets)}")
    if per_subject:
        print(f"{tag}受试者: {subject_id}")

    rl_path = _rl_save_path(dataset_name, subject_id)
    if rl_path.exists():
        planning_agent.load_rl_state(str(rl_path))

    initial_model_code = _read_model_source(
        selected_model_name or 'EEGNet'
    )

    from src.main.tools.summary_tool import init_summary_dir
    summary_dir = init_summary_dir(output_agent.model_output_dir)

    log_fh = open(summary_dir / 'process_log.txt', 'w', encoding='utf-8')
    sys.stdout = TeeWriter(sys.__stdout__, log_fh)

    return {
        'iteration': 0,
        'max_iterations': max_iterations,
        'planning_agent': planning_agent,
        'execution_agent': execution_agent,
        'output_agent': output_agent,
        'current_plan': {},
        'execution_result': {},
        'test_results': {},
        'current_accuracy': 0.0,
        'old_accuracy': 0.0,
        'best_accuracy': 0.0,
        'best_results': {},
        'iteration_history': [],
        'done': False,
        'selected_model_name': selected_model_name or 'EEGNet',
        'selected_model_path': selected_model_path,
        'test_mode': test_mode,
        'selected_datasets': selected_datasets,
        'current_model_code': initial_model_code,
        'initial_model_code': initial_model_code,
        'best_model_code': initial_model_code,
        'model_code_before_planning': initial_model_code,
        'model_code_before_structure_update': None,
        'in_structure_tuning_phase': False,
        'structure_tuning_remaining': 0,
        'force_structure_update': False,
        'iteration_log_parts': [],
        'experience_tracker': ExperienceTracker(),
        'last_execution_error': None,
        'tee_log_file': log_fh,
        'dataset_name': dataset_name or '',
        'subject_id': subject_id or 0,
        'rl_save_path': str(rl_path),
        'ablation_mode': ablation_mode,
    }
