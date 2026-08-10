import json
from pathlib import Path
from datetime import datetime
from src.main.utils.client import chat
from src.main.prompts import load_prompt
from src.main.tools.training_tool import (
    run_model_training_all_datasets,
    run_single_subject_training,
)


PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_ROOT = PROJECT_ROOT / 'src'
CONFIGS_DIR = SRC_ROOT / 'configs'

SYSTEM_PROMPT_TEMPLATE = load_prompt('execution_agent')

DATASET_DESCRIPTIONS = {
    'bcicIV2a': '9 个受试者，4 类（左手/右手/双脚/舌头），22 通道，较高难度',
    'OpenBMI': '54 个受试者，2 类（左手/右手），62 通道，受试者间差异大',
}


class ExecutionAgent:
    def __init__(self, test_mode=False, selected_datasets=None,
                 dataset_name=None, subject_id=None):
        self.current_model_name = None
        self.current_datasets = selected_datasets or ['bcicIV2a']
        self.test_mode = test_mode
        self.dataset_name = dataset_name
        self.subject_id = subject_id

        ds_lines = []
        for ds in self.current_datasets:
            desc = DATASET_DESCRIPTIONS.get(ds, '未知数据集')
            ds_lines.append(f"- **{ds}**：{desc}")
        dataset_info = "\n".join(ds_lines)
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.replace("{dataset_info}", dataset_info)

        tag = f"[{dataset_name}/Sub{subject_id}] " if dataset_name and subject_id else ""
        print(f"{tag}ExecutionAgent 初始化完成")

    @property
    def _per_subject(self):
        return self.dataset_name is not None and self.subject_id is not None

    @property
    def _tag(self):
        if self._per_subject:
            return f"[{self.dataset_name}/Sub{self.subject_id}] "
        return ""

    def parse_execution_plan(self, plan):
        action = plan.get('action')
        print(f"{self._tag}解析执行计划，动作: {action}")
        return action, plan

    def execute_plan(self, plan, model_code=None, model_save_dir=None):
        action, details = self.parse_execution_plan(plan)
        self.current_model_name = details.get('model_name', self.current_model_name)

        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] {self._tag}开始执行动作: {action}")

        config_overrides = details.get('config_overrides')

        if action == 'initial_test':
            result = self._run_initial_test(
                details, model_code, model_save_dir,
            )
        elif action in ('parameter_evolution', 'structure_update',
                        'continue_current'):
            result = self._run_with_improvements(
                details, model_code, config_overrides, model_save_dir,
            )
        else:
            result = {'success': False, 'error': f'未知动作: {action}'}

        status = "成功" if result.get('success') else "失败"
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] {self._tag}动作执行{status}: {action}")
        return result

    def _run_training(self, model_name, model_code, model_save_dir,
                      config_overrides=None):
        if self._per_subject:
            accuracy, detail = run_single_subject_training(
                dataset_name=self.dataset_name,
                model_name=model_name,
                subject_id=self.subject_id,
                model_code=model_code,
                model_save_dir=model_save_dir,
                config_overrides=config_overrides,
            )
            results = {self.dataset_name: accuracy} if accuracy is not None else {}
            detailed = {self.dataset_name: detail} if detail is not None else {}
            return results, detailed
        else:
            datasets = self.current_datasets
            return run_model_training_all_datasets(
                model_name=model_name,
                datasets=datasets,
                model_code=model_code,
                model_save_dir=model_save_dir,
                config_overrides=config_overrides,
                test_mode=self.test_mode,
            )

    def _run_initial_test(self, details, model_code=None, model_save_dir=None):
        model_name = details['model_name']
        self.current_model_name = model_name

        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] {self._tag}开始初始测试，模型: {model_name}")

        results, detailed_results = self._run_training(
            model_name, model_code, model_save_dir,
        )

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {self._tag}初始测试完成")
        return {
            'success': True,
            'results': results,
            'detailed_results': detailed_results
        }

    def _run_with_improvements(self, details, model_code=None,
                               config_overrides=None, model_save_dir=None):
        model_name = details.get('model_name', self.current_model_name)
        action = details.get('action', 'unknown')

        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] {self._tag}执行改进方案: {action}")

        if model_code:
            print(f"[{ts}] {self._tag}使用改进后的模型代码（保存至 output 迭代文件夹，动态加载）")
        if config_overrides:
            print(f"[{ts}] {self._tag}应用配置覆盖: {json.dumps(config_overrides, ensure_ascii=False)[:200]}")

        results, detailed_results = self._run_training(
            model_name, model_code, model_save_dir,
            config_overrides=config_overrides,
        )

        return {
            'success': True,
            'results': results,
            'detailed_results': detailed_results,
            'changes_made': True
        }

    def compare_with_best(self, current_results, best_accuracy,
                          detailed_results=None, model_name=None,
                          skip_llm=False):
        current_avg = (
            sum(current_results.values()) / len(current_results)
            if current_results else 0.0
        )
        numeric_improved = current_avg > best_accuracy

        llm_evaluation = None
        if not skip_llm and detailed_results and model_name:
            llm_evaluation = self.llm_evaluate_metrics(
                current_results, detailed_results, best_accuracy, model_name
            )

        return {
            'improved': numeric_improved,
            'current_accuracy': current_avg,
            'best_accuracy': best_accuracy,
            'improvement': current_avg - best_accuracy if numeric_improved else 0,
            'llm_evaluation': llm_evaluation
        }

    def llm_evaluate_metrics(self, current_results, detailed_results,
                              best_accuracy, model_name):
        metrics_lines = []
        for dataset_name, dataset_data in detailed_results.items():
            if not isinstance(dataset_data, dict):
                continue
            for subject, sessions in dataset_data.items():
                for session, folds in sessions.items():
                    for fold, metrics in folds.items():
                        metrics_lines.append(
                            f"  {dataset_name} S{subject} Sess{session} F{fold}: "
                            f"mean_acc={metrics.get('mean_accuracy', 0):.4f} "
                            f"max_acc={metrics.get('max_accuracy', 0):.4f} "
                            f"mean_prec={metrics.get('mean_precision', 0):.4f} "
                            f"mean_rec={metrics.get('mean_recall', 0):.4f} "
                            f"mean_f1={metrics.get('mean_f1_score', 0):.4f} "
                            f"mean_kappa={metrics.get('mean_kappa', 0):.4f}"
                        )

        metrics_text = "\n".join(metrics_lines) if metrics_lines else "无详细指标"

        prompt = (
            f"请综合评估模型 {model_name} 的训练结果。\n"
            f"各数据集平均准确率: {json.dumps(current_results, ensure_ascii=False)}\n"
            f"历史最佳平均准确率: {best_accuracy:.4f}\n\n"
            f"各受试者详细指标:\n{metrics_text}\n\n"
            f"请简要总结各指标的表现，100字以内。"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        response = chat(messages, max_tokens=1024, label="execution.llm_evaluate_metrics")
        return response

    def generate_summary_report(self, execution_result, comparison_result):
        return {
            'execution_success': execution_result.get('success', False),
            'test_results': execution_result.get('results', {}),
            'improved': comparison_result.get('improved', False),
            'current_accuracy': comparison_result.get('current_accuracy', 0),
            'best_accuracy': comparison_result.get('best_accuracy', 0),
            'improvement': comparison_result.get('improvement', 0)
        }
