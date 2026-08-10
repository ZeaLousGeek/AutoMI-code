import json
import importlib.util
from datetime import datetime
from pathlib import Path
from src.main.utils.client import chat
from src.main.prompts import load_prompt, load_prompt_template


PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'

DATASET_DESCRIPTIONS = {
    'bcicIV2a': '9 受试者，4 类运动想象，22 通道，高难度多分类任务',
    'OpenBMI': '54 受试者，2 类运动想象，62 通道，大规模高密度数据集',
}


def _build_system_prompt(datasets_in_use):
    template = load_prompt('output_agent')

    ds_lines = []
    for ds in datasets_in_use:
        desc = DATASET_DESCRIPTIONS.get(ds, '未知数据集')
        ds_lines.append(f"- **{ds}**：{desc}")
    ds_text = "\n".join(ds_lines)

    ar_lines = []
    for ds in datasets_in_use:
        ar_lines.append(f'        "{ds}": {{"accuracy": 0.xx, "key_metrics": "..."}}')
    ar_text = ",\n".join(ar_lines)

    return template.replace("{datasets_in_use}", ds_text).replace(
        "{actual_results_template}", ar_text
    )


class OutputAgent:

    def __init__(self, model_name=None, model_path=None,
                 selected_datasets=None,
                 base_output_dir=None, dataset_name=None, subject_id=None):
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)
        self.model_name = model_name or 'EEGNet'
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.subject_id = subject_id

        if base_output_dir:
            self.model_output_dir = Path(base_output_dir)
        else:
            self.timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            self.model_output_dir = self.output_dir / f'{self.model_name}_{self.timestamp}'
        self.model_output_dir.mkdir(parents=True, exist_ok=True)

        self.selected_datasets = selected_datasets or ['bcicIV2a']
        self.system_prompt = _build_system_prompt(self.selected_datasets)

    def get_iteration_folder(self, iteration):
        if self.dataset_name is not None and self.subject_id is not None:
            iter_folder = (self.model_output_dir / f'{iteration:03d}'
                           / self.dataset_name / str(self.subject_id))
        else:
            iter_folder = self.model_output_dir / f'{iteration:03d}'
        iter_folder.mkdir(parents=True, exist_ok=True)
        return iter_folder

    def generate_iteration_log(self, iteration, plan, execution_result,
                               comparison_result, planning_reasoning=None,
                               rl_state_info=None):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        test_results = execution_result.get('results', {})
        detailed_results = execution_result.get('detailed_results', {})
        action = plan.get('action', '未知')

        lines = [
            f"# 迭代 {iteration} 详细日志",
            f"",
            f"**生成时间:** {ts}",
            f"**模型:** {self.model_name}",
            f"**执行动作:** {action}",
            f"",
            f"---",
            f"",
            f"## 一、规划阶段",
            f"",
            f"### RL 决策",
            f"",
        ]

        if rl_state_info:
            lines.append(f"- 动作方向: {rl_state_info.get('action_name', action)}")
            lines.append(f"- 探索率: {rl_state_info.get('exploration_rate', 'N/A')}")
            lines.append(f"")

        reasoning = plan.get('reasoning', '')
        lines.append(f"### LLM 分析与推理")
        lines.append(f"")
        if reasoning:
            lines.append(f"{reasoning}")
        else:
            lines.append(f"无详细推理记录")

        plan_for_log = {k: v for k, v in plan.items()
                        if k not in ('llm_reasoning', 'model_code', 'training_strategy')}
        if plan.get('model_code'):
            plan_for_log['model_code'] = "参见 model_improved.py"
        if plan.get('training_strategy'):
            plan_for_log['training_strategy'] = "参见 training_strategy.py"

        lines.extend([
            f"",
            f"### 执行计划",
            f"",
            f"```json",
            json.dumps(plan_for_log, ensure_ascii=False, indent=2, default=str),
            f"```",
            f"",
        ])

        retrieve = plan.get('retrieve')
        if retrieve:
            lines.extend([
                f"### 参考文献检索",
                f"",
            ])
            for idx, paper_info in sorted(retrieve.items(), key=lambda x: str(x[0])):
                lines.append(f"**{idx}. {paper_info.get('name', '')}** (arXiv: {paper_info.get('number', '')})")
                lines.append(f"")
                lines.append(f"{paper_info.get('content', '')}")
                lines.append(f"")

        lines.extend([
            f"---",
            f"",
            f"## 二、执行阶段",
            f"",
            f"### 训练结果",
            f"",
        ])

        for ds, acc in test_results.items():
            lines.append(f"- **{ds}**: 平均准确率 {acc * 100:.2f}%")

        lines.extend([
            f"",
            f"### 详细指标",
            f"",
        ])

        metrics_text = self._format_detailed_results(detailed_results)
        lines.append(metrics_text)

        lines.extend([
            f"",
            f"---",
            f"",
            f"## 三、评估阶段",
            f"",
        ])

        if comparison_result:
            improved = comparison_result.get('improved', False)
            lines.append(f"- **是否改进:** {'是' if improved else '否'}")
            lines.append(
                f"- **当前准确率:** "
                f"{comparison_result.get('current_accuracy', 0) * 100:.2f}%"
            )
            lines.append(
                f"- **历史最佳:** "
                f"{comparison_result.get('best_accuracy', 0) * 100:.2f}%"
            )
            if comparison_result.get('llm_evaluation'):
                lines.extend([
                    f"",
                    f"### LLM 评估",
                    f"",
                    f"{comparison_result['llm_evaluation']}",
                ])

        lines.append(f"")
        return "\n".join(lines)

    def generate_no_improvement_analysis(self, iteration, test_results, plan,
                                         detailed_results=None):
        action = plan.get('action', '未知') if plan else '未知'
        plan_desc = json.dumps(plan, ensure_ascii=False, indent=2) if plan else '无'
        metrics_summary = self._format_detailed_results(detailed_results)

        prompt_template = load_prompt_template(
            'output_agent', 'NO_IMPROVEMENT_ANALYSIS_PROMPT'
        )
        prompt = prompt_template.format(
            model_name=self.model_name,
            iteration=iteration,
            action=action,
            plan_desc=plan_desc,
            test_results_json=json.dumps(test_results, ensure_ascii=False),
            metrics_summary=metrics_summary,
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        analysis = chat(messages, max_tokens=2048, label="output.no_improvement_analysis")
        if analysis is None:
            analysis = f"迭代 {iteration} 未产生有效改进，LLM分析不可用。"
        return analysis

    def generate_improvement_suggestions(self, iteration, test_results,
                                          detailed_results=None):
        from src.main.tools.web_search import search_papers, search_github

        query = self._generate_search_query(iteration, test_results)
        references = []

        papers = search_papers(query, max_results=3)
        for paper in papers:
            references.append(
                f"论文: {paper.get('title', '')} - {paper.get('summary', '')[:200]}"
            )

        repos = search_github(f"{self.model_name} EEG BCI", max_results=3)
        for repo in repos:
            references.append(
                f"GitHub: {repo.get('name', '')} - {repo.get('description', '')}"
            )

        search_context = "\n".join(references) if references else "未找到相关参考资料"
        metrics_summary = self._format_detailed_results(detailed_results)

        prompt_template = load_prompt_template(
            'output_agent', 'IMPROVEMENT_SUGGESTION_PROMPT'
        )
        prompt = prompt_template.format(
            model_name=self.model_name,
            iteration=iteration,
            test_results_json=json.dumps(test_results, ensure_ascii=False),
            metrics_summary=metrics_summary,
            search_context=search_context,
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        response = chat(messages, max_tokens=2048, label="output.improvement_suggestions")
        if response is None:
            response = json.dumps([{
                "improvement_name": "LLM不可用_默认建议",
                "principle": "LLM服务暂时不可用",
                "implementation": "请检查API配置后重试"
            }], ensure_ascii=False, indent=2)

        return response

    def _generate_search_query(self, iteration, test_results):
        from src.main.tools.web_search import get_searched_titles

        prompt_template = load_prompt_template(
            'planning_agent', 'SEARCH_QUERY_PROMPT'
        )
        prompt = prompt_template.format(
            model_name=self.model_name,
            iteration=iteration,
            current_accuracy=sum(test_results.values()) / len(test_results) if test_results else 0.0,
            last_action='improvement_suggestions',
            searched_titles=", ".join(get_searched_titles()) or "无",
        )

        messages = [{"role": "user", "content": prompt}]
        try:
            query = chat(messages, max_tokens=100, label="output.search_query")
            if query and query.strip():
                return query.strip()
        except Exception:
            pass
        return f"{self.model_name} motor imagery EEG classification improvement"

    def format_results_dataframe(self, test_results, detailed_results):
        rows = []
        if detailed_results:
            for dataset_name, dataset_data in detailed_results.items():
                if not isinstance(dataset_data, dict):
                    rows.append({
                        'dataset': dataset_name,
                        'mean_accuracy': dataset_data,
                    })
                    continue
                for subject, sessions in dataset_data.items():
                    for session, folds in sessions.items():
                        for fold, metrics in folds.items():
                            row = {
                                'dataset': dataset_name,
                                'subject': int(subject),
                                'session': int(session),
                                'fold': int(fold),
                            }
                            row.update(metrics)
                            rows.append(row)
        else:
            for ds, acc in test_results.items():
                rows.append({'dataset': ds, 'mean_accuracy': acc})
        return rows

    def _format_detailed_results(self, detailed_results):
        if not detailed_results:
            return "无详细指标数据"

        lines = []
        for dataset_name, dataset_data in detailed_results.items():
            lines.append(f"\n数据集: {dataset_name}")
            if not isinstance(dataset_data, dict):
                lines.append(f"  平均准确率: {dataset_data}")
                continue
            for subject, sessions in dataset_data.items():
                for session, folds in sessions.items():
                    for fold, metrics in folds.items():
                        lines.append(
                            f"  Subject={subject} Session={session} Fold={fold}: "
                            f"mean_acc={metrics.get('mean_accuracy', 'N/A'):.4f} "
                            f"max_acc={metrics.get('max_accuracy', 'N/A'):.4f} "
                            f"precision={metrics.get('mean_precision', 'N/A'):.4f} "
                            f"recall={metrics.get('mean_recall', 'N/A'):.4f} "
                            f"f1={metrics.get('mean_f1_score', 'N/A'):.4f} "
                            f"kappa={metrics.get('mean_kappa', 'N/A'):.4f}"
                        )
        return "\n".join(lines) if lines else "无详细指标数据"
