import os
import json
import operator
from pathlib import Path
from src.main.utils.client import chat
from src.main.prompts import load_prompt, load_prompt_template
from src.main.rl.q_learning import QLearningAgent, create_state_representation, calculate_reward


PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / 'src' / 'models'
MODEL_REPO_URL = 'https://github.com/ZeaLousGeek/Awesome-MI-EEG-Classification'

SYSTEM_PROMPT_TEMPLATE = load_prompt('planning_agent')

DATASET_DESCRIPTIONS = {
    'bcicIV2a': '9 个受试者，4 类运动想象（左手、右手、双脚、舌头），22 个 EEG 通道，采样率 250Hz，时间窗 0.5-3.5s',
    'OpenBMI': '54 个受试者，2 类运动想象（左手、右手），62 个 EEG 通道，采样率 250Hz',
}


def _build_dataset_info(selected_datasets):
    lines = []
    for ds in selected_datasets:
        desc = DATASET_DESCRIPTIONS.get(ds, '未知数据集')
        lines.append(f"- **{ds}**：{desc}")
    return "\n".join(lines)


class PlanningAgent:
    ALL_DATASET_PATHS = {
        'bcicIV2a': os.path.join(os.environ.get("AUTOMI_DATA_ROOT", ""), 'bcicIV2a', 'gdf'),
        'OpenBMI': os.path.join(os.environ.get("AUTOMI_DATA_ROOT", ""), 'OpenBMI', 'mat'),
    }

    def __init__(self, selected_datasets=None, max_consecutive_param_failures=3,
                 dataset_name=None, subject_id=None, ablation_mode=None):
        self.rl_agent = QLearningAgent(
            state_size=3,
            action_size=3
        )
        self.accuracy_history = []
        self.iteration_count = 0
        self.current_model = None
        self.best_accuracy = 0.0
        self.best_model = None
        self.last_action = 0
        self.selected_datasets = selected_datasets or ['bcicIV2a']
        self.max_consecutive_param_failures = max_consecutive_param_failures
        self.consecutive_param_failures = 0
        self.dataset_name = dataset_name
        self.subject_id = subject_id
        self.ablation_mode = ablation_mode

        dataset_info = _build_dataset_info(self.selected_datasets)
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.replace("{dataset_info}", dataset_info)

    def check_existing_models(self):
        if not MODELS_DIR.exists():
            return False
        model_files = list(MODELS_DIR.glob('*.py'))
        model_dirs = [d for d in MODELS_DIR.iterdir() if d.is_dir()]
        return len(model_files) > 0 or len(model_dirs) > 0

    def fetch_models_from_repo(self):
        from src.main.tools.code_fetcher import list_repo_models
        print("从 Awesome-MI-EEG-Classification 仓库查找模型...")
        result = list_repo_models()
        models = result.get('awesome_models', [])
        if not models:
            print("未能从仓库获取模型，使用本地已有的模型")
        return models

    def download_model_code(self, repo_url, model_name):
        from src.main.tools.code_fetcher import download_model_from_repo
        result = download_model_from_repo(repo_url, model_name)
        return result is not None

    def select_initial_model(self):
        if self.check_existing_models():
            model_files = list(MODELS_DIR.glob('*.py'))
            if model_files:
                model_files.sort(key=operator.attrgetter('name'))
                return model_files[0].stem
        return 'EEGNet'

    def create_test_plan(self, model_name=None):
        datasets = list(self.selected_datasets)
        dataset_paths = {
            ds: self.ALL_DATASET_PATHS[ds] for ds in datasets
        }

        if model_name is None:
            model_name = self.select_initial_model()

        self.current_model = model_name

        return {
            'model_name': model_name,
            'datasets': datasets,
            'dataset_paths': dataset_paths,
            'action': 'initial_test'
        }

    def analyze_results(self, test_results):
        avg_accuracy = sum(test_results.values()) / len(test_results) if test_results else 0.0
        self.accuracy_history.append(avg_accuracy)

        if avg_accuracy > self.best_accuracy:
            self.best_accuracy = avg_accuracy
            self.best_model = self.current_model

        print(f"平均准确率: {avg_accuracy * 100:.2f}%")
        print(f"历史最佳准确率: {self.best_accuracy * 100:.2f}%")

        return avg_accuracy

    def make_decision(self, current_accuracy):
        state = create_state_representation(
            current_accuracy,
            self.accuracy_history,
            self.iteration_count
        )

        if self.ablation_mode == 'random-action':
            import random
            action = random.randint(0, 2)
            action_name = self.rl_agent.get_action_name(action)
            print(f"消融实验 (random-action): 随机选择动作 {action_name}")
            return action, action_name

        if self.ablation_mode == 'no-structure-update':
            import numpy as np
            state_key = str(state)
            if state_key not in self.rl_agent.q_table:
                self.rl_agent.q_table[state_key] = np.array([0.0, -1000.0, 0.0])
            else:
                if not isinstance(self.rl_agent.q_table[state_key], np.ndarray):
                    self.rl_agent.q_table[state_key] = np.array(self.rl_agent.q_table[state_key])
                self.rl_agent.q_table[state_key][1] = -1000.0
            action = self.rl_agent.select_action(state)
            if action == 1:
                action = 0
            action_name = self.rl_agent.get_action_name(action)
            print(f"消融实验 (no-structure-update): RL决策={action_name}")
            return action, action_name

        action = self.rl_agent.select_action(state)
        action_name = self.rl_agent.get_action_name(action)

        print(f"RL决策: {action_name}")
        return action, action_name

    def create_improvement_plan(self, action_name, current_accuracy,
                                model_code=None, experience_text=None,
                                error_info=None):
        if (action_name == 'parameter_evolution'
                and self.consecutive_param_failures >= self.max_consecutive_param_failures):
            from datetime import datetime
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(
                f"[{ts}] 连续 {self.consecutive_param_failures} 次参数优化未改进，"
                f"强制切换为 structure_update"
            )
            action_name = 'structure_update'

        plan = self._llm_generate_improvement_plan(
            action_name, current_accuracy, model_code,
            experience_text=experience_text,
            error_info=error_info,
        )
        plan['datasets'] = list(self.selected_datasets)
        plan['dataset_paths'] = {
            ds: self.ALL_DATASET_PATHS[ds] for ds in self.selected_datasets
        }
        return plan

    def _generate_search_query(self, action_name, current_accuracy,
                               experience_text=None):
        from src.main.tools.web_search import get_searched_titles

        recent_history = ""
        if experience_text:
            recent_history = experience_text[-500:]

        prompt_template = load_prompt_template(
            'planning_agent', 'SEARCH_QUERY_PROMPT'
        )
        prompt = prompt_template.format(
            model_name=self.current_model,
            iteration=self.iteration_count + 1,
            current_accuracy=current_accuracy,
            current_action=action_name,
            recent_history=recent_history or "无",
            searched_titles=", ".join(get_searched_titles()) or "无",
        )

        messages = [{"role": "user", "content": prompt}]
        try:
            query = chat(messages, max_tokens=100, label="planning.search_query")
            if query and query.strip():
                return query.strip()
        except Exception:
            pass

        fallback_queries = {
            'parameter_evolution': f"{self.current_model} motor imagery EEG hyperparameter optimization training strategy",
            'structure_update': f"{self.current_model} motor imagery EEG model architecture attention mechanism",
            'continue_current': f"{self.current_model} motor imagery EEG training tricks data augmentation",
        }
        return fallback_queries.get(action_name, f"{self.current_model} motor imagery EEG improvement")

    def _llm_generate_improvement_plan(self, action_name, current_accuracy,
                                       model_code=None,
                                       experience_text=None,
                                       error_info=None):
        from datetime import datetime

        reference_context = ""
        retrieve_dict = {}

        need_arxiv = action_name == 'structure_update'

        if self.ablation_mode == 'no-literature':
            need_arxiv = False
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{ts}] 消融实验 (no-literature): 禁用文献检索")

        if need_arxiv:
            from src.main.tools.web_search import search_and_extract_suggestions
            query = self._generate_search_query(
                action_name, current_accuracy, experience_text
            )
            suggestions_text, papers, retrieve_dict = search_and_extract_suggestions(
                query,
                model_name=self.current_model,
                max_papers=5,
                action_name=action_name,
            )
            if suggestions_text:
                reference_context = f"LLM 提炼的改进建议:\n{suggestions_text}"
            elif papers:
                ref_lines = []
                for paper in papers:
                    ref_lines.append(
                        f"- {paper.get('title', '')}: {paper.get('summary', '')[:200]}"
                    )
                reference_context = "\n".join(ref_lines)
        else:
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{ts}] 动作方向为 {action_name}，跳过 arXiv 检索")

        if self.ablation_mode == 'no-experience':
            experience_section = "无历史迭代经验（消融实验：禁用经验追踪）"
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{ts}] 消融实验 (no-experience): 禁用经验追踪")
        else:
            experience_section = experience_text or "无历史迭代经验"

        model_code_text = model_code if model_code else "未提供当前模型代码"

        error_section = ""
        if error_info:
            error_section = (
                f"\n=== 上一轮执行错误（必须优先修复） ===\n"
                f"上一轮改进后的模型在训练时发生了以下错误，"
                f"本轮必须首先修复这个问题，确保模型能正常运行：\n"
                f"```\n{error_info}\n```\n"
                f"请仔细分析错误原因，在生成的 model_code 中修复该问题。\n"
                f"修复后的代码必须保证不会再出现同样的错误。\n\n"
            )

        subject_context = ""
        if self.dataset_name and self.subject_id:
            subject_context = (
                f"优化目标: 数据集 {self.dataset_name} 的受试者 {self.subject_id}\n"
            )

        if self.ablation_mode == 'random-action':
            action_source = f"随机选择的动作方向（消融实验：禁用RL决策）: {action_name}"
        else:
            action_source = f"RL选择的动作方向: {action_name}"

        prompt = (
            f"当前模型: {self.current_model}\n"
            f"{subject_context}"
            f"当前平均准确率: {current_accuracy:.4f}\n"
            f"历史最佳准确率: {self.best_accuracy:.4f}\n"
            f"{action_source}\n\n"
            f"{error_section}"
            f"=== 前序迭代经验（按结构分组） ===\n{experience_section}\n\n"
            f"当前模型代码:\n```python\n{model_code_text}\n```\n\n"
            f"相关论文参考:\n{reference_context or '无（本轮未检索）'}\n\n"
            f"请基于以上信息，在 {action_name} 方向上生成具体的改进方案。\n"
            f"注意避免重复已尝试过且无效的改动。\n"
            f"必须返回严格的 JSON 格式，包含 action, reasoning, model_code, "
            f"config_overrides, training_strategy, improvements 字段。\n"
            f"如果该方向不涉及某个字段的改动，将其设为 null。\n"
            f"model_code 如果有改动必须是完整的 Python 文件代码。"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        response = chat(messages, max_tokens=4096, label="planning.improvement_plan")

        plan = self._parse_llm_plan(response, action_name)
        plan['llm_reasoning'] = response
        plan['retrieve'] = retrieve_dict
        return plan

    def _parse_llm_plan(self, response, action_name):
        default_plan = {
            'action': action_name,
            'reasoning': '',
            'model_code': None,
            'config_overrides': None,
            'training_strategy': None,
            'improvements': [],
        }

        if response is None:
            default_plan['reasoning'] = 'LLM 不可用，使用默认配置'
            return default_plan

        try:
            text = response.strip()
            json_start = text.find('{')
            json_end = text.rfind('}')
            if json_start != -1 and json_end != -1:
                json_text = text[json_start:json_end + 1]
                parsed = json.loads(json_text)
                for key in default_plan:
                    if key in parsed:
                        default_plan[key] = parsed[key]
                default_plan['action'] = action_name
                return default_plan
        except (json.JSONDecodeError, ValueError):
            pass

        default_plan['reasoning'] = response[:500] if response else ''
        return default_plan

    def _get_structure_improvement_ideas(self):
        from src.main.tools.web_search import search_papers

        search_results = search_papers(
            f"{self.current_model} motor imagery EEG improvement", max_results=3
        )
        reference_context = ""
        if search_results:
            ref_lines = []
            for paper in search_results:
                ref_lines.append(
                    f"- {paper.get('title', '')}: {paper.get('summary', '')[:150]}"
                )
            reference_context = "\n相关论文参考:\n" + "\n".join(ref_lines)

        prompt = (
            f"基于运动想象脑电信号分类模型 {self.current_model}，\n"
            f"请提供具体的模型结构改进建议，\n"
            f"每个建议包含：\n"
            f"1. 改进名称\n"
            f"2. 改进原理\n"
            f"3. 具体实现思路\n"
            f"{reference_context}\n\n"
            f"请以JSON格式返回。"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        response = chat(messages, max_tokens=2048, label="planning.structure_ideas")
        if response is not None:
            return response
        return [
            {
                'name': '添加注意力机制',
                'principle': '通过注意力机制聚焦重要特征',
                'implementation': '在卷积层后添加SE-Block'
            }
        ]

    def update_rl_agent(self, old_accuracy, new_accuracy, done=False):
        if old_accuracy is not None:
            old_state = create_state_representation(
                old_accuracy,
                self.accuracy_history[:-1],
                self.iteration_count - 1
            )

            new_state = create_state_representation(
                new_accuracy,
                self.accuracy_history,
                self.iteration_count
            )

            reward = calculate_reward(old_accuracy, new_accuracy)

            self.rl_agent.update_q_value(
                old_state, self.last_action, reward, new_state, done
            )

        self.iteration_count += 1

    def save_rl_state(self, filepath):
        self.rl_agent.save(filepath)

    def load_rl_state(self, filepath):
        self.rl_agent.load(filepath)
