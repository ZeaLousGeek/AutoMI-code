from datetime import datetime

from src.main.utils.token_usage_context import set_token_log_targets
from src.main.workflow.state import AgentState


def _subject_tag(state: AgentState) -> str:
    ds = state.get('dataset_name', '')
    sid = state.get('subject_id', 0)
    if ds and sid:
        return f"[{ds}/Sub{sid}] "
    return ""


def planning_node(state: AgentState) -> AgentState:
    output_agent = state['output_agent']
    next_iter = state['iteration'] + 1
    iter_folder = output_agent.get_iteration_folder(next_iter)
    ds = state.get('dataset_name') or None
    sid = state.get('subject_id')
    sid = sid if sid else None
    set_token_log_targets(iter_folder, output_agent.model_output_dir, ds, sid)

    tag = _subject_tag(state)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "=" * 80)
    print(
        f"{tag}迭代周期 {state['iteration'] + 1}/{state['max_iterations']} 开始 "
        f"- [规划Agent→执行Agent→反馈Agent→规划Agent]"
    )
    print("=" * 80)
    print("=" * 60)
    print(f"{tag}迭代 {state['iteration'] + 1}/{state['max_iterations']} - 规划阶段")
    print("=" * 60)
    print(f"[{ts}] {tag}[开始规划阶段]")

    state['iteration_log_parts'] = []
    state['model_code_before_planning'] = state['current_model_code']

    planning_agent = state['planning_agent']
    selected_model = state['selected_model_name']

    if state['iteration'] == 0:
        if not planning_agent.check_existing_models():
            print("未发现现有模型，正在从仓库获取...")
            planning_agent.fetch_models_from_repo()

        plan = planning_agent.create_test_plan(model_name=selected_model)

        state['iteration_log_parts'].append(
            f"## 规划阶段\n\n"
            f"首轮迭代，执行初始测试。\n"
            f"选定模型: {selected_model}\n"
        )
    else:
        last_error = state.get('last_execution_error')

        action, action_name = planning_agent.make_decision(
            state['current_accuracy']
        )

        if last_error:
            ablation_mode = state.get('ablation_mode')
            if ablation_mode == 'no-structure-update':
                ts2 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{ts2}] 消融实验 (no-structure-update): 检测到执行错误，但禁用 structure_update，保持 {action_name}")
            else:
                ts2 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{ts2}] 检测到上一轮执行错误，强制 structure_update 以修复模型")
                action_name = 'structure_update'

        last_entry = state['iteration_history'][-1] if state['iteration_history'] else None
        last_action_name = last_entry.get('plan', {}).get('action', '') if last_entry else ''

        if not last_error:
            ablation_mode = state.get('ablation_mode')
            if ablation_mode == 'no-structure-update':
                if state.get('force_structure_update'):
                    state['force_structure_update'] = False
                    ts2 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[{ts2}] 消融实验 (no-structure-update): 跳过强制 structure_update，保持 {action_name}")
            elif state.get('force_structure_update'):
                action_name = 'structure_update'
                state['force_structure_update'] = False
                ts2 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{ts2}] 参数调优阶段耗尽，强制执行新的 structure_update")
            elif state.get('in_structure_tuning_phase'):
                action_name = 'parameter_evolution'
                ts2 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(
                    f"[{ts2}] 结构调优阶段中，强制 parameter_evolution "
                    f"(剩余 {state['structure_tuning_remaining']} 次)"
                )

        planning_agent.last_action = action

        experience_text = state['experience_tracker'].build_planning_context(
            keep_recent=5
        )

        plan = planning_agent.create_improvement_plan(
            action_name,
            state['current_accuracy'],
            model_code=state['current_model_code'],
            experience_text=experience_text,
            error_info=last_error,
        )

        state['last_execution_error'] = None

        if selected_model:
            plan['model_name'] = selected_model

        if plan.get('model_code') and isinstance(plan['model_code'], str):
            if plan['model_code'].strip().startswith(
                ('import ', 'from ', '#', 'class ', 'def ')
            ):
                state['current_model_code'] = plan['model_code']
                ts2 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{ts2}] 模型代码已更新（来自 LLM 改进方案）")

        rl_info = (
            f"- RL 动作方向: {action_name}\n"
            f"- 探索率: {planning_agent.rl_agent.exploration_rate:.4f}\n"
        )
        reasoning_text = plan.get('reasoning', '')
        state['iteration_log_parts'].append(
            f"## 规划阶段\n\n"
            f"### RL 决策\n{rl_info}\n"
            f"### LLM 推理\n{reasoning_text}\n"
        )

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {tag}规划完成，动作: {plan.get('action')}")
    print(f"[{ts}] {tag}[规划阶段结束]")
    state['current_plan'] = plan
    return state
