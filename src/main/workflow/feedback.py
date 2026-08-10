from datetime import datetime
from src.main.workflow.state import AgentState


def _subject_tag(state: AgentState) -> str:
    ds = state.get('dataset_name', '')
    sid = state.get('subject_id', 0)
    if ds and sid:
        return f"[{ds}/Sub{sid}] "
    return ""


def feedback_node(state: AgentState) -> AgentState:
    tag = _subject_tag(state)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("=" * 60)
    print(f"{tag}迭代 {state['iteration'] + 1}/{state['max_iterations']} - 反馈阶段")
    print("=" * 60)
    print(f"[{ts}] {tag}[开始反馈阶段]")

    planning_agent = state['planning_agent']
    done = state['iteration'] >= state['max_iterations'] - 1
    execution_failed = not state.get('execution_result', {}).get('success', False)

    rl_path = state.get('rl_save_path', '')

    if execution_failed:
        ts2 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts2}] {tag}上一轮执行失败，跳过 RL 更新，回滚模型代码到规划前版本")
        rollback_code = state.get('model_code_before_planning')
        if rollback_code:
            state['current_model_code'] = rollback_code
        elif state.get('best_model_code'):
            state['current_model_code'] = state['best_model_code']
    else:
        planning_agent.update_rl_agent(
            state['old_accuracy'],
            state['current_accuracy'],
            done
        )
        if rl_path:
            planning_agent.save_rl_state(rl_path)

        last_entry = state['iteration_history'][-1] if state['iteration_history'] else None

        if last_entry:
            last_action = last_entry.get('plan', {}).get('action', '')
            improved = last_entry.get('improved', False)
            if last_action == 'parameter_evolution' and not improved:
                planning_agent.consecutive_param_failures += 1
            else:
                planning_agent.consecutive_param_failures = 0

        if last_entry:
            improved = last_entry.get('improved', False)

            if state.get('in_structure_tuning_phase'):
                if improved:
                    state['in_structure_tuning_phase'] = False
                    state['structure_tuning_remaining'] = 0
                    state['model_code_before_structure_update'] = None
                    ts2 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[{ts2}] {tag}结构调优阶段：参数调优成功改进，退出调优阶段")
                else:
                    state['structure_tuning_remaining'] -= 1
                    ts2 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    if state['structure_tuning_remaining'] <= 0:
                        rollback_code = state.get('model_code_before_structure_update')
                        if rollback_code:
                            state['current_model_code'] = rollback_code
                        elif state.get('best_model_code'):
                            state['current_model_code'] = state['best_model_code']
                        state['in_structure_tuning_phase'] = False
                        state['force_structure_update'] = True
                        state['model_code_before_structure_update'] = None
                        print(
                            f"[{ts2}] {tag}结构调优阶段：参数调优次数耗尽仍未改进，"
                            f"回退到结构更新前的模型，下轮强制 structure_update"
                        )
                    else:
                        print(
                            f"[{ts2}] {tag}结构调优阶段：参数调优未改进，"
                            f"保持新结构继续调优 (剩余 {state['structure_tuning_remaining']} 次)"
                        )
            elif not improved:
                last_action = last_entry.get('plan', {}).get('action', '') if last_entry else ''
                ts2 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                if last_action == 'structure_update':
                    max_pf = planning_agent.max_consecutive_param_failures
                    state['in_structure_tuning_phase'] = True
                    state['structure_tuning_remaining'] = max_pf
                    state['model_code_before_structure_update'] = state.get('best_model_code')
                    planning_agent.consecutive_param_failures = 0
                    print(
                        f"[{ts2}] {tag}structure_update 未立即改进，进入结构调优阶段，"
                        f"保留新结构，后续 {max_pf} 次 parameter_evolution"
                    )
                elif (planning_agent.consecutive_param_failures
                      >= planning_agent.max_consecutive_param_failures):
                    state['force_structure_update'] = True
                    planning_agent.consecutive_param_failures = 0
                    print(
                        f"[{ts2}] {tag}连续 "
                        f"{planning_agent.max_consecutive_param_failures} 次未改进，"
                        f"下轮强制 structure_update（保留当前模型）"
                    )
                elif state.get('best_model_code'):
                    state['current_model_code'] = state['best_model_code']
                    print(
                        f"[{ts2}] {tag}上一轮未改进，回滚模型代码到最佳版本 "
                        f"(最佳准确率: {state['best_accuracy'] * 100:.2f}%)"
                    )

    state['done'] = done

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {tag}[反馈阶段结束]")
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"{tag}[迭代周期 {state['iteration'] + 1}/{state['max_iterations']} 结束]"
    )
    print("=" * 80 + "\n")

    state['iteration'] += 1
    return state


def should_continue(state: AgentState) -> str:
    if state['done']:
        return 'output'
    return 'planning'
