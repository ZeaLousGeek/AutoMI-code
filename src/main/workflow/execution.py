import traceback
from contextvars import copy_context
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from src.main.workflow.state import AgentState
from src.main.utils.token_usage_context import set_token_log_targets
from src.main.workflow.output import (
    _write_live_summary,
    _write_iteration_output,
    _write_error_iteration_output,
)


def _truncate_traceback(tb_string, max_lines=15):
    lines = tb_string.strip().splitlines()
    if len(lines) <= max_lines:
        return tb_string.strip()
    return "\n".join(lines[-max_lines:])


def _subject_tag(state: AgentState) -> str:
    ds = state.get('dataset_name', '')
    sid = state.get('subject_id', 0)
    if ds and sid:
        return f"[{ds}/Sub{sid}] "
    return ""


def execution_node(state: AgentState) -> AgentState:
    tag = _subject_tag(state)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("=" * 60)
    print(f"{tag}迭代 {state['iteration'] + 1}/{state['max_iterations']} - 执行阶段")
    print("=" * 60)
    print(f"[{ts}] {tag}[开始执行阶段]")

    execution_agent = state['execution_agent']
    output_agent = state['output_agent']
    plan = state['current_plan']

    iteration = state['iteration'] + 1
    iter_folder = output_agent.get_iteration_folder(iteration)
    ds = state.get('dataset_name') or None
    sid = state.get('subject_id')
    sid = sid if sid else None
    set_token_log_targets(iter_folder, output_agent.model_output_dir, ds, sid)

    model_code_before = state.get('model_code_before_planning')
    model_code_after = state.get('current_model_code')
    has_code_change = (
        model_code_after is not None
        and model_code_before is not None
        and model_code_after != model_code_before
    )
    initial_code = state.get('initial_model_code')
    is_model_modified = (
        model_code_after is not None
        and initial_code is not None
        and model_code_after != initial_code
    )
    inject_code = model_code_after if (has_code_change or is_model_modified) else None

    try:
        execution_result = execution_agent.execute_plan(
            plan,
            model_code=inject_code,
            model_save_dir=str(iter_folder),
        )
    except Exception:
        tb_string = traceback.format_exc()
        error_summary = _truncate_traceback(tb_string)
        ts_err = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts_err}] 执行阶段发生异常:")
        print(error_summary)
        execution_result = {'success': False, 'error': error_summary}

    state['execution_result'] = execution_result

    if execution_result.get('success'):
        _process_success_result(state, plan, execution_result,
                                model_code_before, model_code_after,
                                has_code_change, iteration)
    else:
        _process_error_result(state, plan, execution_result,
                              has_code_change,
                              model_code_before, model_code_after,
                              iteration)

    _write_live_summary(state, iteration)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {tag}[执行阶段结束]")
    return state


def _process_success_result(state, plan, execution_result,
                            model_code_before, model_code_after,
                            has_code_change, iteration):
    execution_agent = state['execution_agent']
    output_agent = state['output_agent']
    model_name = state['selected_model_name']

    state['last_execution_error'] = None

    test_results = execution_result.get('results', {})
    detailed_results = execution_result.get('detailed_results', {})
    state['test_results'] = test_results

    planning_agent = state['planning_agent']
    avg_accuracy = planning_agent.analyze_results(test_results)

    state['old_accuracy'] = state['current_accuracy']
    state['current_accuracy'] = avg_accuracy

    previous_best = state['best_accuracy']

    comparison = execution_agent.compare_with_best(
        test_results, previous_best,
        detailed_results=detailed_results,
        model_name=model_name,
        skip_llm=True,
    )
    improved = comparison['improved']

    futures = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        if model_name:
            ctx_eval = copy_context()
            futures['llm_eval'] = pool.submit(
                ctx_eval.run,
                execution_agent.llm_evaluate_metrics,
                test_results,
                detailed_results,
                previous_best,
                model_name,
            )
        if not improved:
            ctx_noimp = copy_context()
            futures['no_improve'] = pool.submit(
                ctx_noimp.run,
                output_agent.generate_no_improvement_analysis,
                iteration,
                test_results,
                plan,
                detailed_results,
            )

    llm_evaluation = futures['llm_eval'].result() if 'llm_eval' in futures else None
    no_improve_analysis = futures['no_improve'].result() if 'no_improve' in futures else None
    comparison['llm_evaluation'] = llm_evaluation

    if avg_accuracy > state['best_accuracy']:
        state['best_accuracy'] = avg_accuracy
        state['best_results'] = test_results.copy()
        state['best_model_code'] = state['current_model_code']
        ts2 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts2}] 新的最佳准确率，已保存最佳模型代码")

    history_entry = {
        'iteration': iteration,
        'plan': plan,
        'results': test_results,
        'detailed_results': detailed_results,
        'accuracy': avg_accuracy,
        'improved': improved,
        'config_overrides': plan.get('config_overrides'),
        'model_code_changed': has_code_change,
        'llm_evaluation': llm_evaluation,
        'no_improve_analysis': no_improve_analysis,
    }
    state['iteration_history'].append(history_entry)

    tracker = state['experience_tracker']
    tracker.add_iteration(
        iteration=iteration,
        action=plan.get('action', 'unknown'),
        config_overrides=plan.get('config_overrides'),
        accuracy=avg_accuracy,
        improved=improved,
        model_code_changed=has_code_change,
        reasoning=plan.get('reasoning', ''),
        llm_evaluation=llm_evaluation,
        no_improve_analysis=no_improve_analysis,
    )

    results_text = "\n".join(
        f"- {ds}: {acc * 100:.2f}%" for ds, acc in test_results.items()
    )
    state['iteration_log_parts'].append(
        f"## 执行阶段\n\n"
        f"### 训练结果\n{results_text}\n\n"
        f"### 评估\n"
        f"- 是否改进: {'是' if improved else '否'}\n"
        f"- 当前准确率: {avg_accuracy * 100:.2f}%\n"
        f"- 历史最佳: {state['best_accuracy'] * 100:.2f}%\n"
    )
    if llm_evaluation:
        state['iteration_log_parts'].append(f"\n### LLM 评估\n{llm_evaluation}\n")

    _write_iteration_output(
        state, plan, execution_result, comparison,
        model_code_before, model_code_after,
        no_improve_analysis=no_improve_analysis,
    )


def _process_error_result(state, plan, execution_result, has_code_change,
                          model_code_before, model_code_after, iteration):
    error_msg = execution_result.get('error', '未知错误')
    state['last_execution_error'] = error_msg
    state['old_accuracy'] = state['current_accuracy']

    history_entry = {
        'iteration': iteration,
        'plan': plan,
        'results': {},
        'detailed_results': {},
        'accuracy': 0.0,
        'improved': False,
        'config_overrides': plan.get('config_overrides'),
        'model_code_changed': has_code_change,
        'llm_evaluation': None,
        'no_improve_analysis': None,
        'execution_error': error_msg,
    }
    state['iteration_history'].append(history_entry)

    tracker = state['experience_tracker']
    tracker.add_iteration(
        iteration=iteration,
        action=plan.get('action', 'unknown'),
        config_overrides=plan.get('config_overrides'),
        accuracy=0.0,
        improved=False,
        model_code_changed=has_code_change,
        reasoning=plan.get('reasoning', ''),
        llm_evaluation=None,
        no_improve_analysis=f"执行阶段异常: {error_msg}",
    )

    state['iteration_log_parts'].append(
        f"## 执行阶段\n\n"
        f"### 训练异常\n"
        f"模型训练过程中发生错误，本轮迭代跳过。\n\n"
        f"```\n{error_msg}\n```\n\n"
        f"下一轮迭代将基于此错误信息改进模型代码。\n"
    )

    _write_error_iteration_output(
        state, plan, error_msg, model_code_before, model_code_after,
    )

    tag = _subject_tag(state)
    ts_err = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts_err}] {tag}执行失败，错误已记录，将在下一轮迭代中修复")
