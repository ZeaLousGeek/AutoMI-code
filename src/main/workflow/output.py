import sys
import json
import importlib.util
from datetime import datetime
from src.main.workflow.state import AgentState

import yaml


def _subject_tag(state) -> str:
    ds = state.get('dataset_name', '')
    sid = state.get('subject_id', 0)
    if ds and sid:
        return f"[{ds}/Sub{sid}] "
    return ""


def _write_live_summary(state, iteration):
    from src.main.tools.summary_tool import (
        append_iteration_log, update_accuracy_summary,
        update_accuracy_summary_for_subject,
    )
    from src.main.utils.token_usage_context import finalize_iteration_token_file

    output_agent = state['output_agent']
    model_output_dir = output_agent.model_output_dir

    iter_folder = output_agent.get_iteration_folder(iteration)
    finalize_iteration_token_file(iter_folder, iteration)

    log_file = iter_folder / 'iteration_log.md'
    if log_file.exists():
        log_content = log_file.read_text(encoding='utf-8')
        append_iteration_log(model_output_dir, iteration, log_content,
                             dataset_name=state.get('dataset_name'),
                             subject_id=state.get('subject_id'))

    ds = state.get('dataset_name')
    sid = state.get('subject_id')
    if ds and sid:
        update_accuracy_summary_for_subject(
            model_output_dir,
            dataset_name=ds,
            subject_id=sid,
            iteration_history=state['iteration_history'],
            best_accuracy=state['best_accuracy'],
        )
    else:
        update_accuracy_summary(
            model_output_dir,
            state['iteration_history'],
            state['best_accuracy'],
        )


def _write_error_iteration_output(state, plan, error_msg,
                                  model_code_before, model_code_after):
    output_agent = state['output_agent']
    iteration = state['iteration'] + 1
    iter_folder = output_agent.get_iteration_folder(iteration)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] 写入失败迭代 {iteration} 的输出文件到: {iter_folder}")

    plan_safe = {k: v for k, v in plan.items() if k != 'llm_reasoning'}
    if plan_safe.get('model_code'):
        plan_safe['model_code'] = "参见 model_improved.py"
    plan_safe['execution_error'] = error_msg
    (iter_folder / 'plan.json').write_text(
        json.dumps(plan_safe, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8'
    )

    if model_code_before:
        (iter_folder / 'model_before.py').write_text(
            model_code_before, encoding='utf-8'
        )

    model_improved_file = iter_folder / 'model_improved.py'
    if not model_improved_file.exists():
        model_improved = model_code_after or model_code_before
        if model_improved:
            model_improved_file.write_text(model_improved, encoding='utf-8')

    log_content = (
        f"# 迭代 {iteration} - 执行失败\n\n"
        f"## 规划\n"
        f"- 动作: {plan.get('action', 'unknown')}\n"
        f"- 推理: {plan.get('reasoning', '')[:300]}\n\n"
        f"## 错误信息\n\n"
        f"```\n{error_msg}\n```\n\n"
        f"本轮训练因模型代码异常未能完成，下一轮迭代将基于此错误修复模型。\n"
    )
    (iter_folder / 'iteration_log.md').write_text(
        log_content, encoding='utf-8'
    )

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 失败迭代 {iteration} 输出已保存")


def _write_iteration_output(state, plan, execution_result, comparison,
                            model_code_before, model_code_after,
                            no_improve_analysis=None):
    output_agent = state['output_agent']
    iteration = state['iteration'] + 1
    test_results = execution_result.get('results', {})
    detailed_results = execution_result.get('detailed_results', {})
    improved = comparison.get('improved', False)

    iter_folder = output_agent.get_iteration_folder(iteration)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] 开始写入迭代 {iteration} 的输出文件到: {iter_folder}")

    plan_safe = {k: v for k, v in plan.items() if k != 'llm_reasoning'}
    if plan_safe.get('model_code'):
        plan_safe['model_code'] = "参见 model_improved.py"
    if (plan_safe.get('training_strategy')
            and isinstance(plan_safe['training_strategy'], str)
            and plan_safe['training_strategy'].strip()):
        plan_safe['training_strategy'] = "参见 training_strategy.py"
    (iter_folder / 'plan.json').write_text(
        json.dumps(plan_safe, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8'
    )

    if model_code_before:
        (iter_folder / 'model_before.py').write_text(
            model_code_before, encoding='utf-8'
        )

    model_improved_file = iter_folder / 'model_improved.py'
    if not model_improved_file.exists():
        model_improved = model_code_after or model_code_before
        if model_improved:
            model_improved_file.write_text(model_improved, encoding='utf-8')

    config_overrides = plan.get('config_overrides')
    if config_overrides:
        (iter_folder / 'config_improved.yaml').write_text(
            yaml.dump(config_overrides, allow_unicode=True, default_flow_style=False),
            encoding='utf-8'
        )
    else:
        (iter_folder / 'config_improved.yaml').write_text(
            "# 本轮无参数配置变更\n", encoding='utf-8'
        )

    training_strategy = plan.get('training_strategy')
    if training_strategy and isinstance(training_strategy, str) and training_strategy.strip():
        (iter_folder / 'training_strategy.py').write_text(
            training_strategy, encoding='utf-8'
        )
    else:
        (iter_folder / 'training_strategy.py').write_text(
            "# 本轮无训练策略变更\n", encoding='utf-8'
        )

    _write_results_excel(iter_folder, test_results, detailed_results, output_agent)

    log_content = output_agent.generate_iteration_log(
        iteration=iteration,
        plan=plan,
        execution_result=execution_result,
        comparison_result=comparison,
        rl_state_info={
            'action_name': plan.get('action', ''),
            'exploration_rate': state['planning_agent'].rl_agent.exploration_rate,
        }
    )

    if not improved:
        if no_improve_analysis is None:
            no_improve_analysis = output_agent.generate_no_improvement_analysis(
                iteration, test_results, plan, detailed_results
            )
        log_content += f"\n\n---\n\n## 未改进分析\n\n{no_improve_analysis}\n"

    (iter_folder / 'iteration_log.md').write_text(log_content, encoding='utf-8')

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 迭代 {iteration} 输出已保存")


def _write_results_excel(iter_folder, test_results, detailed_results,
                         output_agent):
    rows = output_agent.format_results_dataframe(test_results, detailed_results)

    pandas_available = importlib.util.find_spec('pandas') is not None
    openpyxl_available = importlib.util.find_spec('openpyxl') is not None

    if pandas_available and openpyxl_available and rows:
        import pandas as pd
        excel_file = iter_folder / 'results.xlsx'
        df = pd.DataFrame(rows)
        df.to_excel(excel_file, index=False, engine='openpyxl')
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Excel结果已保存: {excel_file}"
        )
    else:
        csv_file = iter_folder / 'results.csv'
        if rows:
            import csv
            with open(csv_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        else:
            with open(csv_file, 'w', encoding='utf-8') as f:
                f.write("Dataset,Accuracy\n")
                for ds, acc in test_results.items():
                    f.write(f"{ds},{acc}\n")
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"CSV结果已保存: {csv_file}"
        )


def _compute_subject_accuracies(detailed_results):
    per_dataset = {}
    if not detailed_results or not isinstance(detailed_results, dict):
        return per_dataset
    for ds_name, ds_data in detailed_results.items():
        if not isinstance(ds_data, dict):
            continue
        subject_accs = {}
        for subject, sessions in ds_data.items():
            acc_vals = []
            for session, folds in sessions.items():
                for fold, metrics in folds.items():
                    v = metrics.get('max_accuracy')
                    if v is not None:
                        acc_vals.append(v)
            if acc_vals:
                subject_accs[subject] = sum(acc_vals) / len(acc_vals)
        per_dataset[ds_name] = subject_accs
    return per_dataset


def output_node(state: AgentState) -> AgentState:
    tag = _subject_tag(state)
    print("=" * 60)
    print(f"{tag}最终阶段 - 系统运行总结")
    print("=" * 60)

    print(f"{tag}系统运行完成!")
    print(f"{tag}最佳准确率: {state['best_accuracy'] * 100:.2f}%")
    print(f"{tag}总迭代次数: {state['iteration']}")

    print("\n===== 迭代准确率汇总 =====")
    for entry in state['iteration_history']:
        it = entry.get('iteration', '?')
        overall = entry.get('accuracy', 0)
        line = f"迭代 {it}: Overall={overall * 100:.2f}%"

        ds_results = entry.get('results', {})
        detailed = entry.get('detailed_results', {})
        subject_info = _compute_subject_accuracies(detailed)

        ds_parts = []
        for ds_name, ds_acc in ds_results.items():
            subj_accs = subject_info.get(ds_name, {})
            subj_str = ", ".join(
                f"Sub{s}={a * 100:.1f}%"
                for s, a in sorted(subj_accs.items(), key=lambda x: int(x[0]))
            )
            ds_part = f"{ds_name}({ds_acc * 100:.2f}%)"
            if subj_str:
                ds_part += f" [{subj_str}]"
            ds_parts.append(ds_part)

        if ds_parts:
            line += "  | " + "  ".join(ds_parts)
        print(line)

    print("=" * 60)

    from src.main.tools.summary_tool import generate_summary
    output_agent = state['output_agent']
    generate_summary(
        model_output_dir=output_agent.model_output_dir,
        iteration_history=state['iteration_history'],
        best_accuracy=state['best_accuracy'],
        dataset_name=state.get('dataset_name'),
        subject_id=state.get('subject_id'),
    )

    log_fh = state.get('tee_log_file')
    if log_fh:
        try:
            sys.stdout = sys.__stdout__
            log_fh.close()
        except Exception:
            pass

    return state
