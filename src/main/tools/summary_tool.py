
import json
from pathlib import Path
from datetime import datetime


def _build_accuracy_summary(iteration_history, best_accuracy):
    lines = [
        "# AutoMI 迭代准确率汇总",
        "",
        f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**最佳准确率:** {best_accuracy * 100:.2f}%",
        f"**总迭代次数:** {len(iteration_history)}",
        "",
        "---",
        "",
        "## 各迭代准确率",
        "",
        "| 迭代 | 动作 | Overall | 改进 | 各数据集 |",
        "|------|------|---------|------|----------|",
    ]

    for entry in iteration_history:
        it = entry.get('iteration', '?')
        action = entry.get('plan', {}).get('action', '?')
        overall = entry.get('accuracy', 0)
        improved = '是' if entry.get('improved') else '否'

        ds_parts = []
        for ds, acc in entry.get('results', {}).items():
            ds_parts.append(f"{ds}: {acc * 100:.2f}%")
        ds_text = ", ".join(ds_parts) if ds_parts else "-"

        lines.append(
            f"| {it} | {action} | {overall * 100:.2f}% | {improved} | {ds_text} |"
        )

    lines.extend(["", "---", "", "## 各迭代受试者准确率详情", ""])

    for entry in iteration_history:
        it = entry.get('iteration', '?')
        lines.append(f"### 迭代 {it}")
        lines.append("")

        detailed = entry.get('detailed_results', {})
        if not detailed or not isinstance(detailed, dict):
            lines.append("无详细数据")
            lines.append("")
            continue

        for ds_name, ds_data in detailed.items():
            if not isinstance(ds_data, dict):
                continue
            lines.append(f"**{ds_name}:**")
            lines.append("")
            for subject in sorted(ds_data.keys(), key=lambda x: int(x)):
                sessions = ds_data[subject]
                acc_vals = []
                for session, folds in sessions.items():
                    for fold, metrics in folds.items():
                        v = metrics.get('max_accuracy')
                        if v is not None:
                            acc_vals.append(v)
                if acc_vals:
                    avg = sum(acc_vals) / len(acc_vals)
                    lines.append(f"- Subject {subject}: {avg * 100:.2f}%")
            lines.append("")

    return "\n".join(lines)


def init_summary_dir(model_output_dir):
    summary_dir = Path(model_output_dir) / '000'
    summary_dir.mkdir(exist_ok=True)
    (summary_dir / 'all_iteration_logs').mkdir(exist_ok=True)
    (summary_dir / 'accuracy').mkdir(exist_ok=True)
    (summary_dir / 'token').mkdir(exist_ok=True)
    return summary_dir


def _token_jsonl_is_api_call(obj):
    if not isinstance(obj, dict):
        return False
    if obj.get('kind') == 'iteration_summary':
        return False
    return obj.get('attempt') is not None


def rebuild_root_token_txt(model_output_dir):
    base = Path(model_output_dir)
    summary_dir = base / '000'
    shard_root = summary_dir / 'token'
    out_path = summary_dir / 'token.jsonl'
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    shard_files = []
    if shard_root.exists():
        shard_files = sorted(shard_root.rglob('*.jsonl'))
        if not shard_files:
            shard_files = sorted(shard_root.rglob('*.txt'))

    use_flat_iter = not shard_files
    if use_flat_iter:
        for child in sorted(base.iterdir()):
            if (
                child.is_dir()
                and len(child.name) == 3
                and child.name.isdigit()
            ):
                pjl = child / 'token.jsonl'
                ptxt = child / 'token.txt'
                if pjl.is_file():
                    shard_files.append(pjl)
                elif ptxt.is_file():
                    shard_files.append(ptxt)

    root_for_rel = shard_root if not use_flat_iter else base

    merged_lines: list[str] = []
    source_stats: list[dict] = []

    for sf in shard_files:
        try:
            rel = sf.relative_to(root_for_rel)
        except ValueError:
            rel = sf.relative_to(base)
        text = sf.read_text(encoding='utf-8', errors='replace')
        n_calls = 0
        subtotal = 0
        for ln in text.splitlines():
            s = ln.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except (json.JSONDecodeError, TypeError):
                continue
            if _token_jsonl_is_api_call(obj):
                n_calls += 1
                v = obj.get('total_tokens')
                if isinstance(v, int):
                    subtotal += v
            merged_lines.append(s)

        source_stats.append({
            'path': rel.as_posix(),
            'api_calls': n_calls,
            'total_tokens_sum': subtotal,
        })

    grand_calls = sum(s['api_calls'] for s in source_stats)
    grand_total = sum(s['total_tokens_sum'] for s in source_stats)

    footer = {
        'kind': 'aggregate_footer',
        'rebuilt': ts,
        'sources': source_stats,
        'totals': {'api_calls': grand_calls, 'total_tokens_sum': grand_total},
    }
    merged_lines.append(json.dumps(footer, ensure_ascii=False, default=str))
    out_path.write_text('\n'.join(merged_lines) + '\n', encoding='utf-8')


def append_iteration_log(model_output_dir, iteration, log_content,
                         dataset_name=None, subject_id=None):
    summary_dir = Path(model_output_dir) / '000'
    logs_dir = summary_dir / 'all_iteration_logs'
    logs_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name and subject_id:
        ds_dir = logs_dir / dataset_name
        ds_dir.mkdir(exist_ok=True)
        log_file = ds_dir / f'{dataset_name}_{subject_id}.md'
    else:
        log_file = logs_dir / 'iteration_logs.md'

    header = f"### [{dataset_name}/Sub{subject_id}] 迭代 {iteration}\n\n" if dataset_name and subject_id else ""

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(header)
        f.write(log_content)
        f.write("\n\n---\n\n")


def update_accuracy_summary(model_output_dir, iteration_history, best_accuracy):
    summary_dir = Path(model_output_dir) / '000'
    summary_dir.mkdir(exist_ok=True)

    text = _build_accuracy_summary(iteration_history, best_accuracy)
    (summary_dir / 'accuracy_summary.md').write_text(text, encoding='utf-8')


def update_accuracy_summary_for_subject(model_output_dir, dataset_name, subject_id,
                                        iteration_history, best_accuracy):
    summary_dir = Path(model_output_dir) / '000'
    acc_dir = summary_dir / 'accuracy' / dataset_name
    acc_dir.mkdir(parents=True, exist_ok=True)

    iterations = []
    for entry in iteration_history:
        iterations.append({
            'iteration': entry.get('iteration', 0),
            'action': entry.get('plan', {}).get('action', '?'),
            'accuracy': entry.get('accuracy', 0.0),
            'improved': entry.get('improved', False),
        })

    subject_data = {
        'dataset': dataset_name,
        'subject_id': subject_id,
        'best_accuracy': best_accuracy,
        'iterations': iterations,
    }

    json_file = acc_dir / f'{dataset_name}_{subject_id}.json'
    json_file.write_text(
        json.dumps(subject_data, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    all_subject_data = _collect_accuracy_data(summary_dir)
    text = _build_live_accuracy_summary(all_subject_data)
    (summary_dir / 'accuracy_summary.md').write_text(text, encoding='utf-8')


def _collect_accuracy_data(summary_dir):
    acc_root = summary_dir / 'accuracy'
    all_subject_data = []
    if not acc_root.exists():
        return all_subject_data
    for jf in sorted(acc_root.rglob('*.json')):
        if jf.name == 'all_results.json':
            continue
        try:
            data = json.loads(jf.read_text(encoding='utf-8'))
            all_subject_data.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return all_subject_data


def _build_live_accuracy_summary(all_subject_data):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    lines = [
        "# AutoMI 跨受试者准确率汇总",
        "",
        f"**最后更新:** {ts}",
        f"**总受试者数:** {len(all_subject_data)}",
        "",
        "---",
        "",
    ]

    by_dataset = {}
    for s in all_subject_data:
        ds = s.get('dataset', 'unknown')
        by_dataset.setdefault(ds, []).append(s)

    for ds_name in sorted(by_dataset):
        subjects = sorted(by_dataset[ds_name],
                          key=lambda x: x.get('subject_id', 0))

        max_iter = max(
            (len(s.get('iterations', [])) for s in subjects), default=0
        )
        accs = [s.get('best_accuracy', 0.0) for s in subjects]
        avg_best = sum(accs) / len(accs) if accs else 0.0

        lines.append(f"## 数据集: {ds_name}")
        lines.append("")
        lines.append(f"**受试者数:** {len(subjects)}  |  "
                      f"**平均最佳准确率:** {avg_best * 100:.2f}%")
        lines.append("")

        iter_headers = " | ".join(f"Iter{i}" for i in range(1, max_iter + 1))
        header = f"| 受试者 | {iter_headers} | 最佳准确率 |" if max_iter else "| 受试者 | 最佳准确率 |"
        sep_parts = "|--------" + "".join("|-------" for _ in range(max_iter)) + "|-----------|"
        lines.append(header)
        lines.append(sep_parts)

        iter_col_sums = [0.0] * max_iter
        iter_col_counts = [0] * max_iter

        for s in subjects:
            sid = s.get('subject_id', '?')
            best = s.get('best_accuracy', 0.0)
            iters = s.get('iterations', [])

            iter_cells = []
            for i in range(max_iter):
                if i < len(iters):
                    acc = iters[i].get('accuracy', 0.0)
                    iter_cells.append(f"{acc * 100:.2f}%")
                    iter_col_sums[i] += acc
                    iter_col_counts[i] += 1
                else:
                    iter_cells.append("-")

            iter_str = " | ".join(iter_cells)
            lines.append(f"| Sub{sid} | {iter_str} | {best * 100:.2f}% |")

        avg_iter_cells = []
        for i in range(max_iter):
            if iter_col_counts[i] > 0:
                avg_iter_cells.append(
                    f"{iter_col_sums[i] / iter_col_counts[i] * 100:.2f}%"
                )
            else:
                avg_iter_cells.append("-")
        avg_iter_str = " | ".join(avg_iter_cells)
        lines.append(f"| 平均值 | {avg_iter_str} | {avg_best * 100:.2f}% |")
        lines.append("")

    return "\n".join(lines)


def generate_summary(model_output_dir, iteration_history, best_accuracy,
                     dataset_name=None, subject_id=None):
    model_output_dir = Path(model_output_dir)
    summary_dir = model_output_dir / '000'
    summary_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] 开始生成最终汇总: {summary_dir}")

    logs_dir = summary_dir / 'all_iteration_logs'
    logs_dir.mkdir(exist_ok=True)

    if dataset_name and subject_id:
        ds_log_dir = logs_dir / dataset_name
        ds_log_dir.mkdir(exist_ok=True)
        log_file = ds_log_dir / f'{dataset_name}_{subject_id}.md'
    else:
        log_file = logs_dir / 'iteration_logs.md'

    merged_lines = []
    for iter_idx in range(1, len(iteration_history) + 1):
        iter_folder = model_output_dir / f'{iter_idx:03d}'
        src_log = iter_folder / 'iteration_log.md'
        if src_log.exists():
            content = src_log.read_text(encoding='utf-8')
            if dataset_name and subject_id:
                merged_lines.append(
                    f"### [{dataset_name}/Sub{subject_id}] 迭代 {iter_idx}"
                )
                merged_lines.append("")
            merged_lines.append(content)
            merged_lines.append("\n---\n")

    log_file.write_text("\n".join(merged_lines), encoding='utf-8')

    if dataset_name and subject_id:
        update_accuracy_summary_for_subject(
            model_output_dir, dataset_name, subject_id,
            iteration_history, best_accuracy,
        )
    else:
        accuracy_text = _build_accuracy_summary(iteration_history, best_accuracy)
        (summary_dir / 'accuracy_summary.md').write_text(
            accuracy_text, encoding='utf-8'
        )

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 最终汇总生成完成")


def generate_aggregated_summary(base_output_dir, all_results):
    base = Path(base_output_dir)
    summary_dir = base / '000'
    summary_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] 开始生成跨受试者聚合汇总: {summary_dir}")

    successful = [r for r in all_results if r.get('success')]
    failed = [r for r in all_results if not r.get('success')]

    all_subject_data = _collect_accuracy_data(summary_dir)

    if all_subject_data:
        summary_text = _build_live_accuracy_summary(all_subject_data)
    else:
        summary_text = _build_aggregated_summary_from_results(
            ts, all_results, successful, failed,
        )

    if failed:
        fail_lines = ["", "---", "", "## 失败的受试者", ""]
        for r in failed:
            err_brief = (r.get('error') or '未知错误')[:200]
            fail_lines.append(
                f"- **{r['dataset']}/Sub{r['subject_id']}**: {err_brief}"
            )
        fail_lines.append("")
        summary_text += "\n" + "\n".join(fail_lines)

    (summary_dir / 'accuracy_summary.md').write_text(summary_text, encoding='utf-8')

    logs_dir = summary_dir / 'all_iteration_logs'
    logs_dir.mkdir(exist_ok=True)

    max_iters = max(
        (len(r.get('iteration_history', [])) for r in successful),
        default=0,
    )

    per_subject_logs = {}
    for iter_idx in range(1, max_iters + 1):
        iter_dir = base / f'{iter_idx:03d}'
        if not iter_dir.exists():
            continue
        for ds_dir in sorted(iter_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            for subj_dir in sorted(ds_dir.iterdir()):
                if not subj_dir.is_dir():
                    continue
                log_file = subj_dir / 'iteration_log.md'
                if log_file.exists():
                    content = log_file.read_text(encoding='utf-8')
                    key = (ds_dir.name, subj_dir.name)
                    per_subject_logs.setdefault(key, []).append(
                        f"### [{ds_dir.name}/Sub{subj_dir.name}] 迭代 {iter_idx}\n\n"
                        f"{content}\n\n---\n"
                    )

    for (ds_name, subj_name), log_parts in per_subject_logs.items():
        ds_log_dir = logs_dir / ds_name
        ds_log_dir.mkdir(exist_ok=True)
        out_file = ds_log_dir / f'{ds_name}_{subj_name}.md'
        out_file.write_text("\n".join(log_parts), encoding='utf-8')

    acc_dir = summary_dir / 'accuracy'
    acc_dir.mkdir(exist_ok=True)

    results_json = []
    for r in all_results:
        results_json.append({
            'dataset': r['dataset'],
            'subject_id': r['subject_id'],
            'best_accuracy': r['best_accuracy'],
            'success': r['success'],
            'error': r.get('error'),
            'iterations': len(r.get('iteration_history', [])),
        })
    (acc_dir / 'all_results.json').write_text(
        json.dumps(results_json, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    rebuild_root_token_txt(base)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 聚合汇总生成完成")

    summary_file = summary_dir / 'accuracy_summary.md'
    return str(summary_file) if summary_file.exists() else None


def _build_aggregated_summary_from_results(ts, all_results, successful, failed):
    by_dataset = {}
    for r in successful:
        by_dataset.setdefault(r['dataset'], []).append(r)

    lines = [
        "# AutoMI 跨受试者聚合汇总",
        "",
        f"**生成时间:** {ts}",
        f"**总受试者数:** {len(all_results)}",
        f"**成功:** {len(successful)}  |  **失败:** {len(failed)}",
        "",
        "---",
        "",
    ]

    for ds_name, ds_results in sorted(by_dataset.items()):
        accs = [r['best_accuracy'] for r in ds_results]
        avg_acc = sum(accs) / len(accs) if accs else 0.0
        lines.append(f"## 数据集: {ds_name}")
        lines.append("")
        lines.append(f"**受试者数:** {len(ds_results)}  |  "
                      f"**平均最佳准确率:** {avg_acc * 100:.2f}%")
        lines.append("")
        lines.append("| 受试者 | 最佳准确率 |")
        lines.append("|--------|-----------|")
        for r in sorted(ds_results, key=lambda x: x['subject_id']):
            lines.append(
                f"| Sub{r['subject_id']} | {r['best_accuracy'] * 100:.2f}% |"
            )
        avg_line = f"| 平均值 | {avg_acc * 100:.2f}% |"
        lines.append(avg_line)
        lines.append("")

    return "\n".join(lines)
