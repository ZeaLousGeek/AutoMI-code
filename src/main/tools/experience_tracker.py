
import json
from datetime import datetime


MAX_EXPERIENCE_CHARS = 3000


class StructureGroup:

    def __init__(self, group_id, start_iteration, structure_desc=""):
        self.group_id = group_id
        self.start_iteration = start_iteration
        self.structure_desc = structure_desc
        self.iterations = []
        self.compressed_summary = None

    def add_iteration(self, iteration, action, config_overrides, accuracy,
                      improved, notes="", reasoning="",
                      llm_evaluation=None, no_improve_analysis=None):
        self.iterations.append({
            'iteration': iteration,
            'action': action,
            'config_overrides': config_overrides,
            'accuracy': accuracy,
            'improved': improved,
            'notes': notes,
            'reasoning': reasoning,
            'llm_evaluation': llm_evaluation,
            'no_improve_analysis': no_improve_analysis,
        })

    def to_text(self, detail_level="summary"):
        if self.compressed_summary:
            return (
                f"【结构组 {self.group_id}（迭代 {self.start_iteration} 起，"
                f"已压缩）】\n{self.compressed_summary}\n"
            )

        lines = [
            f"【结构组 {self.group_id}（迭代 {self.start_iteration} 起）】",
            f"结构描述: {self.structure_desc or '初始结构'}",
        ]
        for it in self.iterations:
            overrides_str = ""
            if it['config_overrides']:
                overrides_str = json.dumps(
                    it['config_overrides'], ensure_ascii=False, default=str
                )
                if len(overrides_str) > 200:
                    overrides_str = overrides_str[:200] + "..."
            improved_str = "改进" if it['improved'] else "未改进"
            line = (
                f"  迭代{it['iteration']}: {it['action']}, "
                f"准确率={it['accuracy']:.4f}, {improved_str}"
            )
            if overrides_str:
                line += f", 参数={overrides_str}"
            if it.get('notes'):
                line += f", {it['notes']}"
            lines.append(line)

            if detail_level == "full":
                if it.get('reasoning'):
                    lines.append(f"    推理: {it['reasoning']}")
                if it.get('llm_evaluation'):
                    lines.append(f"    LLM评估: {it['llm_evaluation']}")
                if it.get('no_improve_analysis'):
                    lines.append(f"    未改进分析: {it['no_improve_analysis']}")

        return "\n".join(lines) + "\n"

    @property
    def total_iterations(self):
        return len(self.iterations)

    @property
    def char_count(self):
        return len(self.to_text())


class ExperienceTracker:

    def __init__(self, max_chars=MAX_EXPERIENCE_CHARS):
        self.groups = []
        self.max_chars = max_chars
        self._current_group = None
        self._group_counter = 0

    def _start_new_group(self, iteration, structure_desc=""):
        self._group_counter += 1
        group = StructureGroup(self._group_counter, iteration, structure_desc)
        self.groups.append(group)
        self._current_group = group

    def add_iteration(self, iteration, action, config_overrides, accuracy,
                      improved, model_code_changed=False, notes="",
                      reasoning="", llm_evaluation=None,
                      no_improve_analysis=None):
        is_structure_change = (
            action == 'structure_update' or model_code_changed
        )

        if self._current_group is None or is_structure_change:
            desc = f"结构更新于迭代{iteration}" if is_structure_change else ""
            self._start_new_group(iteration, desc)

        self._current_group.add_iteration(
            iteration, action, config_overrides, accuracy, improved, notes,
            reasoning=reasoning,
            llm_evaluation=llm_evaluation,
            no_improve_analysis=no_improve_analysis,
        )

    def _count_total_iterations(self):
        return sum(g.total_iterations for g in self.groups)

    def build_planning_context(self, keep_recent=5):
        if not self.groups:
            return "无历史迭代经验"

        total_iters = self._count_total_iterations()
        old_count = max(0, total_iters - keep_recent)

        if old_count > 0:
            self._compress_old_iterations(old_count)

        parts = []
        seen = 0
        for group in self.groups:
            if group.compressed_summary:
                parts.append(group.to_text())
                continue

            group_old = max(0, min(len(group.iterations), old_count - seen))
            seen += group_old

            if group_old >= len(group.iterations):
                parts.append(group.to_text(detail_level="summary"))
            elif group_old > 0:
                old_lines = [
                    f"【结构组 {group.group_id}（迭代 {group.start_iteration} 起）】",
                    f"结构描述: {group.structure_desc or '初始结构'}",
                ]
                for it in group.iterations[:group_old]:
                    improved_str = "改进" if it['improved'] else "未改进"
                    old_lines.append(
                        f"  迭代{it['iteration']}: {it['action']}, "
                        f"准确率={it['accuracy']:.4f}, {improved_str}"
                    )
                old_lines.append("  --- 以下为最近详细记录 ---")
                for it in group.iterations[group_old:]:
                    old_lines.extend(self._format_iteration_full(it))
                parts.append("\n".join(old_lines) + "\n")
            else:
                parts.append(group.to_text(detail_level="full"))

        text = "\n".join(parts)

        if len(text) > self.max_chars:
            self._compress_old_groups()
            parts = [g.to_text(detail_level="full") for g in self.groups]
            text = "\n".join(parts)

        return text

    @staticmethod
    def _format_iteration_full(it):
        overrides_str = ""
        if it['config_overrides']:
            overrides_str = json.dumps(
                it['config_overrides'], ensure_ascii=False, default=str
            )
            if len(overrides_str) > 200:
                overrides_str = overrides_str[:200] + "..."
        improved_str = "改进" if it['improved'] else "未改进"
        lines = [
            f"  迭代{it['iteration']}: {it['action']}, "
            f"准确率={it['accuracy']:.4f}, {improved_str}"
            + (f", 参数={overrides_str}" if overrides_str else "")
        ]
        if it.get('reasoning'):
            lines.append(f"    推理: {it['reasoning']}")
        if it.get('llm_evaluation'):
            lines.append(f"    LLM评估: {it['llm_evaluation']}")
        if it.get('no_improve_analysis'):
            lines.append(f"    未改进分析: {it['no_improve_analysis']}")
        return lines

    def _compress_old_iterations(self, old_count):
        seen = 0
        for group in self.groups:
            if group.compressed_summary:
                continue
            seen += len(group.iterations)
            if seen <= old_count and group is not self.groups[-1]:
                if group.compressed_summary is None:
                    self._compress_single_group(group)

    def _compress_single_group(self, group):
        pass

    def _compress_old_groups(self):
        groups_to_compress = [
            g for g in self.groups[:-1] if g.compressed_summary is None
        ]
        if not groups_to_compress:
            return

        old_text_parts = [g.to_text(detail_level="summary")
                          for g in groups_to_compress]
        old_text = "\n".join(old_text_parts)

        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] 迭代经验过长（{len(old_text)}字符），正在 LLM 压缩旧组...")

        try:
            from src.main.utils.client import chat
            prompt = (
                "以下是运动想象脑电分类模型自动迭代优化的历史记录，"
                "包含多个结构组的参数优化尝试及结果。\n\n"
                f"{old_text}\n\n"
                "请将以上内容压缩为简洁摘要，保留关键信息：\n"
                "1. 每个结构组尝试了哪些参数/结构变更\n"
                "2. 各次尝试的准确率变化趋势\n"
                "3. 哪些改动有效、哪些无效\n"
                "总字数控制在300字以内，用中文回答。"
            )
            messages = [{"role": "user", "content": prompt}]
            summary = chat(
                messages,
                max_tokens=512,
                label="tools.experience_tracker.compress_groups",
            )
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"LLM 压缩失败: {e}，保留原文")
            return

        if summary:
            for g in groups_to_compress:
                g.compressed_summary = "(已合并压缩，见下方统一摘要)"
            groups_to_compress[0].compressed_summary = summary
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{ts}] 旧组压缩完成，摘要长度: {len(summary)}字符")
