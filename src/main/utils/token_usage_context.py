
from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TokenLogTargets:

    iter_token_path: Path
    shard_token_path: Path | None


_targets: ContextVar[TokenLogTargets | None] = ContextVar(
    "automi_token_log_targets", default=None
)


def get_token_log_targets() -> TokenLogTargets | None:
    return _targets.get()


def set_token_log_targets(
    iter_folder: Path,
    model_output_dir: Path,
    dataset_name: str | None,
    subject_id: int | None,
) -> Any:
    iter_path = Path(iter_folder) / "token.jsonl"
    shard: Path | None = None
    if dataset_name and subject_id is not None:
        root = Path(model_output_dir) / "000" / "token" / dataset_name
        root.mkdir(parents=True, exist_ok=True)
        shard = root / f"{dataset_name}_{subject_id}.jsonl"
    return _targets.set(
        TokenLogTargets(iter_token_path=iter_path, shard_token_path=shard)
    )


def reset_token_log_targets(token: Any) -> None:
    _targets.reset(token)


def finalize_iteration_token_file(iter_folder: Path, iteration: int) -> None:
    folder = Path(iter_folder)
    path_jsonl = folder / "token.jsonl"
    path_txt = folder / "token.txt"
    read_from = path_jsonl if path_jsonl.exists() else (path_txt if path_txt.exists() else None)
    api_lines: list[str] = []
    if read_from is not None:
        for raw in read_from.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("kind") == "iteration_summary":
                continue
            api_lines.append(line)

    by_label: dict[str, dict[str, int]] = {}
    totals = {
        "api_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for line in api_lines:
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        lbl = o.get("label") or "_unlabeled"
        totals["api_calls"] += 1
        sub = by_label.setdefault(
            lbl,
            {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )
        sub["calls"] += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            v = o.get(key)
            if isinstance(v, int):
                sub[key] += v
                totals[key] += v

    summary = {
        "kind": "iteration_summary",
        "iteration": iteration,
        "by_label": by_label,
        "totals": totals,
    }
    path_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_lines = api_lines + [json.dumps(summary, ensure_ascii=False, default=str)]
    path_jsonl.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
