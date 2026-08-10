
import json
import os
import time
from datetime import datetime
from typing import Optional

try:
    from anthropic import Anthropic, RateLimitError, APIError
except ImportError:
    raise ImportError(
        "anthropic 包未安装。请运行: pip install anthropic>=0.39.0"
    )

from src.main.utils.token_usage_context import get_token_log_targets
from .config import (
    CLAUDE_API_KEY, CLAUDE_BASE_URL, DEFAULT_PARAMS, TASK_TEMPERATURES,
    DETERMINISTIC_MODE, DETERMINISTIC_SEED, ENABLE_THINKING
)

MAX_RETRIES = 3
RETRY_DELAY = 5


def _append_token_usage_line(line: str) -> None:
    targets = get_token_log_targets()
    if targets is None:
        return
    if not line.endswith("\n"):
        line = line + "\n"
    for path in (targets.iter_token_path, targets.shard_token_path):
        if path is None:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()


def get_claude_client():
    return Anthropic(
        api_key=CLAUDE_API_KEY,
        base_url=CLAUDE_BASE_URL,
    )


def _convert_messages_to_claude_format(messages: list) -> tuple[Optional[str], list]:
    system_prompt = None
    claude_messages = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            if system_prompt is None:
                system_prompt = content
            else:
                system_prompt += "\n\n" + content
        elif role in ("user", "assistant"):
            claude_messages.append({"role": role, "content": content})

    return system_prompt, claude_messages


def claude_chat(
    messages: list,
    model: str,
    temperature: float = None,
    max_tokens: int = DEFAULT_PARAMS["max_tokens"],
    task: str = None,
    seed: int = None,
    enable_thinking: bool = None,
    label: Optional[str] = None,
):
    if task and task in TASK_TEMPERATURES:
        temperature = TASK_TEMPERATURES[task]
    elif temperature is None:
        temperature = DEFAULT_PARAMS["temperature"]

    if max_tokens is None:
        max_tokens = DEFAULT_PARAMS["max_tokens"]

    client = get_claude_client()

    system_prompt, claude_messages = _convert_messages_to_claude_format(messages)

    request_params = {
        "model": model,
        "messages": claude_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if system_prompt:
        request_params["system"] = system_prompt

    if DETERMINISTIC_MODE:
        effective_enable_thinking = enable_thinking if enable_thinking is not None else False
    else:
        effective_enable_thinking = enable_thinking if enable_thinking is not None else ENABLE_THINKING

    if effective_enable_thinking:
        request_params["thinking"] = {
            "type": "enabled",
            "budget_tokens": 10000
        }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            t_start = datetime.now().isoformat(timespec="seconds")
            response = client.messages.create(**request_params)
            t_end = datetime.now().isoformat(timespec="seconds")

            completion_content = ""
            for block in response.content:
                if block.type == "text":
                    completion_content += block.text

            usage = response.usage
            prompt_tokens = usage.input_tokens if hasattr(usage, "input_tokens") else None
            completion_tokens = usage.output_tokens if hasattr(usage, "output_tokens") else None
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0) if prompt_tokens and completion_tokens else None

            rec = {
                "start": t_start,
                "end": t_end,
                "model": model,
                "attempt": attempt + 1,
                "request_messages": messages,
                "completion_content": completion_content,
                "finish_reason": response.stop_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

            if label:
                rec["label"] = label

            line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
            _append_token_usage_line(line)

            log_path = os.environ.get("AUTOMI_LOG_TOKEN_USAGE")
            if log_path:
                legacy = {
                    "model": model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(legacy, ensure_ascii=False) + "\n")

            return completion_content

        except RateLimitError as e:
            last_error = e
            error_msg = str(e)
            if "insufficient_quota" in error_msg or "credit" in error_msg.lower():
                raise RuntimeError(
                    f"❌ Claude API 配额已用完！请检查账户余额。\n"
                    f"原始错误: {e}"
                ) from e
            if attempt < MAX_RETRIES - 1:
                print(f"⚠️ Claude API 限流，{RETRY_DELAY}秒后重试 ({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(RETRY_DELAY)

        except APIError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                print(f"⚠️ Claude API 错误，{RETRY_DELAY}秒后重试 ({attempt + 1}/{MAX_RETRIES}): {e}")
                time.sleep(RETRY_DELAY)

    raise last_error
