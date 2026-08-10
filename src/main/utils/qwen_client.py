
import json
import os
import time
from datetime import datetime
from typing import Optional
from openai import OpenAI, RateLimitError, APIError

from src.main.utils.token_usage_context import get_token_log_targets
from .config import (
    QWEN_API_KEY, QWEN_BASE_URL, DEFAULT_PARAMS, TASK_TEMPERATURES,
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


def get_qwen_client():
    if not QWEN_API_KEY:
        raise RuntimeError("QWEN_API_KEY is not set; export QWEN_API_KEY before running AutoMI")
    return OpenAI(
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
    )


def qwen_chat(
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
    client = get_qwen_client()

    request_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if DETERMINISTIC_MODE:
        request_params["seed"] = seed if seed is not None else DETERMINISTIC_SEED
        effective_enable_thinking = enable_thinking if enable_thinking is not None else False
    else:
        if seed is not None:
            request_params["seed"] = seed
        effective_enable_thinking = enable_thinking if enable_thinking is not None else ENABLE_THINKING

    if effective_enable_thinking and model in ["qwen3-max", "deepseek-v3.2"]:
        request_params["extra_body"] = {"enable_thinking": True}

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            t_start = datetime.now().isoformat(timespec="seconds")
            response = client.chat.completions.create(**request_params)
            t_end = datetime.now().isoformat(timespec="seconds")
            usage = getattr(response, "usage", None)
            choice = response.choices[0]
            assistant_msg = choice.message
            completion_content = getattr(assistant_msg, "content", None)
            rec = {
                "start": t_start,
                "end": t_end,
                "model": model,
                "attempt": attempt + 1,
                "request_messages": messages,
                "completion_content": completion_content,
                "finish_reason": getattr(choice, "finish_reason", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None)
                if usage is not None
                else None,
                "completion_tokens": getattr(usage, "completion_tokens", None)
                if usage is not None
                else None,
                "total_tokens": getattr(usage, "total_tokens", None)
                if usage is not None
                else None,
            }
            tool_calls = getattr(assistant_msg, "tool_calls", None)
            if tool_calls:
                try:
                    rec["tool_calls"] = json.loads(
                        json.dumps(tool_calls, default=str)
                    )
                except (TypeError, ValueError):
                    rec["tool_calls"] = str(tool_calls)
            refusal = getattr(assistant_msg, "refusal", None)
            if refusal:
                rec["refusal"] = refusal
            if label:
                rec["label"] = label
            line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
            _append_token_usage_line(line)

            log_path = os.environ.get("AUTOMI_LOG_TOKEN_USAGE")
            if log_path:
                if usage is not None:
                    legacy = {
                        "model": model,
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(
                            usage, "completion_tokens", None
                        ),
                        "total_tokens": getattr(usage, "total_tokens", None),
                    }
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(legacy, ensure_ascii=False) + "\n")
            return completion_content
        except RateLimitError as e:
            last_error = e
            error_msg = str(e)
            if "insufficient_quota" in error_msg:
                raise RuntimeError(
                    f"❌ API 配额已用完！请登录阿里云控制台充值。\n"
                    f"详情: https://help.aliyun.com/zh/model-studio/error-code#token-limit\n"
                    f"原始错误: {e}"
                ) from e
            if attempt < MAX_RETRIES - 1:
                print(f"⚠️ API 限流，{RETRY_DELAY}秒后重试 ({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(RETRY_DELAY)
        except APIError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                print(f"⚠️ API 错误，{RETRY_DELAY}秒后重试 ({attempt + 1}/{MAX_RETRIES}): {e}")
                time.sleep(RETRY_DELAY)

    raise last_error
