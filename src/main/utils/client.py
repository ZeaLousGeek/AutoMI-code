
from typing import Optional

from .config import DEFAULT_MODEL, LLM_PROVIDER, CLAUDE_MODELS
from .qwen_client import qwen_chat
from .claude_client import claude_chat


def chat(
    messages: list,
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    task: str = None,
    seed: int = None,
    enable_thinking: bool = None,
    label: Optional[str] = None,
):
    if model is None:
        model = DEFAULT_MODEL

    if model in CLAUDE_MODELS:
        provider = "claude"
    else:
        provider = LLM_PROVIDER

    if provider == "claude":
        return claude_chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            task=task,
            seed=seed,
            enable_thinking=enable_thinking,
            label=label,
        )
    else:
        return qwen_chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            task=task,
            seed=seed,
            enable_thinking=enable_thinking,
            label=label,
        )


def simple_chat(
    prompt: str,
    model: str = None,
    label: Optional[str] = None,
):
    messages = [{"role": "user", "content": prompt}]
    return chat(messages, model=model, label=label)


if __name__ == "__main__":
    response = simple_chat("你好，请介绍一下你自己")
    print(response)

