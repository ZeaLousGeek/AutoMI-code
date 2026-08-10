
import os

LLM_PROVIDER = os.environ.get("AUTOMI_LLM_PROVIDER", "qwen")

QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

CLAUDE_API_KEY = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
CLAUDE_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

CLAUDE_DEFAULT_MODEL = "claude-sonnet-4-6"

CLAUDE_MODELS = [
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
]

QWEN_MODELS = [
    "qwen3.5-plus",
    "qwen3.5-flash",
    "qwen-flash",
    "qwen-plus",
    "qwen3-max",
    "deepseek-v3.2",
    "qwen3-235b-a22b",
    "qwen3-30b-a3b",
    "qwen3-32b",
    "qwen3-14b",
    "qwen3-8b",
    "qwen3-4b",
    "qwen3-1.7b",
    "qwen3-0.6b",
    "qwen3-next-80b-a3b-instruct",
    "qwen3-next-80b-a3b-thinking",
    "qwen3-30b-a3b-instruct-2507",
    "qwen3-30b-a3b-thinking-2507",
    "qwen3-235b-a22b-thinking-2507",
    "qwen3-235b-a22b-instruct-2507",
    "kimi-k2-thinking",
    "kimi-k2.5",
    "glm-5",
    "glm-4.7",
    "MiniMax-M2.5",
    "abab6.5s-chat"
]

AVAILABLE_MODELS = QWEN_MODELS + CLAUDE_MODELS

if LLM_PROVIDER == "claude":
    DEFAULT_MODEL = CLAUDE_DEFAULT_MODEL
else:
    DEFAULT_MODEL = "deepseek-v3.2"

DEFAULT_PARAMS = {
    "temperature": 0.,
    "max_tokens": 2048,
    "top_p": 0.9,
}

TASK_TEMPERATURES = {
    "builder": 0,
    "check": 0,
    "filling": 0,
}

DETERMINISTIC_MODE = False
DETERMINISTIC_SEED = 20262026
ENABLE_THINKING = True

