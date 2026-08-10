
import importlib

_MODULE_MAP = {
    'planning_agent': 'src.prompts.planning_agent_prompts',
    'execution_agent': 'src.prompts.execution_agent_prompts',
    'output_agent': 'src.prompts.output_agent_prompts',
}


def load_prompt(prompt_name):
    module_path = _MODULE_MAP.get(prompt_name)
    if module_path is None:
        raise ValueError(f"未知的提示词名称: {prompt_name}")
    mod = importlib.import_module(module_path)
    return mod.SYSTEM_PROMPT


def load_prompt_template(prompt_name, template_name):
    module_path = _MODULE_MAP.get(prompt_name)
    if module_path is None:
        raise ValueError(f"未知的提示词名称: {prompt_name}")
    mod = importlib.import_module(module_path)
    template = getattr(mod, template_name, None)
    if template is None:
        raise AttributeError(
            f"提示词模块 {prompt_name} 中不存在模板: {template_name}"
        )
    return template
