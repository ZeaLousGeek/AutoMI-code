
import importlib.util
from datetime import datetime


AWESOME_REPO_URL = (
    "https://raw.githubusercontent.com/ZeaLousGeek/"
    "Awesome-MI-EEG-Classification/main/README.md"
)


def fetch_models_from_awesome_repo():
    requests_available = importlib.util.find_spec('requests') is not None
    if not requests_available:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"requests 未安装，跳过模型获取")
        return []

    import requests

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"从 Awesome-MI-EEG-Classification 获取模型列表...")

    response = requests.get(AWESOME_REPO_URL, timeout=15)
    if response.status_code != 200:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"获取失败，状态码: {response.status_code}")
        return []

    models = _parse_readme_for_models(response.text)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"从仓库中提取到 {len(models)} 个模型")
    return models


def _parse_readme_for_models(readme_text):
    models = []
    lines = readme_text.split('\n')

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith('- [') and not stripped.startswith('* ['):
            continue

        name_start = stripped.find('[') + 1
        name_end = stripped.find(']', name_start)
        if name_end == -1:
            continue
        name = stripped[name_start:name_end]

        url_start = stripped.find('(', name_end) + 1
        url_end = stripped.find(')', url_start)
        url = stripped[url_start:url_end] if url_end > url_start else ''

        desc = stripped[url_end + 1:].strip(' -:') if url_end > 0 else ''

        if name and ('EEG' in name.upper() or 'EEG' in desc.upper()
                     or 'BCI' in name.upper() or 'BCI' in desc.upper()
                     or 'motor' in desc.lower() or 'imagery' in desc.lower()
                     or url.startswith('http')):
            models.append({
                'name': name,
                'url': url,
                'description': desc[:200],
            })

    return models
