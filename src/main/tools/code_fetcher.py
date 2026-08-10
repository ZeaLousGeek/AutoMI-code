
import importlib.util
import base64
from pathlib import Path
from datetime import datetime

DEFAULT_REPO_URL = 'https://github.com/ZeaLousGeek/Awesome-MI-EEG-Classification'
DEFAULT_REPO_API = 'https://api.github.com/repos/ZeaLousGeek/Awesome-MI-EEG-Classification'

PROJECT_ROOT = Path(__file__).parent.parent.parent


def _parse_repo_api_url(repo_url):
    repo_url = repo_url.rstrip('/')
    if 'api.github.com' in repo_url:
        return repo_url
    parts = repo_url.replace('https://github.com/', '').split('/')
    if len(parts) >= 2:
        owner, repo = parts[0], parts[1]
        return f'https://api.github.com/repos/{owner}/{repo}'
    return None


def _get_requests():
    if importlib.util.find_spec('requests') is None:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] requests 未安装，无法进行 GitHub 操作")
        return None
    import requests
    return requests


def fetch_code_from_repo(repo_url=None, file_path='', branch='main'):
    requests = _get_requests()
    if requests is None:
        return None

    if repo_url is None:
        repo_url = DEFAULT_REPO_URL

    api_base = _parse_repo_api_url(repo_url)
    if api_base is None:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 无法解析仓库 URL: {repo_url}")
        return None

    url = f"{api_base}/contents/{file_path}?ref={branch}"
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] 从 GitHub 拉取文件: {file_path}")

    headers = {'Accept': 'application/vnd.github.v3+json'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"[{ts}] 拉取失败，状态码: {response.status_code}")
            return None

        data = response.json()
        if data.get('encoding') == 'base64':
            content = base64.b64decode(data['content']).decode('utf-8')
        else:
            content = data.get('content', '')

        print(f"[{ts}] 文件拉取成功: {file_path} ({len(content)} 字符)")
        return content
    except Exception as e:
        print(f"[{ts}] 拉取异常: {e}")
        return None


def list_repo_contents(repo_url=None, directory='', branch='main'):
    requests = _get_requests()
    if requests is None:
        return []

    if repo_url is None:
        repo_url = DEFAULT_REPO_URL

    api_base = _parse_repo_api_url(repo_url)
    if api_base is None:
        return []

    url = f"{api_base}/contents/{directory}?ref={branch}"
    headers = {'Accept': 'application/vnd.github.v3+json'}

    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return []

        items = response.json()
        if not isinstance(items, list):
            return []

        return [
            {
                'name': item.get('name', ''),
                'path': item.get('path', ''),
                'type': item.get('type', ''),
                'size': item.get('size', 0),
            }
            for item in items
        ]
    except Exception as e:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] 列出目录内容失败: {e}")
        return []


def list_repo_models(repo_url=None):
    if repo_url is None:
        repo_url = DEFAULT_REPO_URL

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] 列出仓库中的模型文件...")

    from src.main.tools.model_fetcher import fetch_models_from_awesome_repo
    models = fetch_models_from_awesome_repo()

    contents = list_repo_contents(repo_url, 'src/models')
    local_models = []
    for item in contents:
        if item['type'] == 'dir':
            local_models.append({
                'name': item['name'],
                'path': item['path'],
                'source': 'repo_directory',
            })

    print(f"[{ts}] 找到 {len(models)} 个 Awesome 列表模型, {len(local_models)} 个仓库目录模型")
    return {'awesome_models': models, 'repo_models': local_models}


def download_model_from_repo(repo_url=None, model_path='', save_dir=None):
    if save_dir is None:
        save_dir = PROJECT_ROOT / 'src' / 'models'

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    content = fetch_code_from_repo(repo_url, model_path)
    if content is None:
        return None

    filename = Path(model_path).name
    save_path = save_dir / filename
    save_path.write_text(content, encoding='utf-8')

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] 模型已保存到: {save_path}")
    return save_path
