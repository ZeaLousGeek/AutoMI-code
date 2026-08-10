
import re
import urllib.parse
import xml.etree.ElementTree as ET
import importlib.util
from datetime import datetime

_paper_cache = {}


def _extract_arxiv_id(link):
    m = re.search(r'(\d{4}\.\d{4,5})', link or '')
    return m.group(1) if m else None


def get_searched_titles():
    return [v['name'] for v in _paper_cache.values()]


def get_paper_cache():
    return dict(_paper_cache)


def search_papers(query, max_results=5):
    requests_available = importlib.util.find_spec('requests') is not None
    if not requests_available:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"requests 未安装，跳过论文搜索")
        return []

    import requests

    encoded_query = urllib.parse.quote(query)
    url = (
        f"http://export.arxiv.org/api/query"
        f"?search_query=all:{encoded_query}"
        f"&start=0&max_results={max_results}"
        f"&sortBy=relevance&sortOrder=descending"
    )

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"搜索arXiv论文: {query}")

    try:
        response = requests.get(url, timeout=15)
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"arXiv请求失败: {e}")
        return []

    if response.status_code != 200:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"arXiv搜索失败，状态码: {response.status_code}")
        return []

    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    root = ET.fromstring(response.text)
    entries = root.findall('atom:entry', ns)

    results = []
    for entry in entries:
        title_el = entry.find('atom:title', ns)
        summary_el = entry.find('atom:summary', ns)
        link_el = entry.find('atom:id', ns)

        author_els = entry.findall('atom:author/atom:name', ns)
        authors = [a.text.strip() for a in author_els] if author_els else []

        link = link_el.text.strip() if link_el is not None else ''
        arxiv_id = _extract_arxiv_id(link)

        results.append({
            'title': title_el.text.strip().replace('\n', ' ') if title_el is not None else '',
            'summary': summary_el.text.strip().replace('\n', ' ')[:500] if summary_el is not None else '',
            'authors': authors,
            'link': link,
            'arxiv_id': arxiv_id,
        })

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"arXiv搜索完成，找到 {len(results)} 篇论文")
    return results


def search_github(query, max_results=5):
    requests_available = importlib.util.find_spec('requests') is not None
    if not requests_available:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"requests 未安装，跳过GitHub搜索")
        return []

    import requests

    encoded_query = urllib.parse.quote(query)
    url = (
        f"https://api.github.com/search/repositories"
        f"?q={encoded_query}&sort=stars&order=desc&per_page={max_results}"
    )

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"搜索GitHub仓库: {query}")

    headers = {'Accept': 'application/vnd.github.v3+json'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"GitHub请求失败: {e}")
        return []

    if response.status_code != 200:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"GitHub搜索失败，状态码: {response.status_code}")
        return []

    data = response.json()
    items = data.get('items', [])

    results = []
    for item in items:
        results.append({
            'name': item.get('full_name', ''),
            'description': item.get('description', '') or '',
            'url': item.get('html_url', ''),
            'stars': item.get('stargazers_count', 0),
        })

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"GitHub搜索完成，找到 {len(results)} 个仓库")
    return results


def search_web(query, max_results=5):
    requests_available = importlib.util.find_spec('requests') is not None
    if not requests_available:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"requests 未安装，跳过网页搜索")
        return []

    import requests

    encoded_query = urllib.parse.quote(query)
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={encoded_query}&limit={max_results}"
        f"&fields=title,abstract,url,year"
    )

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"搜索Semantic Scholar: {query}")

    try:
        response = requests.get(url, timeout=15)
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"Semantic Scholar请求失败: {e}")
        return []

    if response.status_code != 200:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"Semantic Scholar搜索失败，状态码: {response.status_code}")
        return []

    data = response.json()
    papers = data.get('data', [])

    results = []
    for paper in papers:
        abstract = paper.get('abstract', '') or ''
        results.append({
            'title': paper.get('title', ''),
            'abstract': abstract[:300],
            'url': paper.get('url', ''),
            'year': paper.get('year', ''),
        })

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"Semantic Scholar搜索完成，找到 {len(results)} 篇论文")
    return results


def search_and_extract_suggestions(query, model_name="", max_papers=5,
                                    action_name=""):
    papers = search_papers(query, max_results=max_papers)
    if not papers:
        return "", [], {}

    new_papers = []
    cached_suggestions = []
    retrieve_dict = {}
    idx = 1

    for paper in papers:
        arxiv_id = paper.get('arxiv_id')
        title = paper.get('title', '未知标题')

        if arxiv_id and arxiv_id in _paper_cache:
            cached = _paper_cache[arxiv_id]
            cached_suggestions.append(cached['content'])
            retrieve_dict[str(idx)] = {
                'name': cached['name'],
                'number': cached['number'],
                'content': cached['content'],
            }
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{ts}] 论文缓存命中: {title} ({arxiv_id})")
        else:
            new_papers.append(paper)
            retrieve_dict[str(idx)] = {
                'name': title,
                'number': arxiv_id or '',
                'content': '',
            }
        idx += 1

    new_suggestions = ""
    if new_papers:
        paper_summaries = []
        for i, paper in enumerate(new_papers, 1):
            paper_summaries.append(
                f"{i}. {paper.get('title', '未知标题')}\n"
                f"   摘要: {paper.get('summary', '无摘要')}"
            )
        papers_text = "\n\n".join(paper_summaries)

        from src.main.utils.client import chat as llm_chat

        action_guidance = ""
        if action_name == 'parameter_evolution':
            action_guidance = (
                "当前改进方向为参数优化（parameter_evolution），请重点提炼与以下内容相关的建议：\n"
                "- 超参数调优（学习率、批大小、权重衰减等）\n"
                "- 训练策略（学习率调度、优化器选择、正则化方法等）\n"
                "- 数据增强策略\n"
                "请忽略与模型结构改动和迁移学习相关的内容。\n\n"
            )
        elif action_name == 'structure_update':
            action_guidance = (
                "当前改进方向为结构更新（structure_update），请重点提炼与以下内容相关的建议：\n"
                "- 模型架构改进（注意力机制、卷积设计、残差连接等）\n"
                "- 特征提取方法（多尺度融合、空间滤波、时频分析等）\n"
                "- 轻量化网络设计\n"
                "请忽略迁移学习（transfer learning）和跨受试者适应相关的内容。\n\n"
            )
        elif action_name == 'continue_current':
            action_guidance = (
                "当前改进方向为微调当前策略（continue_current），请重点提炼与以下内容相关的建议：\n"
                "- 训练技巧（标签平滑、损失函数设计等）\n"
                "- 数据增强方法\n"
                "- 微调策略\n\n"
            )

        prompt = (
            f"以下是与 {model_name or 'EEG分类模型'} 运动想象脑电分类相关的最新论文摘要：\n\n"
            f"{papers_text}\n\n"
            f"{action_guidance}"
            f"请为每篇论文分别提炼出 1-2 条简洁、可操作的改进建议。\n"
            f"格式：先写论文标题，再列出建议，每条建议一句话，不超过50字。"
        )

        messages = [{"role": "user", "content": prompt}]

        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] 正在通过 LLM 提炼 {len(new_papers)} 篇新论文的改进建议...")

        try:
            new_suggestions = (
                llm_chat(
                    messages,
                    max_tokens=512,
                    label="tools.web_search.arxiv_paper_digest",
                )
                or ""
            )
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] LLM 提炼建议失败: {e}")
            new_suggestions = ""

        if new_suggestions:
            per_paper = _split_suggestions_by_paper(new_suggestions, new_papers)
            new_paper_idx = 0
            for key in sorted(retrieve_dict.keys(), key=int):
                if not retrieve_dict[key]['content'] and new_paper_idx < len(new_papers):
                    content = per_paper[new_paper_idx] if new_paper_idx < len(per_paper) else new_suggestions
                    retrieve_dict[key]['content'] = content
                    arxiv_id = new_papers[new_paper_idx].get('arxiv_id')
                    if arxiv_id:
                        _paper_cache[arxiv_id] = {
                            'name': new_papers[new_paper_idx].get('title', ''),
                            'number': arxiv_id,
                            'summary': new_papers[new_paper_idx].get('summary', ''),
                            'content': content,
                        }
                    new_paper_idx += 1

    all_suggestions = "\n".join(
        s for s in [*cached_suggestions, new_suggestions] if s
    )

    if all_suggestions:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] === 论文提炼改进建议 ===")
        print(all_suggestions)
        print(f"[{ts}] === 建议结束 ===")

    return all_suggestions or "", papers, retrieve_dict


def _split_suggestions_by_paper(text, papers):
    parts = []
    lines = text.strip().split('\n')
    current = []
    for line in lines:
        if any(p.get('title', '???')[:30] in line for p in papers) and current:
            parts.append('\n'.join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        parts.append('\n'.join(current))
    return parts
