import requests
import datetime
import json


def pushplus_notify(title, content):
    today = datetime.date.today()
    token = 'a1f3de940f584dbbb4c17dbe127e3e84'
    url = 'http://www.pushplus.plus/send'
    data = {
        "token": token,
        "title": title,
        "content": content
    }
    body = json.dumps(data).encode(encoding='utf-8')
    headers = {'Content-Type': 'application/json'}
    requests.post(url, data=body, headers=headers)


if __name__ == '__main__':
    pushplus_notify('code', 'finish')