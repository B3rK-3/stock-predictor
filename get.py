import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) RedditScraper/0.1 by YourUsername'
}

url = 'https://www.reddit.com/r/pennystocks/top.json'
response = requests.get(url, headers=headers)

print(response.status_code)
if response.status_code == 200:
    data = response.json()
    articles = []
    for each in data['data']['children']:
        each = each['data']
        articles.append({
            'title': each['title'],
            'body': each['selftext'],
            'url': each['url']
        })
    print(len(articles))
else:
    print(response.json())

