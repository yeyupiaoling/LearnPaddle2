import os

import requests
import json
import time

# 分类新闻参数
g_cnns = [
    [0, '民生', 'news_story'],
    [1, '文化', 'news_culture'],
    [2, '娱乐', 'news_entertainment'],
    [3, '体育', 'news_sports'],
    [4, '财经', 'news_finance'],
    [5, '房产', 'news_house'],
    [6, '汽车', 'news_car'],
    [7, '教育', 'news_edu'],
    [8, '科技', 'news_tech'],
    [9, '军事', 'news_military'],
    [10, '旅游', 'news_travel'],
    [11, '国际', 'news_world'],
    [12, '证券', 'stock'],
    [13, '农业', 'news_agriculture'],
    [14, '游戏', 'news_game']
]

# 已经下载的新闻标题的ID
downloaded_data_id = []
# 已经下载新闻标题的数量
downloaded_sum = 0


def get_data(tup):
    global downloaded_data_id
    global downloaded_sum
    print('============%s============' % tup[1])
    url = "http://it.snssdk.com/api/news/feed/v63/"
    # 分类新闻的访问参数
    querystring = {"category": tup[2]}
    # 进行网络请求
    response = requests.request("GET", url, params=querystring)
    # 获取返回的数据
    new_data = json.loads(response.text)
    with open('news_classify_data.txt', 'a', encoding='utf-8') as fp:
        for item in new_data['data']:
            item = item['content']
            item = item.replace('\"', '"')
            item = json.loads(item)
            # 判断数据中是否包含id和新闻标题
            if 'item_id' in item.keys() and 'title' in item.keys():
                item_id = item['item_id']
                print(downloaded_sum, tup[0], tup[1], item['item_id'], item['title'])
                # 通过新闻id判断是否已经下载过
                if item_id not in downloaded_data_id:
                    downloaded_data_id.append(item_id)
                    # 安装固定格式追加写入文件中
                    line = u"{}_!_{}_!_{}_!_{}".format(item['item_id'], tup[0], tup[1], item['title'])
                    line = line.replace('\n', '').replace('\r', '')
                    line = line + '\n'
                    fp.write(line)
                    downloaded_sum += 1


def get_routine():
    global downloaded_sum
    # 从文件中读取已经有的数据，避免数据重复
    data_path = 'news_classify_data.txt'
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
            downloaded_sum = len(lines)
            for line in lines:
                item_id = int(line.split('_!_')[0])
                downloaded_data_id.append(item_id)
            print('在文件中已经读起了%d条数据' % downloaded_sum)

    while 1:
        time.sleep(100)
        for tp in g_cnns:
            get_data(tp)

        if downloaded_sum >= 400000:
            break


if __name__ == '__main__':
    get_routine()