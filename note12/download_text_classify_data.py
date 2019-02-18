import os
import random

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
    t = int(time.time() / 10000)
    t = random.randint(6 * t, 10 * t)
    querystring = {"category": tup[2], "max_behot_time": t, "last_refresh_sub_entrance_interval": "1524907088", "loc_mode": "5",
                   "tt_from": "pre_load_more", "cp": "51a5ee4f38c50q1", "plugin_enable": "0", "iid": "31047425023",
                   "device_id": "51425358841", "ac": "wifi", "channel": "tengxun", "aid": "13",
                   "app_name": "news_article", "version_code": "631", "version_name": "6.3.1",
                   "device_platform": "android",
                   "ab_version": "333116,297979,317498,336556,295827,325046,239097,324283,170988,335432,332098,325198,336443,330632,297058,276203,286212,313219,328615,332041,329358,322321,327537,335710,333883,335102,334828,328670,324007,317077,334305,280773,335671,319960,333985,331719,336452,214069,31643,332881,333968,318434,207253,266310,321519,247847,281298,328218,335998,325618,333327,336199,323429,287591,288418,260650,326188,324614,335477,271178,326588,326524,326532",
                   "ab_client": "a1,c4,e1,f2,g2,f7", "ab_feature": "94563,102749", "abflag": "3", "ssmix": "a",
                   "device_type": "MuMu", "device_brand": "Android", "language": "zh", "os_api": "19",
                   "os_version": "4.4.4", "uuid": "008796762094657", "openudid": "b7215ea70ca32066",
                   "manifest_version_code": "631", "resolution": "1280*720", "dpi": "240",
                   "update_version_code": "6310", "_rticket": "1524907088018", "plugin": "256"}

    headers = {
        'cache-control': "no-cache",
        'postman-token': "26530547-e697-1e8b-fd82-7c6014b3ee86",
        'User-Agent': 'Dalvik/1.6.0 (Linux; U; Android 4.4.4; MuMu Build/V417IR) NewsArticle/6.3.1 okhttp/3.7.0.2'
    }

    # 进行网络请求
    response = requests.request("GET", url, headers=headers, params=querystring)
    # 获取返回的数据
    new_data = json.loads(response.text)
    with open('datasets/news_classify_data.txt', 'a', encoding='utf-8') as fp:
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
    data_path = 'datasets/news_classify_data.txt'
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
            downloaded_sum = len(lines)
            for line in lines:
                item_id = int(line.split('_!_')[0])
                downloaded_data_id.append(item_id)
            print('在文件中已经读起了%d条数据' % downloaded_sum)
    else:
        os.makedirs(os.path.dirname(data_path))

    while 1:
        time.sleep(100)
        for tp in g_cnns:
            get_data(tp)

        if downloaded_sum >= 400000:
            break


if __name__ == '__main__':
    get_routine()
