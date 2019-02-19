import os


def create_data_list(data_root_path):
    with open(data_root_path + 'test_list.txt', 'w') as f:
        pass
    with open(data_root_path + 'train_list.txt', 'w') as f:
        pass

    with open(os.path.join(data_root_path, 'dict_txt.txt'), 'r', encoding='utf-8') as f_data:
        dict_txt = eval(f_data.readlines()[0])

    with open(os.path.join(data_root_path, 'news_classify_data.txt'), 'r', encoding='utf-8') as f_data:
        lines = f_data.readlines()
    i = 0
    for line in lines:
        title = line.split('_!_')[-1].replace('\n', '')
        l = line.split('_!_')[1]
        labs = ""
        if i % 10 == 0:
            with open(os.path.join(data_root_path, 'test_list.txt'), 'a', encoding='utf-8') as f_test:
                for s in title:
                    lab = str(dict_txt[s])
                    labs = labs + lab + ','
                labs = labs[:-1]
                labs = labs + '\t' + l + '\n'
                f_test.write(labs)
        else:
            with open(os.path.join(data_root_path, 'train_list.txt'), 'a', encoding='utf-8') as f_train:
                for s in title:
                    lab = str(dict_txt[s])
                    labs = labs + lab + ','
                labs = labs[:-1]
                labs = labs + '\t' + l + '\n'
                f_train.write(labs)
        i += 1
    print("数据列表生成完成！")


# 把下载得数据生成一个字典
def create_dict(data_path, dict_path):
    dict_set = set()
    # 读取已经下载得数据
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 把数据生成一个元组
    for line in lines:
        title = line.split('_!_')[-1].replace('\n', '')
        for s in title:
            dict_set.add(s)
    # 把元组转换成字典，一个字对应一个数字
    dict_list = []
    i = 0
    for s in dict_set:
        dict_list.append([s, i])
        i += 1
    # 添加未知字符
    dict_txt = dict(dict_list)
    end_dict = {"<unk>": i}
    dict_txt.update(end_dict)
    # 把这些字典保存到本地中
    with open(dict_path, 'w', encoding='utf-8') as f:
        f.write(str(dict_txt))

    print("数据字典生成完成！")


# 获取字典的长度
def get_dict_len(dict_path):
    with open(dict_path, 'r', encoding='utf-8') as f:
        line = eval(f.readlines()[0])

    return len(line.keys())


if __name__ == '__main__':
    # 把生产的数据列表都放在自己的总类别文件夹中
    data_root_path = "datasets/"
    data_path = os.path.join(data_root_path, 'news_classify_data.txt')
    dict_path = os.path.join(data_root_path, "dict_txt.txt")
    # 创建数据字典
    create_dict(data_path, dict_path)
    # 创建数据列表
    create_data_list(data_root_path)
