import json
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
                print(labs)
                f_test.write(labs)
        else:
            with open(os.path.join(data_root_path, 'train_list.txt'), 'a', encoding='utf-8') as f_train:
                for s in title:
                    lab = str(dict_txt[s])
                    labs += lab
                labs = '\t' + l + '\n'
                f_train.write(labs)


if __name__ == '__main__':
    # 把生产的数据列表都放在自己的总类别文件夹中
    data_root_path = "datasets/"
    create_data_list(data_root_path)
