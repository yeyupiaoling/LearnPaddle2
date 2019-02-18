from multiprocessing import cpu_count

import numpy as np
import paddle


# 训练数据的预处理
def train_mapper(sample):
    data, label = sample
    data = [int(data) for data in data.split(',')]
    return data, int(label)


# 训练数据的reader
def train_reader(train_list_path):

    def reader():
        with open(train_list_path, 'r') as f:
            lines = f.readlines()
            # 打乱数据
            np.random.shuffle(lines)
            # 开始获取每张图像和标签
            for line in lines:
                data, label = line.split('\t')
                yield data, label

    return paddle.reader.xmap_readers(train_mapper, reader, cpu_count(), 1024)


# 测试数据的预处理
def test_mapper(sample):
    data, label = sample
    data = [int(data) for data in data.split(',')]
    return data, int(label)


# 测试数据的reader
def test_reader(test_list_path):

    def reader():
        with open(test_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data, label = line.split('\t')
                yield data, label

    return paddle.reader.xmap_readers(test_mapper, reader, cpu_count(), 1024)