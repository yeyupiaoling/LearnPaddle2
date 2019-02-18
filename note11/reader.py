import os
import random
from multiprocessing import cpu_count
import numpy as np
import paddle
from PIL import Image


# 训练图片的预处理
def train_mapper(sample):
    img_path, label, crop_size, resize_size = sample
    try:
        img = Image.open(img_path)
        # 统一图片大小
        img = img.resize((resize_size, resize_size), Image.ANTIALIAS)
        # 随机水平翻转
        r1 = random.random()
        if r1 > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # 随机垂直翻转
        r2 = random.random()
        if r2 > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        # 随机角度翻转
        r3 = random.randint(-3, 3)
        img = img.rotate(r3, expand=False)
        # 随机裁剪
        r4 = random.randint(0, int(resize_size - crop_size))
        r5 = random.randint(0, int(resize_size - crop_size))
        box = (r4, r5, r4 + crop_size, r5 + crop_size)
        img = img.crop(box)
        # 把图片转换成numpy值
        img = np.array(img).astype(np.float32)
        # 转换成CHW
        img = img.transpose((2, 0, 1))
        # 转换成BGR
        img = img[(2, 1, 0), :, :] / 255.0
        return img, int(label)
    except:
        print("%s 该图片错误，请删除该图片并重新创建图像数据列表" % img_path)


# 获取训练的reader
def train_reader(train_list_path, crop_size, resize_size):
    father_path = os.path.dirname(train_list_path)

    def reader():
        with open(train_list_path, 'r') as f:
            lines = f.readlines()
            # 打乱图像列表
            np.random.shuffle(lines)
            # 开始获取每张图像和标签
            for line in lines:
                img, label = line.split('\t')
                img = os.path.join(father_path, img)
                yield img, label, crop_size, resize_size

    return paddle.reader.xmap_readers(train_mapper, reader, cpu_count(), 102400)


# 测试图片的预处理
def test_mapper(sample):
    img, label, crop_size = sample
    img = Image.open(img)
    # 统一图像大小
    img = img.resize((crop_size, crop_size), Image.ANTIALIAS)
    # 转换成numpy值
    img = np.array(img).astype(np.float32)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = img[(2, 1, 0), :, :] / 255.0
    return img, int(label)


# 测试的图片reader
def test_reader(test_list_path, crop_size):
    father_path = os.path.dirname(test_list_path)

    def reader():
        with open(test_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img, label = line.split('\t')
                img = os.path.join(father_path, img)
                yield img, label, crop_size

    return paddle.reader.xmap_readers(test_mapper, reader, cpu_count(), 1024)