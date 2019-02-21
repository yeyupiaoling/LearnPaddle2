import os
import random
from multiprocessing import cpu_count
import numpy as np
import paddle
from PIL import Image


# 测试图片的预处理
def train_mapper(sample):
    img, crop_size = sample
    img = Image.open(img)
    # 随机水平翻转
    r1 = random.random()
    if r1 > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # 等比例缩放和中心裁剪
    width = img.size[0]
    height = img.size[1]
    if width < height:
        ratio = width / crop_size
        width = width / ratio
        height = height / ratio
        img = img.resize((int(width), int(height)), Image.ANTIALIAS)
        height = height / 2
        crop_size2 = crop_size / 2
        box = (0, int(height - crop_size2), int(width), int(height + crop_size2))
    else:
        ratio = height / crop_size
        height = height / ratio
        width = width / ratio
        img = img.resize((int(width), int(height)), Image.ANTIALIAS)
        width = width / 2
        crop_size2 = crop_size / 2
        box = (int(width - crop_size2), 0, int(width + crop_size2), int(height))
    img = img.crop(box)
    img = img.resize((crop_size, crop_size), Image.ANTIALIAS)

    # 把单通道图变成3通道
    if len(img.getbands()) == 1:
        img1 = img2 = img3 = img
        img = Image.merge('RGB', (img1, img2, img3))

    # 转换成numpy值
    img = np.array(img).astype(np.float32)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = img[(2, 1, 0), :, :] / 255.0
    return img


# 测试的图片reader
def train_reader(train_image_path, crop_size):
    pathss = []
    for root, dirs, files in os.walk(train_image_path):
        path = [os.path.join(root, name) for name in files]
        pathss.extend(path)

    def reader():
        for line in pathss:
            yield line, crop_size

    return paddle.reader.xmap_readers(train_mapper, reader, cpu_count(), 1024)
