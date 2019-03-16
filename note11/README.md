@[TOC]

GitHub地址：https://github.com/yeyupiaoling/LearnPaddle2/tree/master/note11

# 前言
本章将介绍如何使用PaddlePaddle训练自己的图片数据集，在之前的图像数据集中，我们都是使用PaddlePaddle自带的数据集，本章我们就来学习如何让PaddlePaddle训练我们自己的图片数据集。

# 爬取图像
在本章中，我们使用的是自己的图片数据集，所以我们需要弄一堆图像来制作训练的数据集。下面我们就编写一个爬虫程序，让其帮我们从百度图片中爬取相应类别的图片。

创建一个`download_image.py`文件用于编写爬取图片程序。首先导入所需的依赖包。
```python
import re
import uuid
import requests
import os
import numpy
import imghdr
from PIL import Image
```

然后编写一个下载图片的函数，这个是程序核心代码。参数是下载图片的关键、保存的名字、下载图片的数量。关键字是百度搜索图片的关键。
```python
# 获取百度图片下载图片
def download_image(key_word, save_name, download_max):
    download_sum = 0
    str_gsm = '80'
    # 把每个类别的图片存放在单独一个文件夹中
    save_path = 'images' + '/' + save_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    while download_sum < download_max:
        # 下载次数超过指定值就停止下载
        if download_sum >= download_max:
            break
        str_pn = str(download_sum)
        # 定义百度图片的路径
        url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&' \
              'word=' + key_word + '&pn=' + str_pn + '&gsm=' + str_gsm + '&ct=&ic=0&lm=-1&width=0&height=0'
        print('正在下载 %s 的第 %d 张图片.....' % (key_word, download_sum))
        try:
            # 获取当前页面的源码
            result = requests.get(url, timeout=30).text
            # 获取当前页面的图片URL
            img_urls = re.findall('"objURL":"(.*?)",', result, re.S)
            if len(img_urls) < 1:
                break
            # 把这些图片URL一个个下载
            for img_url in img_urls:
                # 获取图片内容
                img = requests.get(img_url, timeout=30)
                img_name = save_path + '/' + str(uuid.uuid1()) + '.jpg'
                # 保存图片
                with open(img_name, 'wb') as f:
                    f.write(img.content)
                download_sum += 1
                if download_sum >= download_max:
                    break
        except Exception as e:
            print('【错误】当前图片无法下载，%s' % e)
            download_sum += 1
            continue
    print('下载完成')
```

图片下载完成之后，需要删除一家损坏的图片，因为在下载的过程中，由于图片本身的问题或者下载过程造成的图片损坏，需要把这些已经损坏的图片上传。下面的函数就是删除所有损坏的图片，根据图像数据集的目录读取获取所有图片文件的路径，然后使用`imghdr`工具获取图片的类型是否为`png`或者`jpg`来判断图片文件是否完整，最后再删除根据图片的通道数据来删除灰度图片。
```python
# 删除不是JPEG或者PNG格式的图片
def delete_error_image(father_path):
    # 获取父级目录的所有文件以及文件夹
    try:
        image_dirs = os.listdir(father_path)
        for image_dir in image_dirs:
            image_dir = os.path.join(father_path, image_dir)
            # 如果是文件夹就继续获取文件夹中的图片
            if os.path.isdir(image_dir):
                images = os.listdir(image_dir)
                for image in images:
                    image = os.path.join(image_dir, image)
                    try:
                        # 获取图片的类型
                        image_type = imghdr.what(image)
                        # 如果图片格式不是JPEG同时也不是PNG就删除图片
                        if image_type is not 'jpeg' and image_type is not 'png':
                            os.remove(image)
                            print('已删除：%s' % image)
                            continue
                        # 删除灰度图
                        img = numpy.array(Image.open(image))
                        if len(img.shape) is 2:
                            os.remove(image)
                            print('已删除：%s' % image)
                    except:
                        os.remove(image)
                        print('已删除：%s' % image)
    except:
        pass
```

最后在main入口中通过调用两个函数来完成下载图像数据集，使用中文进行百度搜索图片，使用英文是为了出现中文路径导致图片读取错误。
```python
if __name__ == '__main__':
    # 定义要下载的图片中文名称和英文名称，ps：英文名称主要是为了设置文件夹名
    key_words = {'西瓜': 'watermelon', '哈密瓜': 'cantaloupe',
                 '樱桃': 'cherry', '苹果': 'apple', '黄瓜': 'cucumber', '胡萝卜': 'carrot'}
    # 每个类别下载一千个
    max_sum = 500
    for key_word in key_words:
        save_name = key_words[key_word]
        download_image(key_word, save_name, max_sum)

    # 删除错误图片
    delete_error_image('images/')
```

输出信息：
```
正在下载 哈密瓜 的第 0 张图片.....
【错误】当前图片无法下载，HTTPConnectionPool(host='www.boyingsj.com', port=80): Read timed out.
正在下载 哈密瓜 的第 10 张图片.....
```

**注意：** 下载处理完成之后，还可能存在其他杂乱的图片，所以还需要我们手动删除这些不属于这个类别的图片，这才算完成图像数据集的制作。


# 创建图像列表
创建一个`create_data_list.py`文件，在这个程序中，我们只要把爬取保存图片的路径的文件夹路径传进去就可以了，生成固定格式的列表，格式为`图片的路径 <Tab> 图片类别的标签`：
```python
import json
import os

def create_data_list(data_root_path):
    with open(data_root_path + "test.list", 'w') as f:
        pass
    with open(data_root_path + "train.list", 'w') as f:
        pass
    # 所有类别的信息
    class_detail = []
    # 获取所有类别
    class_dirs = os.listdir(data_root_path)
    # 类别标签
    class_label = 0
    # 获取总类别的名称
    father_paths = data_root_path.split('/')
    while True:
        if father_paths[len(father_paths) - 1] == '':
            del father_paths[len(father_paths) - 1]
        else:
            break
    father_path = father_paths[len(father_paths) - 1]

    all_class_images = 0
    other_file = 0
    # 读取每个类别
    for class_dir in class_dirs:
        if class_dir == 'test.list' or class_dir == "train.list" or class_dir == 'readme.json':
            other_file += 1
            continue
        print('正在读取类别：%s' % class_dir)
        # 每个类别的信息
        class_detail_list = {}
        test_sum = 0
        trainer_sum = 0
        # 统计每个类别有多少张图片
        class_sum = 0
        # 获取类别路径
        path = data_root_path + "/" + class_dir
        # 获取所有图片
        img_paths = os.listdir(path)
        for img_path in img_paths:
            # 每张图片的路径
            name_path = class_dir + '/' + img_path
            # 如果不存在这个文件夹,就创建
            if not os.path.exists(data_root_path):
                os.makedirs(data_root_path)
            # 每10张图片取一个做测试数据
            if class_sum % 10 == 0:
                test_sum += 1
                with open(data_root_path + "test.list", 'a') as f:
                    f.write(name_path + "\t%d" % class_label + "\n")
            else:
                trainer_sum += 1
                with open(data_root_path + "train.list", 'a') as f:
                    f.write(name_path + "\t%d" % class_label + "\n")
            class_sum += 1
            all_class_images += 1
        # 说明的json文件的class_detail数据
        class_detail_list['class_name'] = class_dir
        class_detail_list['class_label'] = class_label
        class_detail_list['class_test_images'] = test_sum
        class_detail_list['class_trainer_images'] = trainer_sum
        class_detail.append(class_detail_list)
        class_label += 1
    # 获取类别数量
    all_class_sum = len(class_dirs) - other_file
    # 说明的json文件信息
    readjson = {}
    readjson['all_class_name'] = father_path
    readjson['all_class_sum'] = all_class_sum
    readjson['all_class_images'] = all_class_images
    readjson['class_detail'] = class_detail
    jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',', ': '))
    with open(data_root_path + "readme.json", 'w') as f:
        f.write(jsons)
    print('图像列表已生成')
```

最后执行就可以生成图像的列表。
```python
if __name__ == '__main__':
    # 把生产的数据列表都放在自己的总类别文件夹中
    data_root_path = "images/"
    create_data_list(data_root_path)
```

输出信息：
```
正在读取类别：apple
正在读取类别：cantaloupe
正在读取类别：carrot
正在读取类别：cherry
正在读取类别：cucumber
正在读取类别：watermelon
图像列表已生成
```

运行这个程序之后，会生成在data文件夹中生成一个单独的大类文件夹，比如我们这次是使用到蔬菜类，所以我生成一个`vegetables`文件夹，在这个文件夹下有3个文件：
|文件名|作用|
|:---:|:---:|
|trainer.list|用于训练的图像列表|
|test.list|用于测试的图像列表|
|readme.json|该数据集的json格式的说明,方便以后使用|

`readme.json`文件的格式如下，可以很清楚看到整个数据的图像数量,总类别名称和类别数量，还有每个类对应的标签，类别的名字，该类别的测试数据和训练数据的数量：
```json
{
    "all_class_images": 2200,
    "all_class_name": "images",
    "all_class_sum": 2,
    "class_detail": [
        {
            "class_label": 1,
            "class_name": "watermelon",
            "class_test_images": 110,
            "class_trainer_images": 990
        },
        {
            "class_label": 2,
            "class_name": "cantaloupe",
            "class_test_images": 110,
            "class_trainer_images": 990
        }
    ]
}
```

# 定义模型
创建一个`mobilenet_v1.py`文件，在本章我们使用的是MobileNet神经网络，MobileNet是Google针对手机等嵌入式设备提出的一种轻量级的深层神经网络，它的核心思想就是卷积核的巧妙分解，可以有效减少网络参数，从而达到减小训练时网络的模型。因为太大的模型模型文件是不利于移植到移动设备上的，比如我们把模型文件迁移到Android手机应用上，那么模型文件的大小就直接影响应用安装包的大小。以下就是使用PaddlePaddle定义的MobileNet神经网络：
```python
import paddle.fluid as fluid

def conv_bn_layer(input, filter_size, num_filters, stride,
                  padding, channels=None, num_groups=1, act='relu', use_cudnn=True):
    conv = fluid.layers.conv2d(input=input,
                               num_filters=num_filters,
                               filter_size=filter_size,
                               stride=stride,
                               padding=padding,
                               groups=num_groups,
                               act=None,
                               use_cudnn=use_cudnn,
                               bias_attr=False)

    return fluid.layers.batch_norm(input=conv, act=act)
```

```python
def depthwise_separable(input, num_filters1, num_filters2, num_groups, stride, scale):
    depthwise_conv = conv_bn_layer(input=input,
                                   filter_size=3,
                                   num_filters=int(num_filters1 * scale),
                                   stride=stride,
                                   padding=1,
                                   num_groups=int(num_groups * scale),
                                   use_cudnn=False)

    pointwise_conv = conv_bn_layer(input=depthwise_conv,
                                   filter_size=1,
                                   num_filters=int(num_filters2 * scale),
                                   stride=1,
                                   padding=0)
    return pointwise_conv
```

```python
def net(input, class_dim, scale=1.0):
    # conv1: 112x112
    input = conv_bn_layer(input=input,
                          filter_size=3,
                          channels=3,
                          num_filters=int(32 * scale),
                          stride=2,
                          padding=1)

    # 56x56
    input = depthwise_separable(input=input,
                                num_filters1=32,
                                num_filters2=64,
                                num_groups=32,
                                stride=1,
                                scale=scale)

    input = depthwise_separable(input=input,
                                num_filters1=64,
                                num_filters2=128,
                                num_groups=64,
                                stride=2,
                                scale=scale)

    # 28x28
    input = depthwise_separable(input=input,
                                num_filters1=128,
                                num_filters2=128,
                                num_groups=128,
                                stride=1,
                                scale=scale)

    input = depthwise_separable(input=input,
                                num_filters1=128,
                                num_filters2=256,
                                num_groups=128,
                                stride=2,
                                scale=scale)

    # 14x14
    input = depthwise_separable(input=input,
                                num_filters1=256,
                                num_filters2=256,
                                num_groups=256,
                                stride=1,
                                scale=scale)

    input = depthwise_separable(input=input,
                                num_filters1=256,
                                num_filters2=512,
                                num_groups=256,
                                stride=2,
                                scale=scale)

    # 14x14
    for i in range(5):
        input = depthwise_separable(input=input,
                                    num_filters1=512,
                                    num_filters2=512,
                                    num_groups=512,
                                    stride=1,
                                    scale=scale)
    # 7x7
    input = depthwise_separable(input=input,
                                num_filters1=512,
                                num_filters2=1024,
                                num_groups=512,
                                stride=2,
                                scale=scale)

    input = depthwise_separable(input=input,
                                num_filters1=1024,
                                num_filters2=1024,
                                num_groups=1024,
                                stride=1,
                                scale=scale)

    feature = fluid.layers.pool2d(input=input,
                                  pool_size=0,
                                  pool_stride=1,
                                  pool_type='avg',
                                  global_pooling=True)

    net = fluid.layers.fc(input=feature,
                          size=class_dim,
                          act='softmax')
    return net
```

# 定义数据读取
创建一个`reader.py`文件，这个程序就是用户训练和测试的使用读取数据的。训练的时候，通过这个程序从本地读取图片，然后通过一系列的预处理操作，最后转换成训练所需的Numpy数组。

首先导入所需的包，其中`cpu_count`是获取当前计算机有多少个CPU，然后使用多线程读取数据。
```python
import os
import random
from multiprocessing import cpu_count
import numpy as np
import paddle
from PIL import Image
```

首先定义一个`train_mapper()`函数，这个函数是根据传入进来的图片路径来对图片进行预处理，比如训练的时候需要统一图片的大小，同时也使用多种的数据增强的方式，如水平翻转、垂直翻转、角度翻转、随机裁剪，这些方式都可以让有限的图片数据集在训练的时候成倍的增加。最后因为PIL打开图片存储顺序为H(高度)，W(宽度)，C(通道)，PaddlePaddle要求数据顺序为CHW，所以需要转换顺序。最后返回的是处理后的图片数据和其对应的标签。
```python
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
```

这个`train_reader()`函数是根据已经创建的图像列表解析得到每张图片的路径和其他对应的标签，然后使用`paddle.reader.xmap_readers()`把数据传递给上面定义的`train_mapper()`函数进行处理，最后得到一个训练所需的reader。
```python
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
```

这是一个测试数据的预处理函数`test_mapper()`，这个没有做太多处理，因为测试的数据不需要数据增强操作，只需统一图片大小和设置好图片的通过顺序和数据类型即可。
```python
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
```

这个是测试的reader函数`test_reader()`，这个跟训练的reader函数定义一样。
```python
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
```

# 训练模型
万事俱备，只等训练了。关于PaddlePaddle训练流程，我们已经非常熟悉了，那么我们就简单地过一遍。

创建`train.py`文件，首先导入所需的包，其中包括我们定义的MobileNet模型和数据读取程序：
```python
import os
import shutil
import mobilenet_v1
import paddle as paddle
import reader
import paddle.fluid as fluid
```

然后定义数据输入层，这次我们使用的是图片大小是224，这比之前使用的CIFAR数据集的32大小要大很多，所以训练其他会慢不少。至于`resize_size`是用于统一缩放到这个大小，然后再随机裁剪成`crop_size`大小，`crop_size`才是最终训练图片的大小。
```python
crop_size = 224
resize_size = 250

# 定义输入层
image = fluid.layers.data(name='image', shape=[3, crop_size, crop_size], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
```

接着获取MobileNet网络的分类器，传入的第一个参数就是上面定义的输入层，第二个是分类的类别大小，比如我们这次爬取的图像类别数量是6个。
```python
# 获取分类器，因为这次只爬取了6个类别的图片，所以分类器的类别大小为6
model = mobilenet_v1.net(image, 6)
```

再接着是获取损失函数和平均准确率函数，还有测试程序和优化方法，这个优化方法我加了正则，因为爬取的图片数量太少，在训练容易过拟合，所以加上正则一定程度上可以抑制过拟合。
```python
# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

# 获取训练和测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=1e-3,
                                          regularization=fluid.regularizer.L2DecayRegularizer(1e-4))
opts = optimizer.minimize(avg_cost)
```

这里就是获取训练测试是所以想的数据读取reader，通过使用`paddle.batch()`函数可以把多条数据打包成一个批次，训练的时候是按照一个个批次训练的。
```python
# 获取自定义数据
train_reader = paddle.batch(reader=reader.train_reader('images/train.list', crop_size, resize_size), batch_size=32)
test_reader = paddle.batch(reader=reader.test_reader('images/test.list', crop_size), batch_size=32)
```

执行训练之前，还需要创建一个执行器，建议使用GPU进行训练，因为我们训练的图片比较大，所以使用CPU训练速度会相当的慢。
```python
# 定义一个使用GPU的执行器
place = fluid.CUDAPlace(0)
# place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
```

最后终于可以执行训练了，这里跟在前些章节都几乎一样，就不重复介绍了。
```python
# 训练100次
for pass_id in range(100):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, acc])

        # 每100个batch打印一次信息
        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))

    # 进行测试
    test_accs = []
    test_costs = []
    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=test_program,
                                      feed=feeder.feed(data),
                                      fetch_list=[avg_cost, acc])
        test_accs.append(test_acc[0])
        test_costs.append(test_cost[0])
    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))
    test_acc = (sum(test_accs) / len(test_accs))
    print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))
```

训练的过程中可以保存预测模型，用于之后的预测。笔者一般是每一个pass保存一次模型。
```python
    # 保存预测模型
    save_path = 'infer_model/'
    # 删除旧的模型文件
    shutil.rmtree(save_path, ignore_errors=True)
    # 创建保持模型文件目录
    os.makedirs(save_path)
    # 保存预测模型
    fluid.io.save_inference_model(save_path, feeded_var_names=[image.name], target_vars=[model], executor=exe)
```

训练输出的信息：
```
Pass:0, Batch:0, Cost:1.84754, Accuracy:0.15625
Test:0, Cost:4.66276, Accuracy:0.17857
Pass:1, Batch:0, Cost:1.04008, Accuracy:0.59375
Test:1, Cost:1.23828, Accuracy:0.54464
Pass:2, Batch:0, Cost:1.04778, Accuracy:0.65625
Test:2, Cost:0.99189, Accuracy:0.64286
Pass:3, Batch:0, Cost:1.21555, Accuracy:0.65625
Test:3, Cost:1.01552, Accuracy:0.57589
Pass:4, Batch:0, Cost:0.64620, Accuracy:0.81250
Test:4, Cost:1.19264, Accuracy:0.63393
```

# 预测图片
经过上面训练后，得到了一个预测模型，下面我们就使用一个预测模型来预测一些图片。

创建一个`infer.py`文件作为预测程序。首先导入所需的依赖包。
```python
import paddle.fluid as fluid
from PIL import Image
import numpy as np
```

创建一个执行器，这些不需要训练，所以可以使用CPU进行预测，速度不会太慢，当然，使用GPU的预测速度会更快一些。
```python
# 创建执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
```

然后加载预测模型，获取预测程序和输入层的名字，还有网络的分类器。
```python
# 保存预测模型路径
save_path = 'infer_model/'
# 从模型中获取预测程序、输入数据名称列表、分类器
[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)
```

预测图片之前，还需要对图片进行预处理，处理的方式跟测试的时候处理的方式一样。
```python
# 预处理图片
def load_image(file):
    img = Image.open(file)
    # 统一图像大小
    img = img.resize((224, 224), Image.ANTIALIAS)
    # 转换成numpy值
    img = np.array(img).astype(np.float32)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = img[(2, 1, 0), :, :] / 255.0
    img = np.expand_dims(img, axis=0)
    return img
```

最后获取经过预处理的图片数据，再使用这些图像数据进行预测，得到分类结果。
```python
# 获取图片数据
img = load_image('images/apple/0fdd5422-31e0-11e9-9cfd-3c970e769528.jpg')

# 执行预测
result = exe.run(program=infer_program,
                 feed={feeded_var_names[0]: img},
                 fetch_list=target_var)
```

我们可以通过解析分类的结果，获取概率最大类别标签。关于预测输出的`result`是数据，它是3维的，第一层是输出本身就是一个数组，第二层图片的数量，因为PaddlePaddle支持多张图片同时预测，最后一层就是每个类别的概率，这个概率的总和为1，概率最大的标签就是预测结果。
```python
# 显示图片并输出结果最大的label
lab = np.argsort(result)[0][0][-1]

names = ['苹果', '哈密瓜', '胡萝卜', '樱桃', '黄瓜', '西瓜']

print('预测结果标签为：%d， 名称为：%s， 概率为：%f' % (lab, names[lab], result[0][0][lab]))
```

预测输出的结果：
```
预测结果标签为：0， 名称为：苹果， 概率为：0.948698
```

# 参考资料
1. https://yeyupiaoling.blog.csdn.net/article/details/79095265
