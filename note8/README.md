@[TOC]

# 前言
本系列教程中，前面介绍的都没有保存模型，训练之后也就结束了。那么本章就介绍如果在训练过程中保存模型，用于之后预测或者恢复训练，又或者由于其他数据集的预训练模型。本章会介绍三种保存模型和使用模型的方式。

# 训练模型
在训练模型的过程中我们可以随时保存模型，当时也可以在训练开始之前加载之前训练过程的模型。为了介绍这三个保存模型的方式，一共编写了三个Python程序进行介绍，分别是`save_infer_model.py`、	`save_use_params_model.py`、`save_use_persistables_model.py`。

导入相关的依赖库
```python
import os
import shutil
import paddle as paddle
import paddle.dataset.cifar as cifar
import paddle.fluid as fluid
```

定义一个残差神经网络，这个是目前比较常用的一个网络。该神经模型可以通过增加网络的深度达到提高识别率，而不会像其他过去的神经模型那样，当网络继续加深时,反而会损失精度。
```python
# 定义残差神经网络（ResNet）
def resnet_cifar10(ipt, class_dim):
    def conv_bn_layer(input,
                      ch_out,
                      filter_size,
                      stride,
                      padding,
                      act='relu',
                      bias_attr=False):
        tmp = fluid.layers.conv2d(
            input=input,
            filter_size=filter_size,
            num_filters=ch_out,
            stride=stride,
            padding=padding,
            bias_attr=bias_attr)
        return fluid.layers.batch_norm(input=tmp, act=act)

    def shortcut(input, ch_in, ch_out, stride):
        if ch_in != ch_out:
            return conv_bn_layer(input, ch_out, 1, stride, 0, None)
        else:
            return input

    def basicblock(input, ch_in, ch_out, stride):
        tmp = conv_bn_layer(input, ch_out, 3, stride, 1)
        tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, act=None, bias_attr=True)
        short = shortcut(input, ch_in, ch_out, stride)
        return fluid.layers.elementwise_add(x=tmp, y=short, act='relu')

    # 残差块
    def layer_warp(block_func, input, ch_in, ch_out, count, stride):
        tmp = block_func(input, ch_in, ch_out, stride)
        for i in range(1, count):
            tmp = block_func(tmp, ch_out, ch_out, 1)
        return tmp

    conv1 = conv_bn_layer(ipt, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, 16, 5, 1)
    res2 = layer_warp(basicblock, res1, 16, 32, 5, 2)
    res3 = layer_warp(basicblock, res2, 32, 64, 5, 2)
    pool = fluid.layers.pool2d(input=res3, pool_size=8, pool_type='avg', pool_stride=1)
    predict = fluid.layers.fc(input=pool, size=class_dim, act='softmax')
    return predict
```

定义输出成，这里使用的数据集是cifar数据集，这个数据集的图片是宽高都为32的3通道图片，所以这里定义的图片输入层的shape是`[3, 32, 32]`。
```python
# 定义输入层
image = fluid.layers.data(name='image', shape=[3, 32, 32], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
```

获取残差神经网络的分类器，并指定分类大小是10，因为这个数据集有10个类别。
```python
# 获取分类器
model = resnet_cifar10(image, 10)
```

获取交叉熵损失函数和平均准确率，模型获取的准确率是Top1的。
```python
# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)
```

获取测试程序，用于之后的测试使。
```python
# 获取训练和测试程序
test_program = fluid.default_main_program().clone(for_test=True)
```

定义优化方法。
```python
# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=1e-3)
opts = optimizer.minimize(avg_cost)
```

获取训练和测试数据，使用的是cifar数据集，cifar数据集有两种，一种是100个类别的，一种是10个类别的，这里使用的是10个类别的。
```python
# 获取CIFART数据
train_reader = paddle.batch(cifar.train10(), batch_size=32)
test_reader = paddle.batch(cifar.test10(), batch_size=32)
```

创建执行器，因为我们使用的网络是一个比较大的网络，而且图片也比之前的灰度图要大很多。之前的MNIST数据集的每张图片大小784，而现在的是3072。当然主要是网络比之前的要大很多很多，如果使用CPU训练，速度是非常慢的，所以最好使用GPU进行训练。
```python
# 创建执行器，最好使用GPU，CPU速度太慢了
# place = fluid.CPUPlace()
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())
```

## 加载模型
创建执行器之后，就可以加载之前训练的模型了，有两种加载模型的方式，对应着两种保存模型的方式。这两种模型，可以只使用一种就可以。

 - `save_use_params_model.py`加载之前训练保存的参数模型，对应的保存接口是`fluid.io.save_params`。使用这些模型参数初始化网络参数，进行训练
```python
# 加载之前训练过的参数模型
save_path = 'models/params_model/'
if os.path.exists(save_path):
    print('使用参数模型作为预训练模型')
    fluid.io.load_params(executor=exe, dirname=save_path)
```

 - `save_use_persistables_model.py`加载之前训练保存的持久化变量模型，对应的保存接口是`fluid.io.save_persistables`。使用这些模型参数初始化网络参数，进行训练。
```python
# 加载之前训练过的检查点模型
save_path = 'models/persistables_model/'
if os.path.exists(save_path):
    print('使用持久化变量模型作为预训练模型')
    fluid.io.load_persistables(executor=exe, dirname=save_path)
```


开始训练模型。
```python
# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

for pass_id in range(10):
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

没有加载之前保存的模型
```
Pass:0, Batch:0, Cost:2.73460, Accuracy:0.03125
Pass:0, Batch:100, Cost:1.93663, Accuracy:0.25000
Pass:0, Batch:200, Cost:2.02943, Accuracy:0.12500
Pass:0, Batch:300, Cost:1.94425, Accuracy:0.25000
Pass:0, Batch:400, Cost:1.87802, Accuracy:0.21875
Pass:0, Batch:500, Cost:1.71312, Accuracy:0.25000
Pass:0, Batch:600, Cost:1.94090, Accuracy:0.18750
Pass:0, Batch:700, Cost:2.08904, Accuracy:0.12500
Pass:0, Batch:800, Cost:1.89128, Accuracy:0.12500
Pass:0, Batch:900, Cost:1.95716, Accuracy:0.21875
Pass:0, Batch:1000, Cost:1.65181, Accuracy:0.34375
```

使用参数模型作为预训练模型训练时输出的信息：
```
使用参数模型作为预训练模型
Pass:0, Batch:0, Cost:0.27627, Accuracy:0.90625
Pass:0, Batch:100, Cost:0.40026, Accuracy:0.87500
Pass:0, Batch:200, Cost:0.54928, Accuracy:0.78125
Pass:0, Batch:300, Cost:0.56526, Accuracy:0.84375
Pass:0, Batch:400, Cost:0.53501, Accuracy:0.78125
Pass:0, Batch:500, Cost:0.18596, Accuracy:0.93750
Pass:0, Batch:600, Cost:0.23747, Accuracy:0.96875
Pass:0, Batch:700, Cost:0.45520, Accuracy:0.84375
Pass:0, Batch:800, Cost:0.86205, Accuracy:0.71875
Pass:0, Batch:900, Cost:0.36981, Accuracy:0.87500
Pass:0, Batch:1000, Cost:0.37483, Accuracy:0.81250
```

持久性变量模型作为预训练模型训练时输出的信息：
```
使用持久性变量模型作为预训练模型
Pass:0, Batch:0, Cost:0.51357, Accuracy:0.81250
Pass:0, Batch:100, Cost:0.64380, Accuracy:0.78125
Pass:0, Batch:200, Cost:0.69049, Accuracy:0.62500
Pass:0, Batch:300, Cost:0.52201, Accuracy:0.87500
Pass:0, Batch:400, Cost:0.47289, Accuracy:0.81250
Pass:0, Batch:500, Cost:0.15821, Accuracy:1.00000
Pass:0, Batch:600, Cost:0.36470, Accuracy:0.87500
Pass:0, Batch:700, Cost:0.25326, Accuracy:0.90625
Pass:0, Batch:800, Cost:0.92556, Accuracy:0.78125
Pass:0, Batch:900, Cost:0.27470, Accuracy:0.93750
Pass:0, Batch:1000, Cost:0.34562, Accuracy:0.87500
```

## 保存模型
训练结束之后，就可以进行保存模型。当然也不一样要全部训练结束才保存模型，我们可以在每一个Pass训练结束之后保存一次模型。这里使用三个程序分别保存，当然也可以一次全部保存。

 - `save_infer_model.py`保存预测模型，之后用于预测图像。通过使用这个方式保存的模型，之后预测是非常方便的，具体可以阅读预测部分。
```python
# 保存预测模型
save_path = 'models/infer_model/'
# 删除旧的模型文件
shutil.rmtree(save_path, ignore_errors=True)
# 创建保持模型文件目录
os.makedirs(save_path)
# 保存预测模型
fluid.io.save_inference_model(save_path, feeded_var_names=[image.name], target_vars=[model], executor=exe)
```

 - `save_use_params_model.py`保存参数模型，之后用于初始化模型，进行训练。
```python
# 保存参数模型
save_path = 'models/params_model/'
# 删除旧的模型文件
shutil.rmtree(save_path, ignore_errors=True)
# 创建保持模型文件目录
os.makedirs(save_path)
# 保存参数模型
fluid.io.save_params(executor=exe, dirname=save_path)
```

 - `save_use_persistables_model.py`保存持久化变量模型，之后用于初始化模型，进行训练。
```python
# 保存持久化变量模型
save_path = 'models/persistables_model/'
# 删除旧的模型文件
shutil.rmtree(save_path, ignore_errors=True)
# 创建保持模型文件目录
os.makedirs(save_path)
# 保存持久化变量模型
fluid.io.save_persistables(executor=exe, dirname=save_path)
```


# 预测
在训练的时候使用`fluid.io.save_inference_model`接口保存的模型，可以通过以下`use_infer_model.py`程序预测，通过这个程序，读者会发现通过这个接口保存的模型，再次预测是非常简单的。

导入相关的依赖库
```python
import paddle.fluid as fluid
from PIL import Image
import numpy as np
```

创建一个执行器，预测图片可以使用CPU执行，这个速度不会太慢。
```python
# 创建执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
```

加载模型，这个是整个预测程序的重点，通过加载预测模型我们就可以轻松获取得到一个预测程序，输出参数的名称，以及分类器的输出。
```python
# 保存预测模型路径
save_path = 'models/infer_model/'
# 从模型中获取预测程序、输入数据名称列表、分类器
[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)
```

定义一个图像预处理的函数，这个函数可以统一图像大小，修改图像的存储顺序和图片的通道顺序，转换成numpy数据。
```python
# 预处理图片
def load_image(file):
    im = Image.open(file)
    im = im.resize((32, 32), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32)
    # PIL打开图片存储顺序为H(高度)，W(宽度)，C(通道)。
    # PaddlePaddle要求数据顺序为CHW，所以需要转换顺序。
    im = im.transpose((2, 0, 1))
    # CIFAR训练图片通道顺序为B(蓝),G(绿),R(红),
    # 而PIL打开图片默认通道顺序为RGB,因为需要交换通道。
    im = im[(2, 1, 0), :, :]  # BGR
    im = im / 255.0
    im = np.expand_dims(im, axis=0)
    return im
```

获取数据并进行预测。这里对比之前的预测方式，不需要再输入一个模拟的标签，因为在保存模型的时候，已经对这部分进行修剪，去掉了这部分不必要的输入。
```python
# 获取图片数据
img = load_image('image/cat.png')

# 执行预测
result = exe.run(program=infer_program,
                 feed={feeded_var_names[0]: img},
                 fetch_list=target_var)
```

执行预测之后，得到一个数组，这个数组是表示每个类别的概率，获取最大概率的标签，并根据标签获取获取该类的名称。
```python
# 显示图片并输出结果最大的label
lab = np.argsort(result)[0][0][-1]

names = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

print('预测结果标签为：%d， 名称为：%s， 概率为：%f' % (lab, names[lab], result[0][0][lab]))
```

预测输出结果：
```
预测结果标签为：3， 名称为：猫， 概率为：0.864919
```

关于模型的保存和使用就介绍到这里，读者可以使用这个方式保存之前学过的模型。在这个基础上，下一章我们介绍如何使用预训练模型。

同步到百度AI Studio平台：http://aistudio.baidu.com/?_=1548666175806#/projectdetail/38741
同步到科赛网K-Lab平台：https://www.kesci.com/home/project/5c3f495589f4aa002b845d6b
项目代码GitHub地址：https://github.com/yeyupiaoling/LearnPaddle2/tree/master/note7

**注意：** 最新代码以GitHub上的为准

# 参考资料
1. https://blog.csdn.net/qq_33200967/article/details/79095224
2. http://www.paddlepaddle.org/documentation/docs/zh/1.2/api_cn/io_cn.html
