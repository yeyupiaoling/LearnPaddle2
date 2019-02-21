@[TOC]

# 前言
上一章我们通过学习线性回归例子入门了深度学习，同时也熟悉了PaddlePaddle的使用方式，那么我们在本章学习更有趣的知识点卷积神经网络。深度学习之所以那么流行，很大程度上是得益于它在计算机视觉上得到非常好的效果，而在深度学习上几乎是使用卷积神经网络来提取图像的特征的。在PaddlePaddle上如何定义一个卷积神经网络，并使用它来完成一个图像识别的任务呢。在本章我们通过学习MNIST图像数据集的分类例子，来掌握卷积神经网络的使用。

# 训练模型
创建一个`mnist_classification.py`文件，首先导入所需得包，这次使用到了MNIST数据集接口，也使用了处理图像得工具包。
```python
import numpy as np
import paddle as paddle
import paddle.dataset.mnist as mnist
import paddle.fluid as fluid
from PIL import Image
import matplotlib.pyplot as plt
```

在图像识别上，使用得算法也经过了多次的迭代更新，比如多层感知器，在卷积神经网络广泛使用之前，多层感知器在图像识别上是非常流行的，从这方面来看，多层感知器在当时也是有一定的优势的。那么如下使用PaddlePaddle来定义一个多层感知器呢，我们可以来学习一下。以下的代码判断就是定义一个简单的多层感知器，一共有三层，两个大小为100的隐层和一个大小为10的输出层，因为MNIST数据集是手写0到9的灰度图像，类别有10个，所以最后的输出大小是10。最后输出层的激活函数是Softmax，所以最后的输出层相当于一个分类器。加上一个输入层的话，多层感知器的结构是：`输入层-->>隐层-->>隐层-->>输出层`。
```python
# 定义多层感知器
def multilayer_perceptron(input):
    # 第一个全连接层，激活函数为ReLU
    hidden1 = fluid.layers.fc(input=input, size=100, act='relu')
    # 第二个全连接层，激活函数为ReLU
    hidden2 = fluid.layers.fc(input=hidden1, size=100, act='relu')
    # 以softmax为激活函数的全连接输出层，大小为label大小
    fc = fluid.layers.fc(input=hidden2, size=10, act='softmax')
    return fc
```

卷积神经网络普遍用在图像特征提取上，一些图像分类、目标检测、文字识别几乎都回使用到卷积神经网络作为图像的特征提取方式。卷积神经网络通常由卷积层、池化层和全连接层，有时还有Batch Normalization层和Dropout层。下面我们就创建一个简单卷积神经网络，一共定义了5层，加上输入层的话，它的结构是：`输入层-->>卷积层-->>池化层-->>卷积层-->>池化层-->>输出层`。我们可以通过调用PaddlePaddle的接口`fluid.layers.conv2d()`来做一次卷积操作，我们可以通过`num_filters`参数设置卷积核的数量，通过`filter_size`设置卷积核的大小，还有通过`stride`来设置卷积操作时移动的步长。使用`fluid.layers.pool2d()`接口做一次池化操作，通过参数`pool_size`可以设置池化的大小，通过参数`pool_stride`设置池化滑动的步长，通过参数`pool_type`设置池化的类型，目前有最大池化和平均池化，下面使用的时最大池化，当值为`avg`时是平均池化。
```python
# 卷积神经网络
def convolutional_neural_network(input):
    # 第一个卷积层，卷积核大小为3*3，一共有32个卷积核
    conv1 = fluid.layers.conv2d(input=input,
                                num_filters=32,
                                filter_size=3,
                                stride=1)

    # 第一个池化层，池化大小为2*2，步长为1，最大池化
    pool1 = fluid.layers.pool2d(input=conv1,
                                pool_size=2,
                                pool_stride=1,
                                pool_type='max')

    # 第二个卷积层，卷积核大小为3*3，一共有64个卷积核
    conv2 = fluid.layers.conv2d(input=pool1,
                                num_filters=64,
                                filter_size=3,
                                stride=1)

    # 第二个池化层，池化大小为2*2，步长为1，最大池化
    pool2 = fluid.layers.pool2d(input=conv2,
                                pool_size=2,
                                pool_stride=1,
                                pool_type='max')

    # 以softmax为激活函数的全连接输出层，大小为label大小
    fc = fluid.layers.fc(input=pool2, size=10, act='softmax')
    return fc
```

定义输入层，输入的是图像数据。图像是`28*28`的灰度图，所以输入的形状是`[1, 28, 28]`，如果图像是`32*32`的彩色图，那么输入的形状是`[3. 32, 32]`，因为灰度图只有一个通道，而彩色图有RGB三个通道。理论上它还有一个维度是Batch的，不过这个是PaddlePaddle帮我们默认设置的，我们可以不用理会。
```python
# 定义输入层
image = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
```

上面定义了多层感机器和卷积神经网络，我们可以在这里调用定义好的网络来获取分类器，读者可以尝试这两种不同的网络进行训练，观察一下他们的准确率如何。
```python
# 获取分类器
# model = multilayer_perceptron(image)
model = convolutional_neural_network(image)
```

接着是定义损失函数，这次使用的是交叉熵损失函数，该函数在分类任务上比较常用。定义了一个损失函数之后，还有对它求平均值，因为定义的是一个Batch的损失值。同时我们还可以定义一个准确率函数，这个可以在我们训练的时候输出分类的准确率。
```python
# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)
```

然后我们从主程序中克隆一个程序作为预测程序，之后可以使用这个预测程序预测测试的准确率和预测自己的图像。
```python
# 获取测试程序
test_program = fluid.default_main_program().clone(for_test=True)
```

接着是定义优化方法，这次我们使用的是Adam优化方法，同时指定学习率为0.001。
```python
# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)
```

定义读取MNIST数据集的reader，指定一个Batch的大小为128，也就是一次训练128张图像。
```python
# 获取MNIST数据
train_reader = paddle.batch(mnist.train(), batch_size=128)
test_reader = paddle.batch(mnist.test(), batch_size=128)
```

接着也是定义一个执行器和初始化参数，Fluid版本使用的流程都差不多。
```python
# 定义一个使用CPU的执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())
```

输入的数据维度是图像数据和图像对应的标签，每个类别的图像都要对应一个标签，这个标签是从0递增的整型数值。
```python
# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
```

最后就可以开始训练了，我们这次训练5个Pass，读者可以根据自己的情况自由设置。在上面我们已经定义了一个求准确率的函数，所以我们在训练的时候让它输出当前的准确率，计算准确率的原理很简单，就是把训练是预测的结果和真实的值比较，求出准确率。每一个Pass训练结束之后，再进行一次测试，使用测试集进行测试，并求出当前的Cost和准确率的平均值。
```python
# 开始训练和测试
for pass_id in range(5):
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

输出信息：
```
Pass:0, Batch:0, Cost:3.50138, Accuracy:0.07812
Pass:0, Batch:100, Cost:0.14832, Accuracy:0.96875
Pass:0, Batch:200, Cost:0.13408, Accuracy:0.96875
Pass:0, Batch:300, Cost:0.11601, Accuracy:0.97656
Pass:0, Batch:400, Cost:0.27977, Accuracy:0.92969
Test:0, Cost:0.08879, Accuracy:0.97379
Pass:1, Batch:0, Cost:0.11175, Accuracy:0.96875
Pass:1, Batch:100, Cost:0.07854, Accuracy:0.97656
Pass:1, Batch:200, Cost:0.04025, Accuracy:0.99219
Pass:1, Batch:300, Cost:0.09936, Accuracy:0.98438
Pass:1, Batch:400, Cost:0.19245, Accuracy:0.95312
Test:1, Cost:0.10123, Accuracy:0.97241
Pass:2, Batch:0, Cost:0.13749, Accuracy:0.96094
Pass:2, Batch:100, Cost:0.06074, Accuracy:0.98438
Pass:2, Batch:200, Cost:0.01982, Accuracy:0.99219
Pass:2, Batch:300, Cost:0.06725, Accuracy:0.97656
Pass:2, Batch:400, Cost:0.10043, Accuracy:0.96875
Test:2, Cost:0.13354, Accuracy:0.96776
Pass:3, Batch:0, Cost:0.08895, Accuracy:0.98438
Pass:3, Batch:100, Cost:0.06339, Accuracy:0.96875
Pass:3, Batch:200, Cost:0.05107, Accuracy:0.98438
Pass:3, Batch:300, Cost:0.08062, Accuracy:0.97656
Pass:3, Batch:400, Cost:0.07631, Accuracy:0.96875
Test:3, Cost:0.11465, Accuracy:0.97449
Pass:4, Batch:0, Cost:0.01259, Accuracy:1.00000
Pass:4, Batch:100, Cost:0.01203, Accuracy:1.00000
Pass:4, Batch:200, Cost:0.08451, Accuracy:0.97656
Pass:4, Batch:300, Cost:0.16532, Accuracy:0.98438
Pass:4, Batch:400, Cost:0.09657, Accuracy:0.98438
Test:4, Cost:0.14624, Accuracy:0.97211
```

# 预测图像
训练完成之后，我们可以使用从主程序中克隆的`test_program`来预测我们自己的图像。再预测之前，要对图像进行预处理，处理方式要跟训练的时候一样。首先进行灰度化，然后压缩图像大小为`28*28`，接着将图像转换成一维向量，最后再对一维向量进行归一化处理。
```python
# 对图片进行预处理
def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = im / 255.0 * 2.0 - 1.0
    return im
```

我们从网上下载一张图像，并将它命名为`infer_3.png`。
```python
!wget https://github.com/yeyupiaoling/LearnPaddle2/blob/master/note4/infer_3.png?raw=true -O 'infer_3.png'
```

我们可以使用Matplotlib工具显示这张图像。
```python
img = Image.open('infer_3.png')
plt.imshow(img)
plt.show()
```

输出的图片：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181207115540368.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3,size_16,color_FFFFFF,t_70)

最后把图像转换成一维向量并进行预测，数据从`feed`中的`image`传入，`label`设置一个假的label值传进去。`fetch_list`的值是网络模型的最后一层分类器，所以输出的结果是10个标签的概率值，这些概率值的总和为1。
```python
# 加载数据并开始预测
img = load_image('./infer_3.png')
results = exe.run(program=test_program,
                  feed={'image': img, "label": np.array([[1]]).astype("int64")},
                  fetch_list=[model])
```

拿到每个标签的概率值之后，我们要获取概率最大的标签，并打印出来。
```python
# 获取概率最大的label
lab = np.argsort(results)
print("该图片的预测结果的label为: %d" % lab[0][0][-1])
```

输出信息：
```
该图片的预测结果的label为: 3
```

到处为止，本章就结束了。经过学完这一章节，是不是觉得PaddlePaddle非常好用呢，借助PaddlePaddle我们很容易就定义了一个卷积神经网络，并完成了图像分类的训练和预测。卷积神经网络在图像识别上发挥着巨大的作用，而在自然语言处理上，循环神经网络同样起着巨大的作用，我们下一章就学习一下循环神经网络。

同步到百度AI Studio平台：http://aistudio.baidu.com/aistudio/#/projectdetail/29346
同步到科赛网K-Lab平台：https://www.kesci.com/home/project/5bf8c998954d6e001066d780
项目代码GitHub地址：https://github.com/yeyupiaoling/LearnPaddle2/tree/master/note4

**注意：** 最新代码以GitHub上的为准

# 参考资料
1. https://blog.csdn.net/m_buddy/article/details/80224409
2. http://www.paddlepaddle.org/documentation/docs/zh/1.0/beginners_guide/quick_start/recognize_digits/README.cn.html

