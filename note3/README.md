@[TOC]

# 前言
在第二章，我们已经学习了如何使用PaddlePaddle来进行加法计算，从这个小小的例子中，我们掌握了PaddlePaddle的使用方式。在本章中，我们将介绍使用PaddlePaddle完成一个深度学习非常常见的入门例子——线性回归，我们将分别使用自定义数据集和使用PaddlePaddle提供的数据集接口来训练一个线性回归模型。

# 使用自定义数据
在这一部分，我们将介绍整个线性回归从定义网络到使用自定义的数据进行训练，最后验证我们网络的预测能力。

首先导入PaddlePaddle库和一些工具类库。
```python
import paddle.fluid as fluid
import paddle
import numpy as np
```

定义一个简单的线性网络，这个网络非常简单，结构是：`输出层-->>隐层-->>输出层`，这个网络一共有2层，因为输入层不算网络的层数。更具体的就是一个大小为100，激活函数是ReLU的全连接层和一个输出大小为1的全连接层，就这样构建了一个非常简单的网络。这里使用输入`fluid.layers.data()`定义的输入层类似`fluid.layers.create_tensor()`，也是有`name`属性，之后也是根据这个属性来填充数据的。这里定义输入层的形状为13，这是因为波士顿房价数据集的每条数据有13个属性，我们之后自定义的数据集也是为了符合这一个维度。
```python
# 定义一个简单的线性网络
x = fluid.layers.data(name='x', shape=[13], dtype='float32')
hidden = fluid.layers.fc(input=x, size=100, act='relu')
net = fluid.layers.fc(input=hidden, size=1, act=None)
```

接着定义神经网络的损失函数，这里同样使用了`fluid.layers.data()`这个接口，这个可以理解为数据对应的结果，上面`name`为`x`的`fluid.layers.data()`为属性数据。这里使用了平方差损失函数(square_error_cost)，PaddlePaddle提供了很多的损失函数的接口，比如交叉熵损失函数(cross_entropy)。因为本项目是一个线性回归任务，所以我们使用的是平方差损失函数。因为`fluid.layers.square_error_cost()`求的是一个Batch的损失值，所以我们还要对他求一个平均值。
```python
# 定义损失函数
y = fluid.layers.data(name='y', shape=[1], dtype='float32')
cost = fluid.layers.square_error_cost(input=net, label=y)
avg_cost = fluid.layers.mean(cost)
```

定义损失函数之后，可以在主程序（fluid.default_main_program）中克隆一个程序作为预测程序，用于训练完成之后使用这个预测程序进行预测数据。这个定义的顺序不能错，因为我们定义的网络结构，损失函数等等都是更加顺序记录到PaddlePaddle的主程序中的。主程序定义了神经网络模型，前向反向计算，以及优化算法对网络中可学习参数的更新，是我们整个程序的核心，这个是PaddlePaddle已经帮我们实现的了，我们只需注重网络的构建和训练即可。
```python
# 复制一个主程序，方便之后使用
test_program = fluid.default_main_program().clone(for_test=True)
```

接着是定义训练使用的优化方法，这里使用的是随机梯度下降优化方法。PaddlePaddle提供了大量的优化函数接口，除了本项目使用的随机梯度下降法（SGD），还有Momentum、Adagrad、Adagrad等等，读者可以更加自己项目的需求使用不同的优化方法。
```python
# 定义优化方法
optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01)
opts = optimizer.minimize(avg_cost)
```

然后是创建一个解析器，我们同样是使用CPU来进行训练。创建解析器之后，使用解析器来执行`fluid.default_startup_program()`初始化参数。
```python
# 创建一个使用CPU的执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())
```

我们使用numpy定义一组数据，这组数据的每一条数据有13个，这是因为我们在定义网络的输入层时，`shape`是13，但是每条数据的后面12个数据是没意义的，因为笔者全部都是使用0来填充，纯粹是为了符合数据的格式而已。这组数据是符合`y = 2 * x + 1`，但是程序是不知道的，我们之后使用这组数据进行训练，看看强大的神经网络是否能够训练出一个拟合这个函数的模型。最后定义了一个预测数据，是在训练完成，使用这个数据作为`x`输入，看是否能够预测于正确值相近结果。
```python
# 定义训练和测试数据
x_data = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                   [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                   [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                   [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                   [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).astype('float32')
y_data = np.array([[3.0], [5.0], [7.0], [9.0], [11.0]]).astype('float32')
test_data = np.array([[6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).astype('float32')
```

定义数据之后，我们就可以使用数据进行训练了。我们这次训练了10个pass，读者可根据情况设置更多的训练轮数，通常来说训练的次数越多，模型收敛的越好。同样我们使用的时`profram`是`fluid.default_main_program()`，`feed`中是在训练时把数据传入`fluid.layers.data()`定义的变量中，及那个键值对的`key`对用的就是`fluid.layers.data()`中的`name`的值。我们让训练过程中输出avg_cost的值。

在训练过程中，我们可以看到输出的损失值在不断减小，证明我们的模型在不断收敛。
```python
# 开始训练100个pass
for pass_id in range(10):
    train_cost = exe.run(program=fluid.default_main_program(),
                         feed={'x': x_data, 'y': y_data},
                         fetch_list=[avg_cost])
    print("Pass:%d, Cost:%0.5f" % (pass_id, train_cost[0]))
```

输出信息：
```
Pass:0, Cost:65.61024
Pass:1, Cost:26.62285
Pass:2, Cost:7.78299
Pass:3, Cost:0.59838
Pass:4, Cost:0.02781
Pass:5, Cost:0.02600
Pass:6, Cost:0.02548
Pass:7, Cost:0.02496
Pass:8, Cost:0.02446
Pass:9, Cost:0.02396
```

训练完成之后，我们使用上面克隆主程序得到的预测程序了预测我们刚才定义的预测数据。预测数据同样作为`x`在`feed`输入，在预测时，理论上是不用输入`y`的，但是要符合输入格式，我们模拟一个`y`的数据值，这个值并不会影响我们的预测结果。`fetch_list`的值，也就是我们执行预测之后要输出的结果，这是网络的最后一层，而不是平均损失函数（avg_cost），因为我们是想要预测程序输出预测结果。根据我们上面定义数据时，满足规律`y = 2 * x + 1`，所以当x为6时，y应该时13，最后输出的结果也是应该接近13的。
```python
# 开始预测
result = exe.run(program=test_program,
                 feed={'x': test_data, 'y': np.array([[0.0]]).astype('float32')},
                 fetch_list=[net])
print("当x为6.0时，y为：%0.5f:" % result[0][0][0])
```

输出信息：
```
当x为6.0时，y为：13.23651:
```

# 使用房价数据集训练
在这一部分，我们还是使用上面定义的网络结构，使用波士顿房价数据集进行训练。

在此之前，我们已经完整训练深度学习模型，并使用这个模型来进行预测。而上面使用的是我们自己定义的数据，而且这个数据非常小。PaddlePaddle提供了大量的数据集API，我们可使用这些API来使用一些比较常用的数据集，比如在深度学习中，线性回归最常用的是波士顿房价数据集（UCI Housing Data Set），`uci_housing`就是PaddlePaddle提供的一个波士顿房价数据集。

而且这次我们的数据集不是一下子全部都丢入到训练中，而已把它们分成一个个Batch的小数据集，而每个Batch的大小我们都可以通过`batch_size`进行设置，这个大小一般是2的N次方。这里定义了训练和测试两个数据集。
```python
import paddle.dataset.uci_housing as uci_housing
# 使用房价数据进行训练和测试
# 从paddle接口中获取房价数据集
train_reader = paddle.batch(reader=uci_housing.train(), batch_size=128)
test_reader = paddle.batch(reader=uci_housing.test(), batch_size=128)
```

接着定义数据的维度，在使用自定义数据的时候，我们是使用键值对的方式添加数据的，但是我们调用API来获取数据集时，已经是将属性数据和结果放在一个Batch中，如果再对数据拆分在训练进行填充，那就更麻烦了，所以PaddlePaddle提供了一个`fluid.DataFeeder()`这个接口，这里可以定义输入数据每个维度是属于哪一个`fluid.layers.data()`.
```python
# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
```

接下来我们是使用波士顿房价数据集来进行训练，在训练时，我们是通过一个循环迭代器把reader中的数据按照一个个Batch提取出来加入到训练中。加入训练时使用上面定义的数据维度feeder.feed()添加的。

当每一个Pass训练完成之后，都执行一次测试，测试与预测的作用不同，测试是为了使用于测试数据集预测并与真实结果对比，评估当前模型的好坏。因为测试集不属于训练集，所以测试集的预测结果的好坏能狗体现模型的泛化能力。
```python
# 开始训练和测试
for pass_id in range(10):
    # 开始训练并输出最后一个batch的损失值
    train_cost = 0
    for batch_id, data in enumerate(train_reader()):
        train_cost = exe.run(program=fluid.default_main_program(),
                             feed=feeder.feed(data),
                             fetch_list=[avg_cost])
    print("Pass:%d, Cost:%0.5f" % (pass_id, train_cost[0][0]))

    # 开始测试并输出最后一个batch的损失值
    test_cost = 0
    for batch_id, data in enumerate(test_reader()):
        test_cost = exe.run(program=fluid.default_main_program(),
                            feed=feeder.feed(data),
                            fetch_list=[avg_cost])
    print('Test:%d, Cost:%0.5f' % (pass_id, test_cost[0][0]))
```

输出信息：
```
Pass:0, Cost:35.61119
Test:0, Cost:92.18690
Pass:1, Cost:121.56089
Test:1, Cost:51.94175
Pass:2, Cost:44.66270
Test:2, Cost:34.46148
Pass:3, Cost:33.25787
Test:3, Cost:30.89449
Pass:4, Cost:29.12044
Test:4, Cost:28.29573
Pass:5, Cost:26.75469
Test:5, Cost:26.66773
Pass:6, Cost:24.98260
Test:6, Cost:25.11611
Pass:7, Cost:23.55230
Test:7, Cost:23.84240
Pass:8, Cost:22.41704
Test:8, Cost:22.65791
Pass:9, Cost:21.51291
Test:9, Cost:21.71775
```

到此为止，本章知识已经学完。本章我们学会了如何使用PaddlePaddle完成了深度学习入门的常见例子，相信读者经过学习本章之后，对深度学习和PaddlePaddle的使用有了非常深刻的了解，也恭喜读者正式加入到人工智能行列中，希望读者能够坚定信心，在自己喜欢的领域一直走下去。在下一章，我们将会介绍使用卷积神经网络进行训练MNIST图像数据集，相信下一章你更加喜欢深度学习的，准备学习下一章了吗。

同步到百度AI Studio平台：http://aistudio.baidu.com/#/projectdetail/29342
同步到科赛网K-Lab平台：https://www.kesci.com/home/project/5bf8c7a6954d6e001066d72c
项目代码GitHub地址：https://github.com/yeyupiaoling/LearnPaddle2/tree/master/note3

**注意：** 最新代码以GitHub上的为准

# 参考资料
1. http://www.paddlepaddle.org/documentation/docs/zh/1.0/beginners_guide/quick_start/fit_a_line/README.cn.html
