@[TOC]

# 前言
在深度学习训练中，例如图像识别训练，每次从零开始训练都要消耗大量的时间和资源。而且当数据集比较少时，模型也难以拟合的情况。基于这种情况下，就出现了迁移学习，通过使用已经训练好的模型来初始化即将训练的网络，可以加快模型的收敛速度，而且还能提高模型的准确率。这个用于初始化训练网络的模型是使用大型数据集训练得到的一个模型，而且模型已经完全收敛。最好训练的模型和预训练的模型是同一个网络，这样可以最大限度地初始化全部层。


# 初步训练模型
本章使用的预训练模型是PaddlePaddle官方提供的ResNet50网络模型，训练的数据集是ImageNet，它的下载地址为：http://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_pretrained.zip ，读者可以下载其他更多的模型，可以在[这里下载](https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleCV/image_classification/README_cn.md#%E5%B7%B2%E6%9C%89%E6%A8%A1%E5%9E%8B%E5%8F%8A%E5%85%B6%E6%80%A7%E8%83%BD)。下载之后解压到models目录下。

编写一个`pretrain_model.py`的Python程序，用于初步训练模型。首先导入相关的依赖包。
```python
import os
import shutil
import paddle as paddle
import paddle.dataset.flowers as flowers
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
```

定义一个残差神经网络，这个网络是PaddlePaddle官方提供的，模型地址为[models_name](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification/models_name)。这个网络是在每一个层都由指定参数名字，这是为了方便初始化网络模型，如果网络的结构发生变化了，但是名字没有变化，之后使用预训练模型初始化时，就可以根据每个参数的名字初始化对应的层。
```python
# 定义残差神经网络（ResNet）
def resnet50(input):
    def conv_bn_layer(input, num_filters, filter_size, stride=1, groups=1, act=None, name=None):
        conv = fluid.layers.conv2d(input=input,
                                   num_filters=num_filters,
                                   filter_size=filter_size,
                                   stride=stride,
                                   padding=(filter_size - 1) // 2,
                                   groups=groups,
                                   act=None,
                                   param_attr=ParamAttr(name=name + "_weights"),
                                   bias_attr=False,
                                   name=name + '.conv2d.output.1')
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        return fluid.layers.batch_norm(input=conv,
                                       act=act,
                                       name=bn_name + '.output.1',
                                       param_attr=ParamAttr(name=bn_name + '_scale'),
                                       bias_attr=ParamAttr(bn_name + '_offset'),
                                       moving_mean_name=bn_name + '_mean',
                                       moving_variance_name=bn_name + '_variance', )

    def shortcut(input, ch_out, stride, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return conv_bn_layer(input, ch_out, 1, stride, name=name)
        else:
            return input

    def bottleneck_block(input, num_filters, stride, name):
        conv0 = conv_bn_layer(input=input,
                              num_filters=num_filters,
                              filter_size=1,
                              act='relu',
                              name=name + "_branch2a")
        conv1 = conv_bn_layer(input=conv0,
                              num_filters=num_filters,
                              filter_size=3,
                              stride=stride,
                              act='relu',
                              name=name + "_branch2b")
        conv2 = conv_bn_layer(input=conv1,
                              num_filters=num_filters * 4,
                              filter_size=1,
                              act=None,
                              name=name + "_branch2c")

        short = shortcut(input, num_filters * 4, stride, name=name + "_branch1")

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu', name=name + ".add.output.5")

    depth = [3, 4, 6, 3]
    num_filters = [64, 128, 256, 512]

    conv = conv_bn_layer(input=input, num_filters=64, filter_size=7, stride=2, act='relu', name="conv1")
    conv = fluid.layers.pool2d(input=conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

    for block in range(len(depth)):
        for i in range(depth[block]):
            conv_name = "res" + str(block + 2) + chr(97 + i)
            conv = bottleneck_block(input=conv,
                                    num_filters=num_filters[block],
                                    stride=2 if i == 0 and block != 0 else 1,
                                    name=conv_name)

    pool = fluid.layers.pool2d(input=conv, pool_size=7, pool_type='avg', global_pooling=True)
    return pool
```


定义图片数据和标签数据的输入层，本章使用的图片数据集是[flowers](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)。这个通过使用PaddlePaddle的接口得到的flowers数据集的图片是3通道宽高都是224的彩色图，总类别是102种。
```python
# 定义输入层
image = fluid.layers.data(name='image', shape=[3, 224, 224], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
```

获取一个基本的模型，并从主程序中克隆一个基本的程序，用于之后加载参数使用。
```python
# 获取分类器的上一层
pool = resnet50(image)
# 停止梯度下降
pool.stop_gradient = True
# 由这里创建一个基本的主程序
base_model_program = fluid.default_main_program().clone()
```

这里再加上网络的分类器，因为预训练模型的类别数量是1000，所以要重新修改分类器。这个也是训练新模型的最大不同点，通过分离分类器来解决两个数据集的不同类别的问题。
```python
# 这里再重新加载网络的分类器，大小为本项目的分类大小
model = fluid.layers.fc(input=pool, size=102, act='softmax')
```

然后是获取损失函数，准确率函数和优化方法。
```python
# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=1e-3)
opts = optimizer.minimize(avg_cost)
```


获取flowers数据集，因为这里不需要使用测试，所以这里也不需要读取测试数据集。
```python
# 获取flowers数据
train_reader = paddle.batch(flowers.train(), batch_size=16)
```


创建执行器，最好是使用GPU进行训练，因为数据集和网络都是比较大的。
```python
# 定义一个使用CPU的执行器
place = fluid.CUDAPlace(0)
# place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())
```


这里就是加载预训练模型的重点，通过if_exist函数判断网络所需的模型文件是否存在，然后再通过调用`fluid.io.load_vars`加载存在的模型文件。要留意的是这里使用的是之前克隆的基本程序。
```python
# 官方提供的原预训练模型
src_pretrain_model_path = 'models/ResNet50_pretrained/'


# 通过这个函数判断模型文件是否存在
def if_exist(var):
    path = os.path.join(src_pretrain_model_path, var.name)
    exist = os.path.exists(path)
    if exist:
        print('Load model: %s' % path)
    return exist


# 加载模型文件，只加载存在模型的模型文件
fluid.io.load_vars(executor=exe, dirname=src_pretrain_model_path, predicate=if_exist, main_program=base_model_program)

```

然后使用这个预训练模型进行训练10个Pass。
```python
# 优化内存
optimized = fluid.transpiler.memory_optimize(input_program=fluid.default_main_program(), print_log=False)

# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

# 训练10次
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
```

执行训练输出的信息：
```
Load model: models/ResNet50_pretrained/res5a_branch2a_weights
Load model: models/ResNet50_pretrained/res4c_branch2a_weights
Load model: models/ResNet50_pretrained/res4f_branch2b_weights
Load model: models/ResNet50_pretrained/bn2a_branch2b_variance
Load model: models/ResNet50_pretrained/bn4d_branch2b_variance
Load model: models/ResNet50_pretrained/bn4f_branch2b_variance
Load model: models/ResNet50_pretrained/bn4e_branch2a_offset
Load model: models/ResNet50_pretrained/res4f_branch2c_weights
Load model: models/ResNet50_pretrained/res5c_branch2b_weights
......
Pass:0, Batch:0, Cost:6.92118, Accuracy:0.00000
Pass:0, Batch:100, Cost:3.31085, Accuracy:0.31250
Pass:0, Batch:200, Cost:3.32227, Accuracy:0.18750
Pass:0, Batch:300, Cost:3.85708, Accuracy:0.31250
Pass:1, Batch:0, Cost:3.36264, Accuracy:0.25000
......
```

训练结束之后，使用`fluid.io.save_params`接口保存参数，这个是已经符合这个数据集类别数量的，所以之后会使用都这个模型直接初始化模型，不需要再分离分类器。
```python
# 保存参数模型
save_pretrain_model_path = 'models/pretrain_model/'
# 删除旧的模型文件
shutil.rmtree(save_pretrain_model_path, ignore_errors=True)
# 创建保持模型文件目录
os.makedirs(save_pretrain_model_path)
# 保存参数模型
fluid.io.save_params(executor=exe, dirname=save_pretrain_model_path)
```

到这里预训练的第一步处理原预训练模型算是完成了，接下来就是使用这个已经处理过的模型正式训练了。


# 使用过的模型开始正式训练
这一部分是使用已经处理过的模型开始正式训练，创建一个`train.py`正式训练。首先导入相关的依赖包。

```python
import os
import shutil
import paddle as paddle
import paddle.dataset.flowers as flowers
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
```


定义一个残差神经网络，这个残差神经网络跟上面的基本一样的，只是把分类器也加进去了，这是一个完整的神经网络。
```python
# 定义残差神经网络（ResNet）
def resnet50(input, class_dim):
    def conv_bn_layer(input, num_filters, filter_size, stride=1, groups=1, act=None, name=None):
        conv = fluid.layers.conv2d(input=input,
                                   num_filters=num_filters,
                                   filter_size=filter_size,
                                   stride=stride,
                                   padding=(filter_size - 1) // 2,
                                   groups=groups,
                                   act=None,
                                   param_attr=ParamAttr(name=name + "_weights"),
                                   bias_attr=False,
                                   name=name + '.conv2d.output.1')
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        return fluid.layers.batch_norm(input=conv,
                                       act=act,
                                       name=bn_name + '.output.1',
                                       param_attr=ParamAttr(name=bn_name + '_scale'),
                                       bias_attr=ParamAttr(bn_name + '_offset'),
                                       moving_mean_name=bn_name + '_mean',
                                       moving_variance_name=bn_name + '_variance', )

    def shortcut(input, ch_out, stride, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return conv_bn_layer(input, ch_out, 1, stride, name=name)
        else:
            return input

    def bottleneck_block(input, num_filters, stride, name):
        conv0 = conv_bn_layer(input=input,
                              num_filters=num_filters,
                              filter_size=1,
                              act='relu',
                              name=name + "_branch2a")
        conv1 = conv_bn_layer(input=conv0,
                              num_filters=num_filters,
                              filter_size=3,
                              stride=stride,
                              act='relu',
                              name=name + "_branch2b")
        conv2 = conv_bn_layer(input=conv1,
                              num_filters=num_filters * 4,
                              filter_size=1,
                              act=None,
                              name=name + "_branch2c")

        short = shortcut(input, num_filters * 4, stride, name=name + "_branch1")

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu', name=name + ".add.output.5")

    depth = [3, 4, 6, 3]
    num_filters = [64, 128, 256, 512]

    conv = conv_bn_layer(input=input, num_filters=64, filter_size=7, stride=2, act='relu', name="conv1")
    conv = fluid.layers.pool2d(input=conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

    for block in range(len(depth)):
        for i in range(depth[block]):
            conv_name = "res" + str(block + 2) + chr(97 + i)
            conv = bottleneck_block(input=conv,
                                    num_filters=num_filters[block],
                                    stride=2 if i == 0 and block != 0 else 1,
                                    name=conv_name)

    pool = fluid.layers.pool2d(input=conv, pool_size=7, pool_type='avg', global_pooling=True)
    output = fluid.layers.fc(input=pool, size=class_dim, act='softmax')
    return output
```


然后定义一系列所需的函数，输入层，神经网络的分类器，损失函数，准确率函数，优化方法，获取flowers训练数据和测试数据，并创建一个执行器。
```python
# 定义输入层
image = fluid.layers.data(name='image', shape=[3, 224, 224], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# 获取分类器
model = resnet50(image, 102)

# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

# 获取训练和测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=1e-3)
opts = optimizer.minimize(avg_cost)

# 获取MNIST数据
train_reader = paddle.batch(flowers.train(), batch_size=16)
test_reader = paddle.batch(flowers.test(), batch_size=16)

# 定义一个使用GPU的执行器
place = fluid.CUDAPlace(0)
# place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())
```

这里可以使用`fluid.io.load_params`接口加载已经处理过的预训练模型文件。
```python
# 经过处理的预训练预训练模型
pretrained_model_path = 'models/pretrain_model/'

# 加载经过处理的模型
fluid.io.load_params(executor=exe, dirname=pretrained_model_path)
```

之后就可以正常训练了，从训练输出的日志可以看出，模型收敛得非常快，而且准确率还非常高，如果没有使用预训练模型是很难达到这种准确率的。
```python
# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

# 训练10次
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

执行训练输出的信息：
```
Pass:0, Batch:0, Cost:0.11896, Accuracy:1.00000
Pass:0, Batch:100, Cost:1.73780, Accuracy:0.68750
Pass:0, Batch:200, Cost:1.32758, Accuracy:0.68750
Pass:0, Batch:300, Cost:1.56638, Accuracy:0.56250
Test:0, Cost:1.82441, Accuracy:0.53841
Pass:1, Batch:0, Cost:0.71874, Accuracy:0.87500
......
```

训练结束之后，可以保存预测模型用于之后的预测使用。
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

同步到百度AI Studio平台：http://aistudio.baidu.com/aistudio/#/projectdetail/38853
同步到科赛网K-Lab平台：https://www.kesci.com/home/project/5c3f495589f4aa002b845d6b
项目代码GitHub地址：https://github.com/yeyupiaoling/LearnPaddle2/tree/master/note9

**注意：** 最新代码以GitHub上的为准

# 参考资料
1. https://github.com/oraoto/learn_ml/blob/master/paddle/pretrained.ipynb
2. http://www.paddlepaddle.org/documentation/docs/zh/1.2/api_cn/io_cn.html

