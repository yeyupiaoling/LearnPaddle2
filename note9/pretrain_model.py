import os
import shutil
import paddle as paddle
import paddle.dataset.flowers as flowers
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr


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


# 定义输入层
image = fluid.layers.data(name='image', shape=[3, 224, 224], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# 获取分类器
pool = resnet50(image)
# 停止梯度下降
pool.stop_gradient = True
# 由这里创建一个基本的主程序
base_model_program = fluid.default_main_program().clone()

# 这里再重新加载网络的分类器，大小为本项目的分类大小
model = fluid.layers.fc(input=pool, size=102, act='softmax')

# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=1e-3)
opts = optimizer.minimize(avg_cost)

# 获取flowers数据
train_reader = paddle.batch(flowers.train(), batch_size=16)

# 定义一个使用CPU的解析器
place = fluid.CUDAPlace(0)
# place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

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

# 保存参数模型
save_pretrain_model_path = 'models/pretrain_model/'
# 删除旧的模型文件
shutil.rmtree(save_pretrain_model_path, ignore_errors=True)
# 创建保持模型文件目录
os.makedirs(save_pretrain_model_path)
# 保存参数模型
fluid.io.save_params(executor=exe, dirname=save_pretrain_model_path)
