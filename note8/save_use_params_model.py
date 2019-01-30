import os
import shutil

import paddle as paddle
import paddle.dataset.cifar as cifar
import paddle.fluid as fluid


# 定义VGG16神经网络
def vgg16(input, class_dim=1000):
    def conv_block(conv, num_filter, groups):
        for i in range(groups):
            conv = fluid.layers.conv2d(input=conv,
                                       num_filters=num_filter,
                                       filter_size=3,
                                       stride=1,
                                       padding=1,
                                       act='relu')
        return fluid.layers.pool2d(input=conv, pool_size=2, pool_type='max', pool_stride=2)

    conv1 = conv_block(input, 64, 2)
    conv2 = conv_block(conv1, 128, 2)
    conv3 = conv_block(conv2, 256, 3)
    conv4 = conv_block(conv3, 512, 3)
    conv5 = conv_block(conv4, 512, 3)

    fc1 = fluid.layers.fc(input=conv5, size=512)
    dp1 = fluid.layers.dropout(x=fc1, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=dp1, size=512)
    bn1 = fluid.layers.batch_norm(input=fc2, act='relu')
    fc2 = fluid.layers.dropout(x=bn1, dropout_prob=0.5)
    out = fluid.layers.fc(input=fc2, size=class_dim, act='softmax')

    return out


# 定义输入层
image = fluid.layers.data(name='image', shape=[3, 32, 32], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# 获取分类器
model = vgg16(image, 10)

# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

# 获取训练和测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=1e-3)
opts = optimizer.minimize(avg_cost)

# 获取CIFAR数据
train_reader = paddle.batch(cifar.train10(), batch_size=32)
test_reader = paddle.batch(cifar.test10(), batch_size=32)

# 定义一个使用CPU的执行器
place = fluid.CUDAPlace(0)
# place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 加载之前训练过的参数模型
save_path = 'models/params_model/'
if os.path.exists(save_path):
    print('使用参数模型作为预训练模型')
    fluid.io.load_params(executor=exe, dirname=save_path)

# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

# 开始训练和测试
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

    # 保存参数模型
    save_path = 'models/params_model/'
    # 删除旧的模型文件
    shutil.rmtree(save_path, ignore_errors=True)
    # 创建保持模型文件目录
    os.makedirs(save_path)
    # 保存参数模型
    fluid.io.save_params(executor=exe, dirname=save_path)
