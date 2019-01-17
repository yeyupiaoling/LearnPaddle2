import os
import shutil
import paddle as paddle
import paddle.dataset.cifar as cifar
import paddle.fluid as fluid
import mobilenet_v2
from visualdl import LogWriter

log_writer = LogWriter(dir='log/', sync_cycle=10)

with log_writer.mode('train') as writer:
    train_cost_writer = writer.scalar('cost')
    train_acc_writer = writer.scalar('accuracy')
    histogram = writer.histogram('histogram', num_buckets=50)

with log_writer.mode('test') as writer:
    test_cost_writer = writer.scalar('cost')
    test_acc_writer = writer.scalar('accuracy')

# 定义输入层
image = fluid.layers.data(name='image', shape=[3, 32, 32], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# 获取分类器
model = mobilenet_v2.net(image, 10)

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
train_reader = paddle.batch(cifar.train10(), batch_size=32)
test_reader = paddle.batch(cifar.test10(), batch_size=32)

# 定义一个使用CPU的解析器
place = fluid.CUDAPlace(0)
# place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

# 定义日志的开始位置和获取参数名称
train_step = 0
test_step = 0
params_name = fluid.default_startup_program().global_block().all_parameters()[0].name

# 训练10次
for pass_id in range(10):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc, params = exe.run(program=fluid.default_main_program(),
                                                feed=feeder.feed(data),
                                                fetch_list=[avg_cost, acc, params_name])
        # 保存训练的日志数据
        train_step += 1
        train_cost_writer.add_record(train_step, train_cost[0])
        train_acc_writer.add_record(train_step, train_acc[0])
        histogram.add_record(train_step, params.flatten())

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
        # 保存测试的日志数据
        test_step += 1
        test_cost_writer.add_record(test_step, test_cost[0])
        test_acc_writer.add_record(test_step, test_acc[0])

        test_accs.append(test_acc[0])
        test_costs.append(test_cost[0])
    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))
    test_acc = (sum(test_accs) / len(test_accs))
    print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))
