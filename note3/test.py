import paddle.fluid as fluid
import paddle
import paddle.dataset.uci_housing as uci_housing
import numpy as np

# 定义一个简单的线性网络
x = fluid.layers.data(name='x', shape=[1], dtype='float32')
# x = fluid.layers.data(name='x', shape=[13], dtype='float32')
hidden = fluid.layers.fc(input=x, size=100, act='relu')
net = fluid.layers.fc(input=hidden, size=1, act=None)

# 定义损失函数
y = fluid.layers.data(name='y', shape=[1], dtype='float32')
cost = fluid.layers.square_error_cost(input=net, label=y)
avg_cost = fluid.layers.mean(cost)

# 复制一个主程序，方便之后使用
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01)
opts = optimizer.minimize(avg_cost)

# 创建一个使用CPU的解释器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 定义训练和测试数据
x_data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]).astype('float32')
y_data = np.array([[3.0], [5.0], [7.0], [9.0], [11.0]]).astype('float32')
test_data = np.array([[6.0]]).astype('float32')

# 开始训练100个pass
for pass_id in range(100):
    train_cost = exe.run(program=fluid.default_main_program(),
                         feed={'x': x_data, 'y': y_data},
                         fetch_list=[avg_cost])
    print("Pass:%d, Cost:%0.5f" % (pass_id, train_cost[0]))

# 开始预测
result = exe.run(program=test_program,
                 feed={'x': test_data, 'y': np.array([[0.0]]).astype('float32')},
                 fetch_list=[net])
print("当x为6.0时，y为：%0.5f:" % result[0][0][0])

# 使用房价数据进行训练和测试
# 从paddle接口中获取房价数据集
# train_reader = paddle.batch(reader=uci_housing.train(), batch_size=128)
# test_reader = paddle.batch(reader=uci_housing.test(), batch_size=128)
#
# # 定义输入数据维度
# feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
#
# # 开始训练和测试
# for pass_id in range(100):
#     # 开始训练并输出最后一个batch的损失值
#     train_cost = 0
#     for batch_id, data in enumerate(train_reader()):
#         train_cost = exe.run(program=fluid.default_main_program(),
#                              feed=feeder.feed(data),
#                              fetch_list=[avg_cost])
#     print("Pass:%d, Cost:%0.5f" % (pass_id, train_cost[0][0]))
#
#     # 开始测试并输出最后一个batch的损失值
#     test_cost = 0
#     for batch_id, data in enumerate(test_reader()):
#         test_cost = exe.run(program=fluid.default_main_program(),
#                             feed=feeder.feed(data),
#                             fetch_list=[avg_cost])
#     print('Test:%d, Cost:%0.5f' % (pass_id, test_cost[0][0]))
