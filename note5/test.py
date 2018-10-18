import paddle
import paddle.dataset.imdb as imdb
import paddle.fluid as fluid
import numpy as np


def rnn_net(ipt, input_dim):
    emb = fluid.layers.embedding(input=ipt, size=[input_dim, 128], is_sparse=True)
    sentence = fluid.layers.fc(input=emb, size=128, act='tanh')

    rnn = fluid.layers.DynamicRNN()
    with rnn.block():
        word = rnn.step_input(sentence)
        prev = rnn.memory(shape=[512])
        hidden = fluid.layers.fc(input=[word, prev], size=512, act='relu')
        rnn.update_memory(prev, hidden)
        rnn.output(hidden)

    last = fluid.layers.sequence_last_step(rnn())
    out = fluid.layers.fc(input=last, size=2, act='softmax')
    return out


# 定义长短期记忆网络
def lstm_net(ipt, input_dim):
    # 以数据的IDs作为输入
    emb = fluid.layers.embedding(input=ipt, size=[input_dim, 128], is_sparse=True)

    # 第一个全连接层
    fc1 = fluid.layers.fc(input=emb, size=512)
    # 进行一个长短期记忆操作
    lstm1, _ = fluid.layers.dynamic_lstm(input=fc1, size=512)

    # 第一个最大序列池操作
    fc2 = fluid.layers.sequence_pool(input=fc1, pool_type='max')
    # 第二个最大序列池操作
    lstm2 = fluid.layers.sequence_pool(input=lstm1, pool_type='max')

    # 以softmax作为全连接的输出层，大小为2,也就是正负面
    out = fluid.layers.fc(input=[fc2, lstm2], size=2, act='softmax')
    return out


def stacked_lstm_net(data, input_dim):
    emb = fluid.layers.embedding(
        input=data, size=[input_dim, 128], is_sparse=True)

    fc1 = fluid.layers.fc(input=emb, size=512)
    lstm1, cell1 = fluid.layers.dynamic_lstm(input=fc1, size=512)

    inputs = [fc1, lstm1]

    for i in range(2, 3 + 1):
        fc = fluid.layers.fc(input=inputs, size=512)
        lstm, cell = fluid.layers.dynamic_lstm(
            input=fc, size=512, is_reverse=(i % 2) == 0)
        inputs = [fc, lstm]

    fc_last = fluid.layers.sequence_pool(input=inputs[0], pool_type='max')
    lstm_last = fluid.layers.sequence_pool(input=inputs[1], pool_type='max')

    prediction = fluid.layers.fc(
        input=[fc_last, lstm_last], size=2, act='softmax')
    return prediction

# 定义输入数据， lod_level不为0指定输入数据为序列数据
words = fluid.layers.data(name='words', shape=[1], dtype='int64', lod_level=1)
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# 获取数据字典
print("加载数据字典中...")
word_dict = imdb.word_dict()
# 获取数据字典长度
dict_dim = len(word_dict)
# 获取长短期记忆网络
# model = lstm_net(words, dict_dim)
# model = rnn_net(words, dict_dim)
model = stacked_lstm_net(words, dict_dim)

# 获取损失函数和准确率
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

# 获取训练和预测程序
train_program = fluid.default_main_program()
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.Adagrad(learning_rate=0.001)
opt = optimizer.minimize(avg_cost)

# 创建一个解析器
place = fluid.CPUPlace()
# place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 获取训练和预测数据
print("加载训练数据中...")
train_reader = paddle.batch(
    paddle.reader.shuffle(imdb.train(word_dict), 25000), batch_size=128)
print("加载测试数据中...")
test_reader = paddle.batch(imdb.test(word_dict), batch_size=128)

# 定义输入数据的维度
feeder = fluid.DataFeeder(place=place, feed_list=[words, label])

# 开始训练
for pass_id in range(100):
    # 进行训练
    train_cost = 0
    for batch_id, data in enumerate(train_reader()):
        train_cost = exe.run(program=train_program,
                             feed=feeder.feed(data),
                             fetch_list=[cost])
    print('Pass:%d, Cost:%0.5f' % (pass_id, train_cost[0][0]))

    # 进行测试
    test_costs = []
    test_accs = []
    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=test_program,
                                      feed=feeder.feed(data),
                                      fetch_list=[cost, acc])
        test_costs.append(test_cost[0][0])
        test_accs.append(test_acc[0])
    # 计算平均预测损失在和准确率
    test_cost = (sum(test_costs) / len(test_costs))
    test_acc = (sum(test_accs) / len(test_accs))
    print('Test:%d, Cost:%0.5f, ACC:%0.5f\n' % (pass_id, test_cost, test_acc))

    # 定义预测数据
    reviews_str = ['read the book forget the movie', 'this is a great movie', 'this is very bad']
    # 把每个句子拆成一个个单词
    reviews = [c.split() for c in reviews_str]

    # 获取结束符号的标签
    UNK = word_dict['<unk>']
    # 获取每句话对应的标签
    lod = []
    for c in reviews:
        lod.append([word_dict.get(words, UNK) for words in c])

    # 获取每句话的单词数量
    base_shape = [[len(c) for c in lod]]

    # 生成预测数据
    tensor_words = fluid.create_lod_tensor(lod, base_shape, place)

    # 预测获取预测结果,因为输入的是3个数据，所以要模拟3个label的输入
    results = exe.run(program=test_program,
                      feed={'words': tensor_words, "label": np.array([[0], [0], [0]]).astype("int64")},
                      fetch_list=[model])

    # 打印每句话的正负面概率
    for i, r in enumerate(results[0]):
        print("\'%s\'的预测结果为：正面概率为：%0.5f，负面概率为：%0.5f\n" % (reviews_str[i], r[0], r[1]))
