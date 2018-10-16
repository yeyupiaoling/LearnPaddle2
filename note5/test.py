import paddle
import paddle.dataset.imdb as imdb
import paddle.fluid as fluid


# 定义长短期记忆网络
def lstm_net(data, input_dim):
    emb = fluid.layers.embedding(input=data, size=[input_dim, 128], is_sparse=True)

    fc1 = fluid.layers.fc(input=emb, size=512)
    lstm1, _ = fluid.layers.dynamic_lstm(input=fc1, size=512)

    fc2 = fluid.layers.sequence_pool(input=fc1, pool_type='max')
    lstm2 = fluid.layers.sequence_pool(input=lstm1, pool_type='max')

    out = fluid.layers.fc(input=[fc2, lstm2], size=2, act='softmax')
    return out


# 定义输入数据
words = fluid.layers.data(name='words', shape=[1], dtype='int64', lod_level=1)
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# 获取数据字典
word_dict = imdb.word_dict()
# 获取数据字典长度
dict_dim = len(word_dict)
# 获取长短期记忆网络
model = lstm_net(words, dict_dim)

# 获取损失函数和准确率
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

# 获取训练和预测程序
train_program = fluid.default_main_program()
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.Adagrad(learning_rate=0.002)
opt = optimizer.minimize(avg_cost)

# 创建一个使用CPU的接解析器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 获取训练和预测数据
train_reader = paddle.batch(imdb.train(word_dict), batch_size=128)
test_reader = paddle.batch(imdb.test(word_dict), batch_size=4)

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
    print('Pass:%d, Cost:%0.5f', (pass_id, train_cost[0][0]))

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
    print('Test:', pass_id, ', Cost:', test_cost, ', ACC:', test_acc)

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

# 预测获取预测结果
results = exe.run(program=test_program,
                  feed={'words': tensor_words},
                  fetch_list=[model])

# 打印每句话的正负面概率
for i, r in enumerate(results[0]):
    print("Predict probability of ", r[0], " to be positive and ", r[1], " to be negative for review \'",
          reviews_str[i], "\'")
