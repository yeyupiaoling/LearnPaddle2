@[TOC]

# 前言
除了卷积神经网络，深度学习中还有循环神经网络也是很常用的，循环神经网络更常用于自然语言处理任务上。我们在这一章中，我们就来学习如何使用PaddlePaddle来实现一个循环神经网络，并使用该网络完成情感分析的模型训练。

# 训练模型
创建一个`text_classification.py`的Python文件。首先导入Python库，fluid和numpy库我们在前几章都有使用过，这里就不重复了。这里主要结束是imdb库，这个是一个数据集的库，这个是数据集是一个英文的电影评论数据集，每一条数据都会有两个分类，分别是正面和负面。
```python
import paddle
import paddle.dataset.imdb as imdb
import paddle.fluid as fluid
import numpy as np
```

循环神经网络发展到现在，已经有不少性能很好的升级版的循环神经网络，比如长短期记忆网络等。一下的代码片段是一个比较简单的循环神经网络，首先是经过一个`fluid.layers.embedding()`，这个是接口是接受数据的ID输入，因为输入数据时一个句子，但是在训练的时候我们是把每个单词转换成对应的ID，再输入到网络中，所以这里使用到了`embedding`接口。然后是一个全连接层，接着是一个循环神经网络块，在循环神经网络块之后再经过一个`sequence_last_step`接口，这个接口通常是使用在序列函数的最后一步。最后的输出层的激活函数是Softmax，大小为2，因为数据的结果有2个，为正负面。
```python
def rnn_net(ipt, input_dim):
    # 以数据的IDs作为输入
    emb = fluid.layers.embedding(input=ipt, size=[input_dim, 128], is_sparse=True)
    sentence = fluid.layers.fc(input=emb, size=128, act='tanh')

    rnn = fluid.layers.DynamicRNN()
    with rnn.block():
        word = rnn.step_input(sentence)
        prev = rnn.memory(shape=[128])
        hidden = fluid.layers.fc(input=[word, prev], size=128, act='relu')
        rnn.update_memory(prev, hidden)
        rnn.output(hidden)

    last = fluid.layers.sequence_last_step(rnn())
    out = fluid.layers.fc(input=last, size=2, act='softmax')
    return out
```

下面的代码片段是一个简单的长短期记忆网络，这个网络是有循环神经网络演化过来的。当较长的序列数据，循环神经网络的训练过程中容易出现梯度消失或爆炸现象，而长短期记忆网络就可以解决这个问题。在网络的开始同样是经过一个`embedding`接口，接着是一个全连接层，紧接的是一个`dynamic_lstm`长短期记忆操作接口，有这个接口，我们很容易就搭建一个长短期记忆网络。然后是经过两个序列池操作，该序列池的类型是最大化。最后也是一个大小为2的输出层。
```python
# 定义长短期记忆网络
def lstm_net(ipt, input_dim):
    # 以数据的IDs作为输入
    emb = fluid.layers.embedding(input=ipt, size=[input_dim, 128], is_sparse=True)

    # 第一个全连接层
    fc1 = fluid.layers.fc(input=emb, size=128)
    # 进行一个长短期记忆操作
    lstm1, _ = fluid.layers.dynamic_lstm(input=fc1, size=128)

    # 第一个最大序列池操作
    fc2 = fluid.layers.sequence_pool(input=fc1, pool_type='max')
    # 第二个最大序列池操作
    lstm2 = fluid.layers.sequence_pool(input=lstm1, pool_type='max')

    # 以softmax作为全连接的输出层，大小为2,也就是正负面
    out = fluid.layers.fc(input=[fc2, lstm2], size=2, act='softmax')
    return out
```

这里可以先定义一个输入层，这样要注意的是我们使用的数据属于序列数据，所以我们可以设置`lod_level`为1，当该参数不为0时，表示输入的数据为序列数据，默认`lod_level`的值是0.
```python
# 定义输入数据， lod_level不为0指定输入数据为序列数据
words = fluid.layers.data(name='words', shape=[1], dtype='int64', lod_level=1)
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
```

然后是读取数据字典，因为我们的数据是以数据标签的放方式表示数据一个句子。所以每个句子都是以一串整数来表示的，每个数字都是对应一个单词。所以这个数据集就会有一个数据集字典，这个字典是训练数据中出现单词对应的数字标签。
```python
# 获取数据字典
print("加载数据字典中...")
word_dict = imdb.word_dict()
# 获取数据字典长度
dict_dim = len(word_dict)
```

输出信息：
```
加载数据字典中...
```

这里可以获取我们上面定义的网络作为我们之后训练的网络模型，这两个网络读者都可以试试，可以对比它们的差别。
```python
# 获取长短期记忆网络
model = lstm_net(words, dict_dim)
# 获取循环神经网络
# model = rnn_net(words, dict_dim)
```

接着定义损失函数，这里同样是一个分类任务，所以使用的损失函数也是交叉熵损失函数。这里也可以使用`fluid.layers.accuracy()`接口定义一个输出分类准确率的函数，可以方便在训练的时候，输出测试时的分类准确率，观察模型收敛的情况。
```python
# 获取损失函数和准确率
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)
```

这里克隆一个测试测试程序，用于之后的测试和预测数据使用的。
```python
# 获取预测程序
test_program = fluid.default_main_program().clone(for_test=True)
```

然后是定义优化方法，这里使用的时Adagrad优化方法，Adagrad优化方法多用于处理稀疏数据，设置学习率为0.002。
```python
# 定义优化方法
optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.002)
opt = optimizer.minimize(avg_cost)
```

接着创建一个执行器，这次是的数据集比之前使用的数据集要大不少，所以训练起来先对比较慢，如果读取有GPU环境，可以尝试使用GPU来训练，使用方式是使用`fluid.CUDAPlace(0)`来创建执行器。
```python
# 创建一个执行器
place = fluid.CPUPlace()
# place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())
```

然后把训练数据和测试数据读取到内存中，因为数据集比较大，为了加快数据的数据，使用`paddle.reader.shuffle()`接口来将数据先按照设置的大小读取到缓存中。读入缓存的大小可以根据硬件环境内存大小来设置。
```python
# 获取训练和预测数据
print("加载训练数据中...")
train_reader = paddle.batch(paddle.reader.shuffle(imdb.train(word_dict), 25000), batch_size=128)
print("加载测试数据中...")
test_reader = paddle.batch(imdb.test(word_dict), batch_size=128)
```

输出信息：
```
加载训练数据中...

加载测试数据中...
```

定义数据数据的维度，数据的顺序是一条句子数据对应一个标签。
```python
# 定义输入数据的维度
feeder = fluid.DataFeeder(place=place, feed_list=[words, label])
```

现在就可以开始训练了，这里设置训练的循环是1次，读者可以根据情况设置更多的训练轮数，来让模型完全收敛。我们在训练中，每40个Batch打印一层训练信息和进行一次测试，测试是使用测试集进行预测并输出损失值和准确率，测试完成之后，对之前预测的结果进行求平均值。
```python
# 开始训练
for pass_id in range(1):
    # 进行训练
    train_cost = 0
    for batch_id, data in enumerate(train_reader()):
        train_cost = exe.run(program=fluid.default_main_program(),
                             feed=feeder.feed(data),
                             fetch_list=[avg_cost])

        if batch_id % 40 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f' % (pass_id, batch_id, train_cost[0]))
            # 进行测试
            test_costs = []
            test_accs = []
            for batch_id, data in enumerate(test_reader()):
                test_cost, test_acc = exe.run(program=test_program,
                                              feed=feeder.feed(data),
                                              fetch_list=[avg_cost, acc])
                test_costs.append(test_cost[0])
                test_accs.append(test_acc[0])
            # 计算平均预测损失在和准确率
            test_cost = (sum(test_costs) / len(test_costs))
            test_acc = (sum(test_accs) / len(test_accs))
            print('Test:%d, Cost:%0.5f, ACC:%0.5f' % (pass_id, test_cost, test_acc))
```

输出信息：
```
Pass:0, Batch:0, Cost:0.69274
Test:0, Cost:0.69329, ACC:0.50175
Pass:0, Batch:40, Cost:0.61183
Test:0, Cost:0.61142, ACC:0.82659
Pass:0, Batch:80, Cost:0.55504
Test:0, Cost:0.54904, ACC:0.83959
Pass:0, Batch:120, Cost:0.51100
Test:0, Cost:0.50026, ACC:0.84318
Pass:0, Batch:160, Cost:0.46800
Test:0, Cost:0.46199, ACC:0.84533
```


# 预测数据

我们先定义三个句子，第一句是中性的，第二句偏向正面，第三句偏向负面。然后把这些句子读取到一个列表中。
```python
# 定义预测数据
reviews_str = ['read the book forget the movie', 'this is a great movie', 'this is very bad']
# 把每个句子拆成一个个单词
reviews = [c.split() for c in reviews_str]
```

然后把句子转换成编码，根据数据集的字典，把句子中的单词转换成对应标签。
```python
# 获取结束符号的标签
UNK = word_dict['<unk>']
# 获取每句话对应的标签
lod = []
for c in reviews:
    # 需要把单词进行字符串编码转换
    lod.append([word_dict.get(words.encode('utf-8'), UNK) for words in c])
```

获取输入数据的维度和大小。
```python
# 获取每句话的单词数量
base_shape = [[len(c) for c in lod]]
```

将要预测的数据转换成张量，准备开始预测。
```python
# 生成预测数据
tensor_words = fluid.create_lod_tensor(lod, base_shape, place)
```

开始预测，使用的`program`是克隆的测试程序。预测数据是通过`feed`键值对的方式传入到预测程序中，为了符合输入数据的格式，label中使用了一个假的label输入到程序中。`fetch_list`的值是网络的分类器。
```python
# 预测获取预测结果,因为输入的是3个数据，所以要模拟3个label的输入
results = exe.run(program=test_program,
                  feed={'words': tensor_words, 'label': np.array([[0], [0], [0]]).astype('int64')},
                  fetch_list=[model])
```

最后可以把预测结果输出，因为我们使用了3条数据进行预测，所以输出也会有3个结果。每个结果是类别的概率。
```python
# 打印每句话的正负面概率
for i, r in enumerate(results[0]):
    print("\'%s\'的预测结果为：正面概率为：%0.5f，负面概率为：%0.5f" % (reviews_str[i], r[0], r[1]))
```

输出信息：
```
'read the book forget the movie'的预测结果为：正面概率为：0.53604，负面概率为：0.46396
'this is a great movie'的预测结果为：正面概率为：0.67564，负面概率为：0.32436
'this is very bad'的预测结果为：正面概率为：0.35406，负面概率为：0.64594
```

到处为止，本章就结束了。希望读者经过学习完这一章，可以对PaddlePaddle的使用有更深一步的认识。在下一章中，我们来使用PaddlePaddle实现一个生成对抗网络，生成对抗网络这一两年中可以说时非常火的，同样也非长有趣。那么我们下一章见吧。


同步到百度AI Studio平台：http://aistudio.baidu.com/aistudio/#/projectdetail/29347
同步到科赛网K-Lab平台：https://www.kesci.com/home/project/5bf8cb78954d6e001066d7d8
项目代码GitHub地址：https://github.com/yeyupiaoling/LearnPaddle2/tree/master/note5

**注意：** 最新代码以GitHub上的为准

# 参考资料
1. https://blog.csdn.net/u010089444/article/details/76725843
2. http://ai.stanford.edu/~amaas/data/sentiment/
3. https://github.com/PaddlePaddle/book/tree/develop/06.understand_sentiment
