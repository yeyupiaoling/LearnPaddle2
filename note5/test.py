import paddle
import paddle.dataset.imdb as imdb
import paddle.fluid as fluid

CLASS_DIM = 2
EMB_DIM = 128
HID_DIM = 512


def lstm_net(data, input_dim, class_dim, emb_dim, hid_dim):
    emb = fluid.layers.embedding(input=data, size=[input_dim, emb_dim], is_sparse=True)

    fc1 = fluid.layers.fc(input=emb, size=hid_dim)
    lstm1, _ = fluid.layers.dynamic_lstm(input=fc1, size=hid_dim)

    fc2 = fluid.layers.sequence_pool(input=fc1, pool_type='max')
    lstm2 = fluid.layers.sequence_pool(input=lstm1, pool_type='max')

    out = fluid.layers.fc(input=[fc2, lstm2], size=class_dim, act='softmax')
    return out


words = fluid.layers.data(name='words', shape=[1], dtype='int64', lod_level=1)
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

word_dict = imdb.word_dict()
print(word_dict)
dict_dim = len(word_dict)
model = lstm_net(words, dict_dim, CLASS_DIM, EMB_DIM, HID_DIM)

cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

train_program = fluid.default_main_program()
test_program = fluid.default_main_program().clone(for_test=True)

optimizer = fluid.optimizer.Adagrad(learning_rate=0.002)
opt = optimizer.minimize(avg_cost)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

train_reader = paddle.batch(imdb.train(word_dict), batch_size=128)
test_reader = paddle.batch(imdb.test(word_dict), batch_size=4)

feeder = fluid.DataFeeder(place=place, feed_list=[words, label])

for pass_id in range(100):
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc = exe.run(program=train_program,
                                        feed=feeder.feed(data),
                                        fetch_list=[cost, acc])
        print(train_acc)
        if batch_id % 100 == 0:
            print('Pass:', pass_id, ', Batch:', batch_id, ', Cost:',
                  train_cost[0][0], ', Accuracy:', train_acc[0])

    test_costs = []
    test_accs = []
    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=test_program,
                                      feed=feeder.feed(data),
                                      fetch_list=[cost, acc])
        test_costs.append(test_cost[0][0])
        test_accs.append(test_acc[0])
    test_cost = (sum(test_costs) / len(test_costs))
    test_acc = (sum(test_accs) / len(test_accs))
    print('Test:', pass_id, ', Cost:', test_cost, ', ACC:', test_acc)

reviews_str = ['read the book forget the movie', 'this is a great movie', 'this is very bad']
reviews = [c.split() for c in reviews_str]

UNK = word_dict['<unk>']
lod = []
for c in reviews:
    lod.append([word_dict.get(words, UNK) for words in c])

base_shape = [[len(c) for c in lod]]

tensor_words = fluid.create_lod_tensor(lod, base_shape, place)

results = exe.run(program=test_program,
                  feed={'words': tensor_words},
                  fetch_list=[model])

for i, r in enumerate(results[0]):
    print("Predict probability of ", r[0], " to be positive and ", r[1], " to be negative for review \'",
          reviews_str[i], "\'")
