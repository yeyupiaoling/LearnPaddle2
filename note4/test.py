import numpy as np
import paddle as paddle
import paddle.dataset.mnist as mnist
import paddle.fluid as fluid
from PIL import Image


def multilayer_perceptron(input):
    # 第一个全连接层，激活函数为ReLU
    hidden1 = fluid.layers.fc(input=input, size=100, act='relu')
    # 第二个全连接层，激活函数为ReLU
    hidden2 = fluid.layers.fc(input=hidden1, size=100, act='relu')
    # 以softmax为激活函数的全连接输出层，输出层的大小必须为数字的个数10
    fc = fluid.layers.fc(input=hidden2, size=10, act='softmax')
    return fc


def convolutional_neural_network(input):
    conv1 = fluid.layers.conv2d(input=input,
                                num_filters=32,
                                filter_size=3,
                                stride=1,
                                act='relu')

    pool1 = fluid.layers.pool2d(input=conv1,
                                pool_size=2,
                                pool_stride=1,
                                pool_type='max')

    conv2 = fluid.layers.conv2d(input=pool1,
                                num_filters=64,
                                filter_size=3,
                                stride=1,
                                act='relu')

    pool2 = fluid.layers.pool2d(input=conv2,
                                pool_size=2,
                                pool_stride=1,
                                pool_type='max')

    fc = fluid.layers.fc(input=pool2, size=10, act='softmax')
    return fc


image = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

model = multilayer_perceptron(image)
# model = convolutional_neural_network(image)


cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

test_program = fluid.default_main_program().clone(for_test=True)
train_program = fluid.default_main_program()

optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)

train_reader = paddle.batch(mnist.train(), batch_size=128)
test_reader = paddle.batch(mnist.test(), batch_size=128)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

for pass_id in range(1):
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc = exe.run(program=train_program,
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, acc])
        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))

    test_accs = []
    test_costs = []
    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=test_program,
                                      feed=feeder.feed(data),
                                      fetch_list=[avg_cost, acc])
        test_accs.append(test_acc[0])
        test_costs.append(test_cost[0])
    test_cost = (sum(test_costs) / len(test_costs))
    test_acc = (sum(test_accs) / len(test_accs))
    print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))


def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = im / 255.0 * 2.0 - 1.0
    return im


img = load_image('./infer_3.png')

results = exe.run(program=test_program,
                  feed={'image': img, "label": np.array([[1]]).astype("int64")},
                  fetch_list=[model])

lab = np.argsort(results)
print("Inference result of infer_3.png is: %d" % lab[0][0][-1])
