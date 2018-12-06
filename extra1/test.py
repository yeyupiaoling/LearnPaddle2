import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
import os
from multiprocessing import cpu_count
import matplotlib.pyplot as plt

data_path = './TibetanMnist（350x350）'
data_imgs = os.listdir(data_path)

with open('./train_data.list', 'w') as f_train:
    with open('./test_data.list', 'w') as f_test:
        for i in range(len(data_imgs)):
            if data_imgs[i] == 'lable.txt':
                continue
            if i % 10 == 0:
                f_test.write(os.path.join(data_path, data_imgs[i]) + "\t" + data_imgs[i][0:1] + '\n')
            else:
                f_train.write(os.path.join(data_path, data_imgs[i]) + "\t" + data_imgs[i][0:1] + '\n')
        print('图像列表已生成。')


def train_mapper(sample):
    img, label = sample
    img = paddle.dataset.image.load_image(file=img, is_color=False)
    img = paddle.dataset.image.simple_transform(im=img, resize_size=32, crop_size=28, is_color=False, is_train=True)
    img = img.flatten().astype('float32') / 255.0
    return img, label


def train_r(train_list_path):
    def reader():
        with open(train_list_path, 'r') as f:
            lines = f.readlines()
            del lines[len(lines) - 1]
            for line in lines:
                img, label = line.split('\t')
                yield img, int(label)

    return paddle.reader.xmap_readers(train_mapper, reader, cpu_count(), 1024)


def test_mapper(sample):
    img, label = sample
    img = paddle.dataset.image.load_image(file=img, is_color=False)
    img = paddle.dataset.image.simple_transform(im=img, resize_size=32, crop_size=28, is_color=False, is_train=False)
    img = img.flatten().astype('float32') / 255.0
    return img, label


def test_r(test_list_path):
    def reader():
        with open(test_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img, label = line.split('\t')
                yield img, int(label)

    return paddle.reader.xmap_readers(test_mapper, reader, cpu_count(), 1024)


def cnn(ipt):
    conv1 = fluid.layers.conv2d(input=ipt,
                                num_filters=32,
                                filter_size=3,
                                padding=1,
                                stride=1,
                                name='conv1',
                                act='relu')

    pool1 = fluid.layers.pool2d(input=conv1,
                                pool_size=2,
                                pool_stride=2,
                                pool_type='max',
                                name='pool1')

    bn1 = fluid.layers.batch_norm(input=pool1, name='bn1')

    conv2 = fluid.layers.conv2d(input=bn1,
                                num_filters=64,
                                filter_size=3,
                                padding=1,
                                stride=1,
                                name='conv2',
                                act='relu')

    pool2 = fluid.layers.pool2d(input=conv2,
                                pool_size=2,
                                pool_stride=2,
                                pool_type='max',
                                name='pool2')

    bn2 = fluid.layers.batch_norm(input=pool2, name='bn2')

    fc1 = fluid.layers.fc(input=bn2, size=1024, act='relu', name='fc1')

    fc2 = fluid.layers.fc(input=fc1, size=10, act='softmax', name='fc2')

    return fc2

image = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')
net = cnn(image)

label = fluid.layers.data(name='label', shape=[1], dtype='int64')
cost = fluid.layers.cross_entropy(input=net, label=label)
avg_cost = fluid.layers.mean(x=cost)
acc = fluid.layers.accuracy(input=net, label=label, k=1)

test_program = fluid.default_main_program().clone(for_test=True)

optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
opt = optimizer.minimize(avg_cost)

place = fluid.CPUPlace()
exe = fluid.Executor(place=place)
exe.run(program=fluid.default_startup_program())

train_reader = paddle.batch(reader=paddle.reader.shuffle(reader=train_r('./train_data.list'), buf_size=3000), batch_size=128)
test_reader = paddle.batch(reader=test_r('./test_data.list'), batch_size=128)

feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

for pass_id in range(2):
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, acc])
        if batch_id % 100 == 0:
            print('\nPass：%d, Batch：%d, Cost：%f, Accuracy：%f' % (pass_id, batch_id, train_cost[0], train_acc[0]))
        else:
            print('.', end="")

    test_costs = []
    test_accs = []
    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=test_program,
                                      feed=feeder.feed(data),
                                      fetch_list=[avg_cost, acc])
        test_costs.append(test_cost[0])
        test_accs.append(test_acc[0])
    test_cost = sum(test_costs) / len(test_costs)
    test_acc = sum(test_accs) / len(test_accs)
    print('\nTest：%d, Cost：%f, Accuracy：%f' % (pass_id, test_cost, test_acc))

    fluid.io.save_inference_model(dirname='./model', feeded_var_names=['image'], target_vars=[net], executor=exe)

[infer_program, feeded_var_names, target_vars] = fluid.io.load_inference_model(dirname='./model', executor=exe)

def load_image(path):
    img = paddle.dataset.image.load_image(file=path, is_color=False)
    img = paddle.dataset.image.simple_transform(im=img, resize_size=32, crop_size=28, is_color=False, is_train=False)
    img = img.astype('float32')
    img = img[np.newaxis, ] / 255.0
    return img

infer_imgs = []
infer_imgs.append(load_image('./TibetanMnist（350x350）/0_10_398.jpg'))
infer_imgs = np.array(infer_imgs)

result = exe.run(program=infer_program,
                 feed={feeded_var_names[0]:infer_imgs},
                 fetch_list=target_vars)

lab = np.argsort(result)

im = Image.open('./TibetanMnist（350x350）/0_10_398.jpg')
plt.imshow(im)
plt.show()

print('预测结果为：%d' % lab[0][0][-1])



