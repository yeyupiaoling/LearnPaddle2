@[TOC]

# 前言
我们在第六章介绍了生成对抗网络，并使用生成对抗网络训练mnist数据集，生成手写数字图片。那么本章我们将使用对抗生成网络训练我们自己的图片数据集，并生成图片。在第六章中我们使用的黑白的单通道图片，在这一章中，我们使用的是3通道的彩色图。

GitHub地址：https://github.com/yeyupiaoling/LearnPaddle2/tree/master/note13

# 定义数据读取
我们首先创建一个`image_reader.py`文件，用于读取我们自己定义的图片数据集。首先导入所需的依赖包。
```python
import os
import random
from multiprocessing import cpu_count
import numpy as np
import paddle
from PIL import Image
```

这里的图片预处理主要是对图片进行等比例压缩和中心裁剪，这里为了避免图片在图片在resize时出现变形的情况，导致训练生成的图片不是我们真实图片的样子。这里为了增强数据集，做了随机水平翻转。最后在处理图片的时候，为了避免数据集中有单通道图片导致训练中断，所以还把单通道图转成3通道图片。
```python
# 测试图片的预处理
def train_mapper(sample):
    img, crop_size = sample
    img = Image.open(img)
    # 随机水平翻转
    r1 = random.random()
    if r1 > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # 等比例缩放和中心裁剪
    width = img.size[0]
    height = img.size[1]
    if width < height:
        ratio = width / crop_size
        width = width / ratio
        height = height / ratio
        img = img.resize((int(width), int(height)), Image.ANTIALIAS)
        height = height / 2
        crop_size2 = crop_size / 2
        box = (0, int(height - crop_size2), int(width), int(height + crop_size2))
    else:
        ratio = height / crop_size
        height = height / ratio
        width = width / ratio
        img = img.resize((int(width), int(height)), Image.ANTIALIAS)
        width = width / 2
        crop_size2 = crop_size / 2
        box = (int(width - crop_size2), 0, int(width + crop_size2), int(height))
    img = img.crop(box)
    img = img.resize((crop_size, crop_size), Image.ANTIALIAS)

    # 把单通道图变成3通道
    if len(img.getbands()) == 1:
        img1 = img2 = img3 = img
        img = Image.merge('RGB', (img1, img2, img3))

    # 转换成numpy值
    img = np.array(img).astype(np.float32)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = img[(2, 1, 0), :, :] / 255.0
    return img
```

在这篇文章中，我们读取数据集不需要使用到数据列表，因为我们并没有进行分类，只是把所有的图片用于训练并生成图片。所有这里只需要把文件中的所有图片都读取进行训练就 可以了。
```python
# 测试的图片reader
def train_reader(train_image_path, crop_size):
    pathss = []
    for root, dirs, files in os.walk(train_image_path):
        path = [os.path.join(root, name) for name in files]
        pathss.extend(path)

    def reader():
        for line in pathss:
            yield line, crop_size

    return paddle.reader.xmap_readers(train_mapper, reader, cpu_count(), 1024)
```

# 训练生成模型
下面创建`train.py`文件，用于训练对抗生成模型，并在训练过程中生成图片和保存预测模型。首先导入所需的依赖包。
```python
import os
import shutil
import numpy as np
import paddle
import paddle.fluid as fluid
import matplotlib.pyplot as plt
import image_reader
```

下面时定义生成器的，我们在第六章也介绍过。生成器的作用是尽可能生成满足判别器条件的图像。随着以上训练的进行，判别器不断增强自身的判别能力，而生成器也不断生成越来越逼真的图片，以欺骗判别器。生成器主要由两组全连接和BN层、两组转置卷积运算组成。唯一不同的时在生成器最后输出的大小是3，因为我们生成的图片是3通道的彩色图片，而且使用的激活函数是sigmoid，保证了输出的结果都是在0到1范围之内，这是彩色图片的颜色范围。
```python
# 训练的图片大小
image_size = 112

# 定义生成器
def Generator(y, name="G"):
    def deconv(x, num_filters, filter_size=5, stride=2, dilation=1, padding=2, output_size=None, act=None):
        return fluid.layers.conv2d_transpose(input=x,
                                             num_filters=num_filters,
                                             output_size=output_size,
                                             filter_size=filter_size,
                                             stride=stride,
                                             dilation=dilation,
                                             padding=padding,
                                             act=act)

    with fluid.unique_name.guard(name + "/"):
        # 第一组全连接和BN层
        y = fluid.layers.fc(y, size=2048)
        y = fluid.layers.batch_norm(y)
        # 第二组全连接和BN层
        y = fluid.layers.fc(y, size=int(128 * (image_size / 4) * (image_size / 4)))
        y = fluid.layers.batch_norm(y)
        # 进行形状变换
        y = fluid.layers.reshape(y, shape=[-1, 128, int((image_size / 4)), int((image_size / 4))])
        # 第一组转置卷积运算
        y = deconv(x=y, num_filters=128, act='relu', output_size=[int((image_size / 2)), int((image_size / 2))])
        # 第二组转置卷积运算
        y = deconv(x=y, num_filters=3, act='sigmoid', output_size=[image_size, image_size])
    return y
```

判别器的作用是训练真实的数据集，然后使用训练真实数据集模型去判别生成器生成的假图片。这一过程可以理解判别器为一个二分类问题，判别器在训练真实数据集时，尽量让其输出概率为1，而训练生成器生成的假图片输出概率为0。这样不断给生成器压力，让其生成的图片尽量逼近真实图片，以至于真实到连判别器也无法判断这是真实图像还是假图片。以下判别器由三组卷积池化层和一个最后全连接层组成，全连接层的大小为1，输入一个二分类的结果。
```python
# 判别器 Discriminator
def Discriminator(images, name="D"):
    # 定义一个卷积池化组
    def conv_pool(input, num_filters, act=None):
        return fluid.nets.simple_img_conv_pool(input=input,
                                               filter_size=3,
                                               num_filters=num_filters,
                                               pool_size=2,
                                               pool_stride=2,
                                               act=act)

    with fluid.unique_name.guard(name + "/"):
        y = fluid.layers.reshape(x=images, shape=[-1, 3, image_size, image_size])
        # 第一个卷积池化组
        y = conv_pool(input=y, num_filters=64, act='leaky_relu')
        # 第一个卷积池化加回归层
        y = conv_pool(input=y, num_filters=128)
        y = fluid.layers.batch_norm(input=y, act='leaky_relu')
        # 第二个卷积池化加回归层
        y = fluid.layers.fc(input=y, size=1024)
        y = fluid.layers.batch_norm(input=y, act='leaky_relu')
        # 最后一个分类器输出
        y = fluid.layers.fc(input=y, size=1, act='sigmoid')
    return y
```

然后在这里获取所需的程序，如判别器D识别生成器G生成的假图片程序，判别器D识别真实图片程序，生成器G生成符合判别器D的程序和初始化的程序。最后定义一个`get_params()`函数用于获取参数名称。
```python
# 创建判别器D识别生成器G生成的假图片程序
train_d_fake = fluid.Program()
# 创建判别器D识别真实图片程序
train_d_real = fluid.Program()
# 创建生成器G生成符合判别器D的程序
train_g = fluid.Program()

# 创建共同的一个初始化的程序
startup = fluid.Program()

# 噪声维度
z_dim = 100

# 从Program获取prefix开头的参数名字
def get_params(program, prefix):
    all_params = program.global_block().all_parameters()
    return [t.name for t in all_params if t.name.startswith(prefix)]
```

定义一个判别器识别真实图片的程序，这里判别器传入的数据是真实的图片数据，这里的输出图片是3通道的。这里使用的损失函数是fluid.layers.sigmoid_cross_entropy_with_logits()，这个损失函数是求它们在任务上的错误率，他们的类别是互不排斥的。所以无论真实图片的标签是什么，都不会影响模型识别为真实图片。这里更新的也只有判别器模型的参数，使用的优化方法是Adam。
```python
# 训练判别器D识别真实图片
with fluid.program_guard(train_d_real, startup):
    # 创建读取真实数据集图片的data，并且label为1
    real_image = fluid.layers.data('image', shape=[3, image_size, image_size])
    ones = fluid.layers.fill_constant_batch_size_like(real_image, shape=[-1, 1], dtype='float32', value=1)

    # 判别器D判断真实图片的概率
    p_real = Discriminator(real_image)
    # 获取损失函数
    real_cost = fluid.layers.sigmoid_cross_entropy_with_logits(p_real, ones)
    real_avg_cost = fluid.layers.mean(real_cost)

    # 获取判别器D的参数
    d_params = get_params(train_d_real, "D")

    # 创建优化方法
    optimizer = fluid.optimizer.Adam(learning_rate=2e-4)
    optimizer.minimize(real_avg_cost, parameter_list=d_params)
```

这里定义一个判别器识别生成器生成的图片的程序，这里是使用噪声的维度进行输入。这里判别器识别的是生成器生成的图片，这里使用的损失函数同样是fluid.layers.sigmoid_cross_entropy_with_logits()。这里更新的参数还是判别器模型的参数，也是使用Adam优化方法。
```python
# 训练判别器D识别生成器G生成的图片为假图片
with fluid.program_guard(train_d_fake, startup):
    # 利用创建假的图片data，并且label为0
    z = fluid.layers.data(name='z', shape=[z_dim])
    zeros = fluid.layers.fill_constant_batch_size_like(z, shape=[-1, 1], dtype='float32', value=0)

    # 判别器D判断假图片的概率
    p_fake = Discriminator(Generator(z))

    # 获取损失函数
    fake_cost = fluid.layers.sigmoid_cross_entropy_with_logits(p_fake, zeros)
    fake_avg_cost = fluid.layers.mean(fake_cost)

    # 获取判别器D的参数
    d_params = get_params(train_d_fake, "D")

    # 创建优化方法
    optimizer = fluid.optimizer.Adam(learning_rate=2e-4)
    optimizer.minimize(fake_avg_cost, parameter_list=d_params)
```

最后定义一个训练生成器生成图片的模型，这里也克隆一个预测程序，用于之后在训练的时候输出预测的图片。损失函数和优化方法都一样，但是要更新的参数是生成器的模型参。
```python
# 训练生成器G生成符合判别器D标准的假图片
fake = None
with fluid.program_guard(train_g, startup):
    # 噪声生成图片为真实图片的概率，Label为1
    z = fluid.layers.data(name='z', shape=[z_dim])
    ones = fluid.layers.fill_constant_batch_size_like(z, shape=[-1, 1], dtype='float32', value=1)

    # 生成图片
    fake = Generator(z)
    # 克隆预测程序
    infer_program = train_g.clone(for_test=True)

    # 生成符合判别器的假图片
    p = Discriminator(fake)

    # 获取损失函数
    g_cost = fluid.layers.sigmoid_cross_entropy_with_logits(p, ones)
    g_avg_cost = fluid.layers.mean(g_cost)

    # 获取G的参数
    g_params = get_params(train_g, "G")

    # 只训练G
    optimizer = fluid.optimizer.Adam(learning_rate=2e-4)
    optimizer.minimize(g_avg_cost, parameter_list=g_params)
```

这里创建一个可以生成训练噪声数据的reader函数。
```python
# 噪声生成
def z_reader():
    while True:
        yield np.random.uniform(-1.0, 1.0, (z_dim)).astype('float32')
```

这里定义一个保存在训练过程生成的图片，通过观察生成图片的情况，可以了解到训练的效果。
```python
# 保存图片
def show_image_grid(images):
    for i, image in enumerate(images):
        image = image.transpose((2, 1, 0))
        save_image_path = 'train_image'
        if not os.path.exists(save_image_path):
            os.makedirs(save_image_path)
        plt.imsave(os.path.join(save_image_path, "test_%d.png" % i), image)
```

这里就开始获取自定义的图片数据集，这里只需要把存放图片数据集的文件夹传进去就可以了。
```python
# 生成真实图片reader
mydata_generator = paddle.batch(reader=image_reader.train_reader('datasets', image_size), batch_size=32)
# 生成假图片的reader
z_generator = paddle.batch(z_reader, batch_size=32)()
test_z = np.array(next(z_generator))
```

接着获取执行器，准备进行训练，这里笔者建议最好使用GPU，因为CPU贼慢。
```python
# 创建执行器，最好使用GPU，CPU速度太慢了
# place = fluid.CPUPlace()
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
# 初始化参数
exe.run(startup)
```

最好就可以开始训练啦，我们可以在训练的时候输出训练的损失值。在训练每一个Pass之后又可以使用预测程序生成图片并进行保存到本地。
```python
# 开始训练
for pass_id in range(100):
    for i, real_image in enumerate(mydata_generator()):
        # 训练判别器D识别真实图片
        r_fake = exe.run(program=train_d_fake,
                         fetch_list=[fake_avg_cost],
                         feed={'z': test_z})

        # 训练判别器D识别生成器G生成的假图片
        r_real = exe.run(program=train_d_real,
                         fetch_list=[real_avg_cost],
                         feed={'image': np.array(real_image)})

        # 训练生成器G生成符合判别器D标准的假图片
        r_g = exe.run(program=train_g,
                      fetch_list=[g_avg_cost],
                      feed={'z': test_z})

        if i % 100 == 0:
            print("Pass：%d, Batch：%d, 训练判别器D识别真实图片Cost：%0.5f, "
                  "训练判别器D识别生成器G生成的假图片Cost：%0.5f, "
                  "训练生成器G生成符合判别器D标准的假图片Cost：%0.5f" % (pass_id, i, r_fake[0], r_real[0], r_g[0]))

    # 测试生成的图片
    r_i = exe.run(program=infer_program,
                  fetch_list=[fake],
                  feed={'z': test_z})

    r_i = np.array(r_i).astype(np.float32)
    # 显示生成的图片
    show_image_grid(r_i[0])
```

同时在每个Pass之后又可以保存预测函数，用于之后预测生成图片使用。
```python
   # 保存预测模型
    save_path = 'infer_model/'
    # 删除旧的模型文件
    shutil.rmtree(save_path, ignore_errors=True)
    # 创建保持模型文件目录
    os.makedirs(save_path)
    # 保存预测模型
    fluid.io.save_inference_model(save_path, feeded_var_names=[z.name], target_vars=[fake], executor=exe, main_program=train_g)
```

在训练的过程可以输出每一个训练程序输出的损失值：
```
Pass：0, Batch：0, 训练判别器D识别真实图片Cost：1.03734, 训练判别器D识别生成器G生成的假图片Cost：0.46931, 训练生成器G生成符合判别器D标准的假图片Cost：0.54236
Pass：1, Batch：0, 训练判别器D识别真实图片Cost：1.09766, 训练判别器D识别生成器G生成的假图片Cost：0.32896, 训练生成器G生成符合判别器D标准的假图片Cost：0.44473
Pass：2, Batch：0, 训练判别器D识别真实图片Cost：1.17703, 训练判别器D识别生成器G生成的假图片Cost：0.38643, 训练生成器G生成符合判别器D标准的假图片Cost：0.39445
```

# 使用模型生成图片
在上一个文件中，我们已经训练得到一个预测模型，下面我们将使用这个预测模型直接生成图片。创建`infer.py`文件用于预测生成图片。首先导入相应的依赖包。
```python
import os
import paddle
import matplotlib.pyplot as plt
import numpy as np
import paddle.fluid as fluid
```

然后创建执行器，这里可以使用CPU进行预测可以，因为预测并不需要太大的计算。然后加载上一步训练保存的预测模型，获取预测程序，输入层的名称，和生成器。
```python
# 创建执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 保存预测模型路径
save_path = 'infer_model/'
# 从模型中获取预测程序、输入数据名称列表、分类器
[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)
```

跟训练的时候一样，需要生成噪声数据作为输入数据。这里说明一下，输入数据`z_generator `的batch大小就是生成图片的数量。
```python
# 噪声维度
z_dim = 100

# 噪声生成
def z_reader():
    while True:
        yield np.random.uniform(-1.0, 1.0, (z_dim)).astype('float32')

z_generator = paddle.batch(z_reader, batch_size=32)()
test_z = np.array(next(z_generator))
```

这里创建一个保存生成图片的函数，用于保存预测生成的图片。
```python
# 保存图片
def save_image(images):
    for i, image in enumerate(images):
        image = image.transpose((2, 1, 0))
        save_image_path = 'infer_image'
        if not os.path.exists(save_image_path):
            os.makedirs(save_image_path)
        plt.imsave(os.path.join(save_image_path, "test_%d.png" % i), image)
```

最后执行预测程序，开始生成图片。预测输出的结果就是图片的数据，通过保存这些数据就是保存图片了。
```python
# 测试生成的图片
r_i = exe.run(program=infer_program,
              feed={feeded_var_names[0]: test_z},
              fetch_list=target_var)

r_i = np.array(r_i).astype(np.float32)

# 显示生成的图片
save_image(r_i[0])

print('生成图片完成')
```

目前这个网络在训练比较复杂的图片时，模型的拟合效果并不太好，也就是说生成的图片没有我们想象那么好。所以这个网络还需要不断调整，如果读者有更好的建议，欢迎交流一下。

# 参考资料
1. https://github.com/oraoto/learn_ml/blob/master/paddle/gan-mnist-split.ipynb
2. https://www.cnblogs.com/max-hu/p/7129188.html
3. https://blog.csdn.net/somtian/article/details/72126328