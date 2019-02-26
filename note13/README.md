暂时这样凑合着看，之后有时间再补充文字说明。[微笑]

@[TOC]

# 定义数据读取
`image_reader.py`文件：
```python
import os
import random
from multiprocessing import cpu_count
import numpy as np
import paddle
from PIL import Image
```


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
`train.py`文件：
```python
import os
import shutil
import numpy as np
import paddle
import paddle.fluid as fluid
import matplotlib.pyplot as plt
import image_reader
```

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
```

```python
# 从Program获取prefix开头的参数名字
def get_params(program, prefix):
    all_params = program.global_block().all_parameters()
    return [t.name for t in all_params if t.name.startswith(prefix)]
```

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

```python
# 噪声生成
def z_reader():
    while True:
        yield np.random.uniform(-1.0, 1.0, (z_dim)).astype('float32')
```

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

```python
# 生成真实图片reader
mydata_generator = paddle.batch(reader=image_reader.train_reader('datasets', image_size), batch_size=32)
# 生成假图片的reader
z_generator = paddle.batch(z_reader, batch_size=32)()
test_z = np.array(next(z_generator))
```


```python
# 创建执行器，最好使用GPU，CPU速度太慢了
# place = fluid.CPUPlace()
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
# 初始化参数
exe.run(startup)
```


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

# 使用模型生成图片
`infer.py`文件：
```python
import os
import paddle
import matplotlib.pyplot as plt
import numpy as np
import paddle.fluid as fluid
```


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
