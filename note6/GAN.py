import numpy as np
import paddle
import paddle.fluid as fluid
import matplotlib.pyplot as plt


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
        y = fluid.layers.fc(y, size=128 * 7 * 7)
        y = fluid.layers.batch_norm(y)
        # 进行形状变换
        y = fluid.layers.reshape(y, shape=(-1, 128, 7, 7))
        # 第一组转置卷积运算
        y = deconv(x=y, num_filters=128, act='relu', output_size=[14, 14])
        # 第二组转置卷积运算
        y = deconv(x=y, num_filters=1, act='tanh', output_size=[28, 28])
    return y


# 判别器 Discriminator
def Discriminator(images, name="D"):
    # 定义一个卷积池化组
    def conv_pool(input, num_filters, act=None):
        return fluid.nets.simple_img_conv_pool(input=input,
                                               filter_size=5,
                                               num_filters=num_filters,
                                               pool_size=2,
                                               pool_stride=2,
                                               act=act)

    with fluid.unique_name.guard(name + "/"):
        y = fluid.layers.reshape(x=images, shape=[-1, 1, 28, 28])
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


# 训练判别器D识别真实图片
with fluid.program_guard(train_d_real, startup):
    # 创建读取真实数据集图片的data，并且label为1
    real_image = fluid.layers.data('image', shape=[1, 28, 28])
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

# 训练生成器G生成符合判别器D标准的假图片
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


# 噪声生成
def z_reader():
    while True:
        yield np.random.uniform(-1.0, 1.0, (z_dim)).astype('float32')


# 读取MNIST数据集，不使用label
def mnist_reader(reader):
    def r():
        for img, label in reader():
            yield img.reshape(1, 28, 28)

    return r


# 保存图片
def show_image_grid(images):
    for i, image in enumerate(images[:64]):
        image = image[0]
        plt.imsave("image/test_%d.png" % i, image, cmap='Greys_r')


# 生成真实图片reader
mnist_generator = paddle.batch(
    paddle.reader.shuffle(mnist_reader(paddle.dataset.mnist.train()), 30000), batch_size=128)
# 生成假图片的reader
z_generator = paddle.batch(z_reader, batch_size=128)()

# 创建执行器，最好使用GPU，CPU速度太慢了
# place = fluid.CPUPlace()
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
# 初始化参数
exe.run(startup)

# 测试噪声
test_z = np.array(next(z_generator))

# 开始训练
for pass_id in range(20):
    for i, real_image in enumerate(mnist_generator()):
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

    # 显示生成的图片
    show_image_grid(r_i[0])
