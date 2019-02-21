import os
import paddle
import matplotlib.pyplot as plt
import numpy as np
import paddle.fluid as fluid

# 创建执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 保存预测模型路径
save_path = 'infer_model/'
# 从模型中获取预测程序、输入数据名称列表、分类器
[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)

# 噪声维度
z_dim = 100


# 噪声生成
def z_reader():
    while True:
        yield np.random.uniform(-1.0, 1.0, (z_dim)).astype('float32')


# 保存图片
def save_image(images):
    for i, image in enumerate(images):
        image = image.transpose((2, 1, 0))
        save_image_path = 'infer_image'
        if not os.path.exists(save_image_path):
            os.makedirs(save_image_path)
        plt.imsave(os.path.join(save_image_path, "test_%d.png" % i), image)


z_generator = paddle.batch(z_reader, batch_size=32)()
test_z = np.array(next(z_generator))

# 执行预测
result = exe.run(program=infer_program,
                 feed={feeded_var_names[0]: test_z},
                 fetch_list=target_var)


# 测试生成的图片
r_i = exe.run(program=infer_program,
              feed={feeded_var_names[0]: test_z},
              fetch_list=target_var)

r_i = np.array(r_i).astype(np.float32)

# 显示生成的图片
save_image(r_i[0])

print('生成图片完成')
