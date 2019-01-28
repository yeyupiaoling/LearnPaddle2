import paddle.fluid as fluid
from PIL import Image
import numpy as np

# 创建执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 保存预测模型路径
save_path = 'models/infer_model/'
# 从模型中获取预测程序、输入数据名称列表、分类器
[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)


# 预处理图片
def load_image(file):
    im = Image.open(file)
    im = im.resize((32, 32), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32)
    # PIL打开图片存储顺序为H(高度)，W(宽度)，C(通道)。
    # PaddlePaddle要求数据顺序为CHW，所以需要转换顺序。
    im = im.transpose((2, 0, 1))
    # CIFAR训练图片通道顺序为B(蓝),G(绿),R(红),
    # 而PIL打开图片默认通道顺序为RGB,因为需要交换通道。
    im = im[(2, 1, 0), :, :]  # BGR
    im = im / 255.0
    im = np.expand_dims(im, axis=0)
    return im


# 获取图片数据
img = load_image('image/cat.png')

# 执行预测
result = exe.run(program=infer_program,
                 feed={feeded_var_names[0]: img},
                 fetch_list=target_var)

# 显示图片并输出结果最大的label
lab = np.argsort(result)[0][0][-1]

names = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

print('预测结果标签为：%d， 名称为：%s， 概率为：%f' % (lab, names[lab], result[0][0][lab]))
