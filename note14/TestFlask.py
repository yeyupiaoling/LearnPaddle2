import os
import uuid
import numpy as np
import paddle.fluid as fluid
from PIL import Image
from flask import Flask, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
# 允许跨越访问
CORS(app)


# 根路径，返回一个字符串
@app.route('/')
def hello_world():
    return 'Welcome to PaddlePaddle'


# 上传文件
@app.route('/upload', methods=['POST'])
def upload_file():
    f = request.files['img']
    # 设置保存路径
    save_father_path = 'images'
    img_path = os.path.join(save_father_path, str(uuid.uuid1()) + secure_filename(f.filename).split('.')[-1])
    if not os.path.exists(save_father_path):
        os.makedirs(save_father_path)
    f.save(img_path)
    return 'success, save path: ' + img_path


# 预处理图片
def load_image(file):
    img = Image.open(file)
    # 统一图像大小
    img = img.resize((224, 224), Image.ANTIALIAS)
    # 转换成numpy值
    img = np.array(img).astype(np.float32)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = img[(2, 1, 0), :, :] / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# 创建执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 保存预测模型路径
save_path = 'infer_model/'
# 从模型中获取预测程序、输入数据名称列表、分类器
[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)


@app.route('/infer', methods=['POST'])
def infer():
    f = request.files['img']
    # 设置保存路径
    save_father_path = 'images'
    img_path = os.path.join(save_father_path, str(uuid.uuid1()) + '.' + secure_filename(f.filename).split('.')[-1])
    if not os.path.exists(save_father_path):
        os.makedirs(save_father_path)
    f.save(img_path)
    img = load_image(img_path)
    result = exe.run(program=infer_program,
                     feed={feeded_var_names[0]: img},
                     fetch_list=target_var)

    # 显示图片并输出结果最大的label
    lab = np.argsort(result)[0][0][-1]

    names = ['苹果', '哈密瓜', '胡萝卜', '樱桃', '黄瓜', '西瓜']

    r = '{"label":%d, "name":"%s", "possibility":%f}' % (lab, names[lab], result[0][0][lab])
    print(r)
    return r


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=80)
