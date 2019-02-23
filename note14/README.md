暂时这样凑合着看，之后有时间再补充文字说明。[微笑]

@[TOC]

# 前言
如果读者使用过百度等的一些图像识别的接口，比如百度的细粒度图像识别接口，应该了解这个过程，省略其他的安全方面的考虑。这个接口大体的流程是，我们把图像上传到百度的网站上，然后服务器把这些图像转换成功矢量数据，最后就是拿这些数据传给深度学习的预测接口，比如是PaddlePaddle的预测接口，获取到预测结果，返回给客户端。这个只是简单的流程，真实的复杂性远远不止这些，但是我们只需要了解这些，然后去搭建属于我们的图像识别接口。

# 了解Flask
安装flask很简单，只要一条命令就可以了：
```
pip install flask
```
同时我们也使用到flask_cors，所以我们也要安装这个库
```
pip install flask_cors
```

创建一个`paddle_server.py`文件，然后编写一个简单的程序，了解一些如何使用这个Flask框架，首先导入所需的依赖库：
```python
import os
import uuid
import numpy as np
import paddle.fluid as fluid
from PIL import Image
from flask import Flask, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
```

编写一个`hello_world()`函数，使用`@app.route('/')`是指定访问的路径，该函数的返回值是一个字符串`Welcome to PaddlePaddle`：
```python
# 根路径，返回一个字符串
@app.route('/')
def hello_world():
    return 'Welcome to PaddlePaddle'
```

然后启动这个服务，如果是在Ubuntu的话，可能是需要在root下执行这个程序。
```python
if __name__ == '__main__':
    # 启动服务，并指定端口号
    app.run(port=80)
```

然后浏览器访问`http://127.0.0.1`，返回之前写好的字符串：
```
Welcome to PaddlePaddle
```

要预测图片，上传图片是首要的，所以我们来学习如何使用Flask来上传图片。

 - `secure_filename`是为了能够正常获取到上传文件的文件名
 - `/upload`指定该函数的访问地址
 - `methods=['POST']`指定该路径只能使用POST方法访问
 - `f = request.files['img']`读取表单名称为img的文件
 - `f.save(img_path)`在指定路径保存该文件

```python
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
```

然后再次启动服务
```python
if __name__ == '__main__':
    # 启动服务，并指定端口号
    app.run(port=80)
```

然后再创建`index.html`文件，编写一个表单，指定表单提交的路径`http://127.0.0.1/upload`，并设置表单提交数据的格式`multipart/form-data`，而且支持表单提交方式是POST。
```xml
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>预测图像</title>
</head>
<body>
<!--上传图片的表单-->
<form action="http://127.0.0.1/upload" enctype="multipart/form-data" method="post">
    选择上传的图像：<input type="file" name="img"><br>
    <input type="submit" value="上传">
</form>
</body>
</html>
```



# 预测服务
在`paddle_server.py`中添加：
```python
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
@app.route('/infer', methods=['POST'])
def infer():
    f = request.files['img']

    # 保存图片
    save_father_path = 'images'
    img_path = os.path.join(save_father_path, str(uuid.uuid1()) + '.' + secure_filename(f.filename).split('.')[-1])
    if not os.path.exists(save_father_path):
        os.makedirs(save_father_path)
    f.save(img_path)

    # 开始预测图片
    img = load_image(img_path)
    result = exe.run(program=infer_program,
                     feed={feeded_var_names[0]: img},
                     fetch_list=target_var)

    # 显示图片并输出结果最大的label
    lab = np.argsort(result)[0][0][-1]

    names = ['苹果', '哈密瓜', '胡萝卜', '樱桃', '黄瓜', '西瓜']

    # 打印和返回预测结果
    r = '{"label":%d, "name":"%s", "possibility":%f}' % (lab, names[lab], result[0][0][lab])
    print(r)
    return r
```

```python
if __name__ == '__main__':
    # 启动服务，并指定端口号
    app.run(port=80)
```

在`index.html`文件增加一个表单：
```xml
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>预测图像</title>
</head>
<body>
<!--调用服务器预测接口的表单-->
<form action="http://127.0.0.1/infer" enctype="multipart/form-data" method="post">
    选择预测的图像：<input type="file" name="img"><br>
    <input type="submit" value="预测">
</form>
</body>
</html>
```




