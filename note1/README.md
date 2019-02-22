@[TOC]

# 前言
这一章我们介绍如何安装新版本的PaddlePaddle，这里说的新版本主要是说Fluid版本。Fluid 是设计用来让用户像Pytorch和Tensorflow Eager Execution一样执行程序。在这些系统中，不再有模型这个概念，应用也不再包含一个用于描述Operator图或者一系列层的符号描述，而是像通用程序那样描述训练或者预测的过程。也就是说PaddlePaddle从Fluid版本开始使用动态图机制，所以我们这个系列也是使用Fluid版本编写的教程。

# 环境
 - 系统：64位Windows 10专业版，64位Ubuntu 16.04 
 - Python环境：Python 3.5
 - 内存：8G

# Windows下安装
PaddlePaddle在1.2版本之后开始支持Windows，也就是说使用Windows的用户不需要再安装Docker容器，或者使用Windows的Liunx子系统，直接可以在Windows系统本身安装PaddlePaddle。下面我们就介绍如何在Windows安装PaddlePaddle，分为两个部分介绍，首先安装Python 3.5环境，然后再使用命令安装PaddlePaddle。

## 安装Python
1、本系列使用的是Python 3.5，官方在Windows上支持Python2.7.15，Python3.5.x，Python3.6.x，Python3.7.x。读者根据自己的实际情况安装自己喜欢的版本。官网下载页面：https://www.python.org/downloads/windows/ ，官网下载地址：https://www.python.org/ftp/python/3.5.4/python-3.5.4-amd64.exe
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190124144459791.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3,size_16,color_FFFFFF,t_70)

2、双击运行Python 3.5安装包开始安装，记住要选上添加环境变量，这很重要，之后使用命令都要依赖这个环境变量，要不每次都要进入到`pip`的目录比较麻烦。然后点击`Install Now`开始安装。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190124145152305.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3,size_16,color_FFFFFF,t_70)

3、安装完成之后，测试安装是否成功，打开`Windows PowerShell`或者`cmd`，笔者的系统是Windows 10，可以使用`Windows PowerShell`，如果读者是其他系统，可以使用`cmd`。用命令`python -V`查看是否安装成功。正常安装之后可以显示安装Python的版本。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190124145557657.png)

## 安装PaddlePaddle
PaddlePaddle支持Windows之后，安装起来非常简单，只需要一条命令就可以完成安装。

 - 安装CPU版本，打开`Windows PowerShell`，输入以下命令。可以使用`==`指定安装PaddlePaddle的版本，如没有指定版本，默认安装是最新版本。`-i`后面是镜像源地址，使用国内镜像源可以大大提高下载速度：
```
pip3 install paddlepaddle==1.2.0 -i https://mirrors.aliyun.com/pypi/simple/
```

 - 安装GPU版本，目前不支持Windows的GPU版本，支持后会更新。

 - 测试安装是否成功，在`Windows PowerShell`中输入命令`python`，进入到Python 编辑环境，并输入以下代码，导没有保存证明安装成功：
```
import paddle.fluid
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190124151103405.png)

# Ubuntu下安装
下面介绍在Ubuntu系统下安装PaddlePaddle，PaddlePaddle支持64位的Ubuntu 14.04 /16.04 /18.04系统，Python支持Python2.7.15，Python3.5.x，Python3.6.x，Python3.7.x。

 - 安装Python 3.5（通常不需要执行）。通常情况下Ubuntu 16.04自带的就是Python 3.5，其他Ubuntu的版本自带的可能是其他版本，不过没有关系，PaddlePaddle基本都支持，所以不必专门安装Python3.5。
```
sudo apt install python3.5
sudo apt install python3.5-dev
```

 - 安装CPU版本，打开Ubuntu的终端，快捷键是`Ctrl+Alt+T`，输入以下命令。可以使用`==`指定安装PaddlePaddle的版本，如没有指定版本，默认安装是最新版本。`-i`后面是镜像源地址，使用国内镜像源可以大大提高下载速度：
```
pip3 install paddlepaddle==1.2.0 -i https://mirrors.aliyun.com/pypi/simple/
```

 - 安装GPU版本，安装GPU版本之前，要先安装CUDA，可以查看笔者之前的文章[《Ubuntu安装和卸载CUDA和CUDNN》](https://blog.csdn.net/qq_33200967/article/details/80689543)，安装完成 CUDA 9 和 CUDNN 7 之后，再安装PaddlePaddle的GPU版本，安装命令如下。可以使用`==`指定安装PaddlePaddle的版本和CUDA、CUDNN的版本，这必须要跟读者系统本身安装的CUDA版本对应，比如以下命令就是安装支持CUDA 9.0和CUDNN 7的PaddlePaddle版本。`-i`后面是镜像源地址，使用国内镜像源可以大大提高下载速度：
```
pip3 install paddlepaddle-gpu==1.2.0.post97 -i https://mirrors.aliyun.com/pypi/simple/
```

 - 测试安装是否成功，在终端中输入命令`python3`，进入到Python 编辑环境，并输入以下代码，正确情况下如图所示：
```
import paddle.fluid
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190125093720742.png)

# 源码编译
这部分我们将介绍使用源码编译PaddlePaddle，可以通过这种方式安装符合读者需求的PaddlePaddle，比如笔者的电脑安装的是CUDA 10 和 CUDNN 7，而目前官方提供的没有支持CUDA 10 和 CUDNN 7的PaddlePaddle版本，所以笔者就可以通过源码编译的方式编译PaddlePaddle安装包，当然也要PaddlePaddle支持才行。

## Windows下源码编译
下面我们将介绍在Windows系统下进行源码编译PaddlePaddle。目前支持使用的系统是64位的Windows 10 家庭版/专业版/企业版。

1. 安装`Visual Studio 2015 Update3`。下载地址：https://visualstudio.microsoft.com/zh-hans/vs/older-downloads/ ，因为是旧版本，还有`加入免费的 Dev Essentials 计划`才能正常下载。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190125155716929.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190125155913275.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3,size_16,color_FFFFFF,t_70)


2. 安装`cmake 3.13`，下载cmake的安装包，下载地址：https://cmake.org/download/ ，一路默认，只需要在添加环境变量的时候注意添加环境变量就可以了。如何存在环境变量问题，可以重启系统。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190125152728392.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3,size_16,color_FFFFFF,t_70)
3. 安装Python的依赖库，只要执行以下命令。关于Windows安装Python，在“Windows下安装”部分已经介绍过，这里就不介绍了。
```
pip3 install numpy
pip3 install protobuf
pip3 install wheel
```

4. 安装 git 工具。git的下载地址：https://git-scm.com/downloads ，下载git的安装包，安装的时候一路默认就可以了。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190125153826299.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3,size_16,color_FFFFFF,t_70)

5. 右键打开`Git Bash Here`，执行以下两条命令。将PaddlePaddle的源码clone在当下目录下的Paddle的文件夹中，并进入Padde目录下，操作如下图所示，之后的命令也是在这个终端操作：
```
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190125164055182.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190125164157348.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3,size_16,color_FFFFFF,t_70)

6. 切换到较稳定release分支下进行编译，入笔者选择1.2版本的代码：
```
git checkout release/1.2
```

7. 创建名为build的目录并进入：
```
mkdir build
cd build
```

8. 执行编译
	- 编译**CPU版本**命令如下：
	```
	cmake .. -G "Visual Studio 14 2015 Win64" -DPY_VERSION=3.5 -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS} -DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
	```
	
	-  编译**GPU版本**，目前Windows还不支持GPU，支持后会更新。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190125164353250.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3,size_16,color_FFFFFF,t_70)
9. 下载第三方依赖包（openblas，snappystream），下载地址：https://github.com/wopeizl/Paddle_deps ，将整个`third_party`文件夹放到上面第7步创建的`build`目录下。
10. 使用`Blend for Visual Studio 2015` 打开`paddle.sln`文件，选择平台为`x64`，配置为`Release`，开始编译 
11. 编译成功后进入`\paddle\build\python\dist`目录下找到生成的`.whl`包
12. 执行以下命令安装编译好的PaddlePaddle包：
```
pip3 install （whl包的名字）
```

## Ubuntu本地下源码编译
下面介绍的是使用Ubuntu编译PaddlePaddle源码，笔者的系统是64位的Ubuntu 16.04，Python环境是Python 3.5。

### 安装openCV
1.  更新apt的源，命令如下：
```
sudo apt update
```

2. 下载openCV源码，官方地址：https://opencv.org/releases.html ， 笔者下载的是3.4.5版本，选择的是`Sources`点击下载。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190125095611936.png)

3. 解压openCV源码，命令如下：
```
unzip opencv-3.4.5.zip
```

4. 安装可能需要的依赖库，命令如下：
```
sudo apt-get install cmake
sudo apt-get install build-essential libgtk2.0-dev libavcodec-dev libavformat-dev libjpeg.dev libtiff4.dev libswscale-dev libjasper-dev
```

5. 开始执行cmake。
```
cd opencv-3.4.5/
mkdir my_build_dir
cd my_build_dir
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
```

6. 开始执行编译
```
make -j$(nproc)
```

7. 执行安装命令
```
sudo make install
```

### 安装依赖环境
编译PaddlePaddle源码之前，还需要安装以下的一些依赖环境。
```
sudo apt install python3.5-dev
sudo apt-get udpate
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install curl
sudo curl https://bootstrap.pypa.io/get-pip.py -o - | python3.5
sudo easy_install pip
sudo apt install swig
sudo apt install wget
sudo pip install numpy==1.14.0
sudo pip install protobuf==3.1.0
sudo pip install wheel
sudo apt install patchelf
```

### 编译PaddlePaddle

1. 将PaddlePaddle的源码clone在当下目录下的Paddle的文件夹中，并进入Padde目录下，命令如下：
```
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
```

2. 切换到较稳定release分支下进行编译，比如笔者使用的是1.2版本，读者可以根据自己的情况选择其他版本：
```
git checkout release/1.2
```

3. 创建并进入一个叫build的目录下：
```
mkdir build && cd build
```

4. 执行cmake，这里分为CPU版本和GPU版本。
	- 编译**CPU版本**，命令如下。使用参数`-DPY_VERSION`指定编译的PaddlePaddle支持的Python版本，笔者这里选择的是Python 3.5。并且使用参数`-DWITH_FLUID_ONLY`指定不编译V2版本的PaddlePaddle代码。使用参数`-DWITH_GPU`指定不使用GPU，也就是只编译CPU版本：
	```
	cmake .. -DPY_VERSION=3.5 -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
	```
	- 编译**GPU版本**，还要安装一下依赖环境，如下：
		1. 安装 CUDA 和 CUDNN，可以查看笔者之前的文章[《Ubuntu安装和卸载CUDA和CUDNN》](https://blog.csdn.net/qq_33200967/article/details/80689543)
		2. 安装nccl2，命令如下
		```
		wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
		dpkg -i nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
		sudo apt-get install -y libnccl2=2.2.13-1+cuda9.0 libnccl-dev=2.2.13-1+cuda9.0
		 ```
	 	3. 执行cmake。使用参数`-DPY_VERSION`指定编译的PaddlePaddle支持的Python版本，笔者这里选择的是Python 3.5。并且使用参数`-DWITH_FLUID_ONLY`指定不编译V2版本的PaddlePaddle代码。使用参数`-DWITH_GPU`指定使用GPU，同时编译支持CPU和GPU版本的PaddlePaddle。
	 	```
	 	cmake .. -DPY_VERSION=3.5 -DWITH_FLUID_ONLY=ON -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
		```

5. 使用以下命令正式编译，编译时间比较长：
```
make -j$(nproc)
```

6. 编译成功后进入`/paddle/build/python/dist`目录下找到生成的PaddlePaddle`.whl`包，可以使用这个命令进入到指定目录。
```
cd /paddle/build/python/dist
```

7. 在当前机器或目标机器安装编译好的`.whl`包：
```
pip3 install （whl包的名字）
```

## Ubuntu使用Docker源码编译
使用docker编译的安装包只能支持Ubuntu的PaddlePaddle，因为下载docker镜像也是Ubuntu系统的。通过使用docker编译PaddlePaddle得到的安装包，可以在docker本身使用，之后可以使用docker执行PaddlePaddle。也可以本地的Ubuntu上安装使用，不过要注意的是docker中的系统是Ubuntu 16.04。

### 安装Docker
1. 安装前准备
```python
# 卸载系统原有docker
sudo apt-get remove docker docker-engine docker.io
# 更新apt-get源 
sudo apt-get update
# 安装docker的依赖 
sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - 
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu && $(lsb_release -cs) && stable"
```

2. 安装Docker，编译**CPU版本**使用。
```python
# 再次更新apt-get源 
sudo apt-get update
# 开始安装docker 
sudo apt-get install docker-ce
# 加载docker 
sudo apt-cache madison docker-ce
# 验证docker是否安装成功
sudo docker run hello-world
```
正常情况下输出如下图所示。
![在这里插入图片描述](https://img-blog.csdnimg.cn/201901251129238.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3,size_16,color_FFFFFF,t_70)

3. 安装nvidia-docker，编译**GPU版本**使用（根据情况安装）。安装之前要确认本地有独立显卡并安装的显卡驱动。
```
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
```

### 编译PaddlePaddle
1. 克隆PaddlePaddle源码：
```
git clone https://github.com/PaddlePaddle/Paddle.git
```

2. 进入Paddle目录下：
```
cd Paddle
```

3. 启动docker镜像
	- 编译**CPU版本**，使用命令
	```
	sudo docker run --name paddle-test -v $PWD:/paddle --network=host -it hub.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash
	 ```
	- 编译**GPU版本**，使用命令
	```
	sudo nvidia-docker run --name paddle-test -v $PWD:/paddle --network=host -it hub.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash
	```

4. 进入Docker后进入paddle目录下：
```
cd paddle
```

5. 切换到较稳定release分支下进行编译，读者可以根据自己的情况选择其他版本：
```
git checkout release/1.2
```

6. 创建并进入`/paddle/build`路径下：
```
mkdir -p /paddle/build && cd /paddle/build
```

7. 使用以下命令安装相关依赖：
```
pip3 install protobuf==3.1.0
apt install patchelf
```

8. 执行cmake：
	- 编译**CPU版本**PaddlePaddle的命令。使用参数`-DPY_VERSION`指定编译的PaddlePaddle支持的Python版本，笔者这里选择的是Python 3.5。并且使用参数`-DWITH_FLUID_ONLY`指定不编译V2版本的PaddlePaddle代码。使用参数`-DWITH_GPU`指定不使用GPU，只编译支持CPU的PaddlePaddle：
	```
	cmake .. -DPY_VERSION=3.5 -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
	```
	- 编译**GPU版本**PaddlePaddle的命令。使用参数`-DPY_VERSION`指定编译的PaddlePaddle支持的Python版本，笔者这里选择的是Python 3.5。并且使用参数`-DWITH_FLUID_ONLY`指定不编译V2版本的PaddlePaddle代码。使用参数`-DWITH_GPU`指定使用GPU，同时编译支持CPU和GPU版本的PaddlePaddle。这里要注意一下，我们拉取的这个镜像是CUDA 8.0的，不一定跟读者本地的CUDA版本对应，这可能导致编译的安装包在本地不可用：
	```
	cmake .. -DPY_VERSION=3.5 -DWITH_FLUID_ONLY=ON -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
	```

9. 执行编译：
```
make -j$(nproc)
```

10. 编译成功后，生成的安装包存放在`/paddle/build/python/dist`目录下，如果是想在docker中安装PaddlePaddle，可以直接在docker中打开这个目录。如果要在本地安装的话，还有先退出docker，并进入到这个目录：
```python
# 在docker镜像中安装
cd /paddle/build/python/dist
# 在Ubuntu本地安装】
exit
cd build/python/dist
```

11. 安装PaddlePaddle，执行以下命令：
```
pip3.5 install （whl包的名字）
```

# 测试环境
下面介绍在Windows测试PaddlePaddle的安装情况，Ubuntu环境类似。

1. 开发工具笔者喜欢使用PyCharm，下载地址：https://www.jetbrains.com/pycharm/download/#section=windows ， 笔者使用的是社区版本的PyCharm，因为这个是免费的[坏笑]。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190124163830889.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3,size_16,color_FFFFFF,t_70)

2. 创建一个新项目，并选择系统的Python环境，第一个是创建一个Python的虚拟环境，这里选择第二个外部的Python环境，点击`...`选择外部Python环境。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190124164232564.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3,size_16,color_FFFFFF,t_70)

3. 这里选择系统的Python环境，选择的路径是之前安装Python的路径。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190124164050887.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3,size_16,color_FFFFFF,t_70)

3. 创建一个Python程序文件，并命名为`test_paddle.py`，编写并执行以下测试代码，现在看不懂没有关系，跟着这个系列教程来学，我们会熟悉使用PaddlePaddle的：
```python
# Include libraries.
import paddle
import paddle.fluid as fluid
import numpy
import six

# Configure the neural network.
def net(x, y):
    y_predict = fluid.layers.fc(input=x, size=1, act=None)
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)
    return y_predict, avg_cost

                                
# Define train function.
def train(save_dirname):
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')
    y_predict, avg_cost = net(x, y)
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_cost)
    train_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.uci_housing.train(), buf_size=500),
        batch_size=20)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    def train_loop(main_program):
        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe.run(fluid.default_startup_program())

        PASS_NUM = 1000
        for pass_id in range(PASS_NUM):
            total_loss_pass = 0
            for data in train_reader():
                avg_loss_value, = exe.run(
                    main_program, feed=feeder.feed(data), fetch_list=[avg_cost])
                total_loss_pass += avg_loss_value
                if avg_loss_value < 5.0:
                    if save_dirname is not None:
                        fluid.io.save_inference_model(
                            save_dirname, ['x'], [y_predict], exe)
                    return
            print("Pass %d, total avg cost = %f" % (pass_id, total_loss_pass))

    train_loop(fluid.default_main_program())

# Infer by using provided test data.
def infer(save_dirname=None):
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets] = (
            fluid.io.load_inference_model(save_dirname, exe))
        test_reader = paddle.batch(paddle.dataset.uci_housing.test(), batch_size=20)

        test_data = six.next(test_reader())
        test_feat = numpy.array(list(map(lambda x: x[0], test_data))).astype("float32")
        test_label = numpy.array(list(map(lambda x: x[1], test_data))).astype("float32")

        results = exe.run(inference_program,
                          feed={feed_target_names[0]: numpy.array(test_feat)},
                          fetch_list=fetch_targets)
        print("infer results: ", results[0])
        print("ground truth: ", test_label)

                                
# Run train and infer.
if __name__ == "__main__":
    save_dirname = "fit_a_line.inference.model"
    train(save_dirname)
    infer(save_dirname)
```

正常情况下会输出：
```
Pass 0, total avg cost = 13527.760742
Pass 1, total avg cost = 12497.969727
Pass 2, total avg cost = 11737.727539
Pass 3, total avg cost = 11017.893555
Pass 4, total avg cost = 9801.554688
Pass 5, total avg cost = 9150.510742
Pass 6, total avg cost = 8611.593750
Pass 7, total avg cost = 7924.654297
......
```

PaddlePaddle的安装已经介绍完成，那我们开始进入深度学习的大门吧。本系列教程将会一步步介绍如何使用PaddlePaddle，并使用PaddlePaddle应用到实际项目中。

项目代码GitHub地址：https://github.com/yeyupiaoling/LearnPaddle2/tree/master/note1

**注意：** 最新代码以GitHub上的为准

# 参考资料
1.	http://www.paddlepaddle.org/documentation/docs/zh/1.2/beginners_guide/install/install_Ubuntu.html
2.	http://www.paddlepaddle.org/documentation/docs/zh/1.2/beginners_guide/install/install_Windows.html
3.	https://blog.csdn.net/cocoaqin/article/details/78163171

