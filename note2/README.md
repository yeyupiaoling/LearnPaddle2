@[TOC]

# 前言
在第一章介绍了PaddlePaddle的安装，接下来我们将介绍如何使用PaddlePaddle。PaddlePaddle是百度在2016年9月27日开源的一个深度学习框架，也是目前国内唯一一个开源的深度学习框架。PaddlePaddle在0.11.0版本之后，开始推出Fluid版本，Fluid版本相对之前的V2版本，Fluid的代码结构更加清晰，使用起来更加方便。这本章中我们将会介绍如何使用PaddlePaddle来计算1+1，选择这个简单的例子主要是为了让读者了解PaddlePaddle的Fluid版本的使用，掌握PaddlePaddle的使用流程。我们讲过介绍如何使用PaddlePaddle定义一个张量和如何对张量进行计算。

# 计算常量的1+1
PaddlePaddle类似一个科学计算库，比如Python下我们使用的numpy，提供的大量的计算操作，但是PaddlePaddle的计算对象是张量。我们下面就编写一个`constant_sum.py`Python文件，使用PaddlePaddle计算一个`[[1, 1], [1, 1]] * [[1, 1], [1, 1]]`。

首先导入PaddlePaddle库，大部分的API都在`paddle.fluid`下。
```python 
import paddle.fluid as fluid
```

定义两个张量的常量x1和x2，并指定它们的形状是[2, 2]，并赋值为1铺满整个张量，类型为int64.
```python
# 定义两个张量
x1 = fluid.layers.fill_constant(shape=[2, 2], value=1, dtype='int64')
x2 = fluid.layers.fill_constant(shape=[2, 2], value=1, dtype='int64')
```

接着定义一个操作，该计算是将上面两个张量进行加法计算，并返回一个求和的算子。PaddlePaddle提供了大量的操作，比如加减乘除、三角函数等，读者可以在`fluid.layers`找到。
```python
# 将两个张量求和
y1 = fluid.layers.sum(x=[x1, x2])
```

然后创建一个执行器，可以在这里指定计算使用CPU或GPU。当使用`CPUPlace()`时使用的是CPU，如果是`CUDAPlace()`使用的是GPU。解析器是之后使用它来进行计算过的，比如在执行计算之前我们要先执行参数初始化的`program`也是要使用到解析器的，因为只有解析器才能执行`program`。
```python
# 创建一个使用CPU的执行器
place = fluid.CPUPlace()
exe = fluid.executor.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())
```

最后执行计算，`program`的参数值是主程序，不是上一步使用的是初始化参数的程序，`program`默认一共有两个，分别是`default_startup_program()`和`default_main_program()`。`fetch_list`参数的值是在解析器在run之后要输出的值，我们要输出计算加法之后输出结果值。最后计算得到的也是一个张量。
```python
# 进行运算，并把y的结果输出
result = exe.run(program=fluid.default_main_program(),
                 fetch_list=[y1])
print(result)
```

输出信息：
```
[array([[2, 2],
       [2, 2]], dtype=int64)]
```

# 计算变量的1+1
上面计算的是张量常量的1+1，并不能随意修改常量的值，所以下面我们要编写一个`variable_sum.py`程序文件，使用张量变量作为乘数的程序，类似是一个占位符，等到将要计算时，再把要计算的值添加到占位符中进行计算。

导入PaddlePaddle库和numpy的库。
```python
import paddle.fluid as fluid
import numpy as np
```

定义两个张量，并不指定该张量的形状和值，它们是之后动态赋值的。这里只是指定它们的类型和名字，这个名字是我们之后赋值的关键。
```python
# 定义两个张量
a = fluid.layers.create_tensor(dtype='int64', name='a')
b = fluid.layers.create_tensor(dtype='int64', name='b')
```

使用同样的方式，定义这个两个张量的加法操作。
```python
# 将两个张量求和
y = fluid.layers.sum(x=[a, b])
```

这里我们同样是创建一个使用CPU的执行器，和进行参数初始化。
```python
# 创建一个使用CPU的执行器
place = fluid.CPUPlace()
exe = fluid.executor.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())
```

然后使用numpy创建两个张量值，之后我们要计算的就是这两个值。
```python
# 定义两个要计算的变量
a1 = np.array([3, 2]).astype('int64')
b1 = np.array([1, 1]).astype('int64')
```

这次`exe.run()`的参数有点不一样了，多了一个`feed`参数，这个就是要对张量变量进行赋值的。赋值的方式是使用了键值对的格式，key是定义张量变量是指定的名称，value就是要传递的值。在`fetch_list`参数中，笔者希望把`a, b, y`的值都输出来，所以要使用3个变量来接受返回值。
```python
# 进行运算，并把y的结果输出
out_a, out_b, result = exe.run(program=fluid.default_main_program(),
                               feed={'a': a1, 'b': b1},
                               fetch_list=[a, b, y])
print(out_a, " + ", out_b," = ", result)
```

输出信息：
```
[3 2]  +  [1 1]  =  [4 3]
```


到处为止，本章就结束了。在本章我们学会了PaddlePaddle的使用方式，那在下一章我们使用PaddlePaddle完成我们的第一个安装——线性回归，我们下章见。

同步到百度AI Studio平台：http://aistudio.baidu.com/aistudio/#/projectdetail/29339
同步到科赛网K-Lab平台：https://www.kesci.com/home/project/5bf75387954d6e0010668f76
项目代码GitHub地址：https://github.com/yeyupiaoling/LearnPaddle2/tree/master/note2

**注意：** 最新代码以GitHub上的为准

# 参考资料
1. http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html
