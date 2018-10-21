import paddle.fluid as fluid
import numpy as np

# 定义两个张量
a = fluid.layers.create_tensor(dtype='int64', name='a')
b = fluid.layers.create_tensor(dtype='int64', name='b')

# 将两个张量求和
y = fluid.layers.sum(x=[a, b])

# 创建一个使用CPU的解释器
place = fluid.CPUPlace()
exe = fluid.executor.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

a1 = np.array([3]).astype('int64')
b1 = np.array([1]).astype('int64')

# 进行运算，并把y的结果输出
result, r = exe.run(program=fluid.default_main_program(),
                    feed={'a': a1, 'b': b1},
                    fetch_list=[y, b])
print(result)
print(r)
