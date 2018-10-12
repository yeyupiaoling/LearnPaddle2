import paddle.fluid as fluid


def linear(input_x, input_y):
    x = fluid.layers.create_tensor(dtype='float32', name='x')
    y = fluid.layers.create_tensor(dtype='float32', name='y')

