import paddle.fluid as fluid


def linear(input_x, input_y):
    x = fluid.layers.create_tensor(dtype='float32', name='x')
    y = fluid.layers.create_tensor(dtype='float32', name='y')
    a = fluid.layers.create_tensor(dtype='float32', name='a')
    b = fluid.layers.create_tensor(dtype='float32', name='b')

    return a, b


a, b = linear()

X = [1, 2, 3, 4, 5]
Y = [3, 5, 7, 9, 11]

cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)

optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

a, b = exe.run(program=fluid.default_main_program(),
               feed={'x': X, 'y': Y},
               fetch_list=[a, b])

print('a=', a, ',b=', b)
