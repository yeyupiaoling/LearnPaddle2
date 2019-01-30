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