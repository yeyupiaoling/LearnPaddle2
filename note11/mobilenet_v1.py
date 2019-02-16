import paddle.fluid as fluid


def net(input, class_dim, scale=1.0):
    # conv1: 112x112
    input = conv_bn_layer(input=input,
                          filter_size=3,
                          channels=3,
                          num_filters=int(32 * scale),
                          stride=2,
                          padding=1)

    # 56x56
    input = depthwise_separable(input=input,
                                num_filters1=32,
                                num_filters2=64,
                                num_groups=32,
                                stride=1,
                                scale=scale)

    input = depthwise_separable(input=input,
                                num_filters1=64,
                                num_filters2=128,
                                num_groups=64,
                                stride=2,
                                scale=scale)

    # 28x28
    input = depthwise_separable(input=input,
                                num_filters1=128,
                                num_filters2=128,
                                num_groups=128,
                                stride=1,
                                scale=scale)

    input = depthwise_separable(input=input,
                                num_filters1=128,
                                num_filters2=256,
                                num_groups=128,
                                stride=2,
                                scale=scale)

    # 14x14
    input = depthwise_separable(input=input,
                                num_filters1=256,
                                num_filters2=256,
                                num_groups=256,
                                stride=1,
                                scale=scale)

    input = depthwise_separable(input=input,
                                num_filters1=256,
                                num_filters2=512,
                                num_groups=256,
                                stride=2,
                                scale=scale)

    # 14x14
    for i in range(5):
        input = depthwise_separable(input=input,
                                    num_filters1=512,
                                    num_filters2=512,
                                    num_groups=512,
                                    stride=1,
                                    scale=scale)
    # 7x7
    input = depthwise_separable(input=input,
                                num_filters1=512,
                                num_filters2=1024,
                                num_groups=512,
                                stride=2,
                                scale=scale)

    input = depthwise_separable(input=input,
                                num_filters1=1024,
                                num_filters2=1024,
                                num_groups=1024,
                                stride=1,
                                scale=scale)

    feature = fluid.layers.pool2d(input=input,
                                  pool_size=0,
                                  pool_stride=1,
                                  pool_type='avg',
                                  global_pooling=True)

    net = fluid.layers.fc(input=feature,
                          size=class_dim,
                          act='softmax')
    return net


def conv_bn_layer(input, filter_size, num_filters, stride,
                  padding, channels=None, num_groups=1, act='relu', use_cudnn=True):
    conv = fluid.layers.conv2d(input=input,
                               num_filters=num_filters,
                               filter_size=filter_size,
                               stride=stride,
                               padding=padding,
                               groups=num_groups,
                               act=None,
                               use_cudnn=use_cudnn,
                               bias_attr=False)

    return fluid.layers.batch_norm(input=conv, act=act)


def depthwise_separable(input, num_filters1, num_filters2, num_groups, stride, scale):
    depthwise_conv = conv_bn_layer(input=input,
                                   filter_size=3,
                                   num_filters=int(num_filters1 * scale),
                                   stride=stride,
                                   padding=1,
                                   num_groups=int(num_groups * scale),
                                   use_cudnn=False)

    pointwise_conv = conv_bn_layer(input=depthwise_conv,
                                   filter_size=1,
                                   num_filters=int(num_filters2 * scale),
                                   stride=1,
                                   padding=0)
    return pointwise_conv
