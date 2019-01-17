import paddle.fluid as fluid


def conv_bn_layer(input, filter_size, num_filters, stride, padding, num_groups=1, if_act=True, use_cudnn=True):
    conv = fluid.layers.conv2d(input=input,
                               num_filters=num_filters,
                               filter_size=filter_size,
                               stride=stride,
                               padding=padding,
                               groups=num_groups,
                               use_cudnn=use_cudnn,
                               bias_attr=False)
    bn = fluid.layers.batch_norm(input=conv)
    if if_act:
        return fluid.layers.relu6(bn)
    else:
        return bn


def shortcut(input, data_residual):
    return fluid.layers.elementwise_add(input, data_residual)


def inverted_residual_unit(input,
                           num_in_filter,
                           num_filters,
                           ifshortcut,
                           stride,
                           filter_size,
                           padding,
                           expansion_factor):
    num_expfilter = int(round(num_in_filter * expansion_factor))

    channel_expand = conv_bn_layer(input=input,
                                   num_filters=num_expfilter,
                                   filter_size=1,
                                   stride=1,
                                   padding=0,
                                   num_groups=1,
                                   if_act=True)

    bottleneck_conv = conv_bn_layer(input=channel_expand,
                                    num_filters=num_expfilter,
                                    filter_size=filter_size,
                                    stride=stride,
                                    padding=padding,
                                    num_groups=num_expfilter,
                                    if_act=True,
                                    use_cudnn=False)

    linear_out = conv_bn_layer(input=bottleneck_conv,
                               num_filters=num_filters,
                               filter_size=1,
                               stride=1,
                               padding=0,
                               num_groups=1,
                               if_act=False)
    if ifshortcut:
        out = shortcut(input=input, data_residual=linear_out)
        return out
    else:
        return linear_out


def invresi_blocks(input, in_c, t, c, n, s, name=None):
    first_block = inverted_residual_unit(input=input,
                                         num_in_filter=in_c,
                                         num_filters=c,
                                         ifshortcut=False,
                                         stride=s,
                                         filter_size=3,
                                         padding=1,
                                         expansion_factor=t)

    last_residual_block = first_block
    last_c = c

    for i in range(1, n):
        last_residual_block = inverted_residual_unit(input=last_residual_block,
                                                     num_in_filter=last_c,
                                                     num_filters=c,
                                                     ifshortcut=True,
                                                     stride=1,
                                                     filter_size=3,
                                                     padding=1,
                                                     expansion_factor=t)
    return last_residual_block


def net(input, class_dim, scale=1.0):
    bottleneck_params_list = [
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    # conv1
    input = conv_bn_layer(input,
                          num_filters=int(32 * scale),
                          filter_size=3,
                          stride=2,
                          padding=1,
                          if_act=True)

    # bottleneck sequences
    i = 1
    in_c = int(32 * scale)
    for layer_setting in bottleneck_params_list:
        t, c, n, s = layer_setting
        i += 1
        input = invresi_blocks(input=input,
                               in_c=in_c,
                               t=t,
                               c=int(c * scale),
                               n=n,
                               s=s,
                               name='conv' + str(i))
        in_c = int(c * scale)
    # last_conv
    input = conv_bn_layer(input=input,
                          num_filters=int(1280 * scale) if scale > 1.0 else 1280,
                          filter_size=1,
                          stride=1,
                          padding=0,
                          if_act=True)

    feature = fluid.layers.pool2d(input=input,
                                  pool_size=7,
                                  pool_stride=1,
                                  pool_type='avg',
                                  global_pooling=True)
    net = fluid.layers.fc(input=feature,
                          size=class_dim,
                          act='softmax')
    return net
