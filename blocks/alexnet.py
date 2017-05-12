import tensorflow as tf
import numpy as np

IMAGE_NET_PIXEL_MEAN = 256.0*np.array([0.485, 0.456, 0.406])
IMAGE_NET_PIXEL_STD = 256.0*np.array([0.229, 0.224, 0.225])

CKPT = '/home/piter/PycharmProjects/sal2/ckpts/alexnet.ckpt'

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])



def inference(images, trainable, reuse):
    ''' images are in rgb format returned by imagenet batch manager (zero mean std of 1)'''
    bgr = tf.concat((tf.expand_dims(images[:, :, :, 2], 3), tf.expand_dims(images[:, :, :, 1], 3),
                  tf.expand_dims(images[:, :, :, 0], 3)), 3) * IMAGE_NET_PIXEL_STD

    scope = tf.get_variable_scope()
    old_reuse = scope.reuse
    scope._reuse = reuse

    # conv1
    # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11
    k_w = 11
    c_o = 96
    s_h = 4
    s_w = 4
    conv1W = tf.get_variable('conv1_w', [11, 11, 3, 96], dtype=tf.float32, trainable=trainable)
    conv1b = tf.get_variable('conv1_b', [96], dtype=tf.float32, trainable=trainable)
    conv1_in = conv(bgr, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    # lrn1
    # lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2
    alpha = 2e-05
    beta = 0.75
    bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    # maxpool1
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3
    k_w = 3
    s_h = 2
    s_w = 2
    padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv2
    # conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5
    k_w = 5
    c_o = 256
    s_h = 1
    s_w = 1
    group = 2
    conv2W = tf.get_variable('conv2_w', [5, 5, 48, 256], dtype=tf.float32, trainable=trainable)
    conv2b = tf.get_variable('conv2_b', [256], dtype=tf.float32, trainable=trainable)
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)

    # lrn2
    # lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2
    alpha = 2e-05
    beta = 0.75
    bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    # maxpool2
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3
    k_w = 3
    s_h = 2
    s_w = 2
    padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv3
    # conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3
    k_w = 3
    c_o = 384
    s_h = 1
    s_w = 1
    group = 1
    conv3W = tf.get_variable('conv3_w', [3, 3, 256, 384], dtype=tf.float32, trainable=trainable)
    conv3b = tf.get_variable('conv3_b', [384], dtype=tf.float32, trainable=trainable)
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    # conv4
    # conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3
    k_w = 3
    c_o = 384
    s_h = 1
    s_w = 1
    group = 2
    conv4W = tf.get_variable('conv4_w', [3, 3, 192, 384], dtype=tf.float32, trainable=trainable)
    conv4b = tf.get_variable('conv4_b', [384], dtype=tf.float32, trainable=trainable)
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)

    # conv5
    # conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3
    k_w = 3
    c_o = 256
    s_h = 1
    s_w = 1
    group = 2
    conv5W = tf.get_variable('conv5_w', [3, 3, 192, 256], dtype=tf.float32, trainable=trainable)
    conv5b = tf.get_variable('conv5_b', [256], dtype=tf.float32, trainable=trainable)
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in, name='c5relu')

    # maxpool5
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3
    k_w = 3
    s_h = 2
    s_w = 2
    padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # fc6
    # fc(4096, name='fc6')
    fc6W = tf.get_variable('fc6_w', [9216, 4096], dtype=tf.float32, trainable=trainable)
    fc6b = tf.get_variable('fc6_b', [4096], dtype=tf.float32, trainable=trainable)
    fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

    # fc7
    # fc(4096, name='fc7')
    fc7W = tf.get_variable('fc7_w', [4096, 4096], dtype=tf.float32, trainable=trainable)
    fc7b = tf.get_variable('fc7_b', [4096], dtype=tf.float32, trainable=trainable)
    fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

    # fc8
    # fc(1000, relu=False, name='fc8')
    fc8W = tf.get_variable('fc8_w', [4096, 1000], dtype=tf.float32, trainable=trainable)
    fc8b = tf.get_variable('fc8_b', [1000], dtype=tf.float32, trainable=trainable)
    fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

    scope._reuse = old_reuse
    return fc8
