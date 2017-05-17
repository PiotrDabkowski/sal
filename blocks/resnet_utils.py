import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

FLAGS = tf.app.flags.FLAGS

class Config:
    def __init__(self):
        root = self.Scope('')
        for k, v in FLAGS.__dict__['__flags'].iteritems():
            root[k] = v
        self.stack = [ root ]

    def iteritems(self):
        return self.to_dict().iteritems()

    def to_dict(self):
        self._pop_stale()
        out = {}
        # Work backwards from the flags to top fo the stack
        # overwriting keys that were found earlier.
        for i in range(len(self.stack)):
            cs = self.stack[-i]
            for name in cs:
                out[name] = cs[name]
        return out

    def _pop_stale(self):
        var_scope_name = tf.get_variable_scope().name
        top = self.stack[0]
        while not top.contains(var_scope_name):
            # We aren't in this scope anymore
            self.stack.pop(0)
            top = self.stack[0]

    def __getitem__(self, name):
        self._pop_stale()
        # Recursively extract value
        for i in range(len(self.stack)):
            cs = self.stack[i]
            if name in cs:
                return cs[name]

        raise KeyError(name)

    def set_default(self, name, value):
        if not name in self:
            self[name] = value

    def __contains__(self, name):
        self._pop_stale()
        for i in range(len(self.stack)):
            cs = self.stack[i]
            if name in cs:
                return True
        return False

    def __setitem__(self, name, value):
        self._pop_stale()
        top = self.stack[0]
        var_scope_name = tf.get_variable_scope().name
        assert top.contains(var_scope_name)

        if top.name != var_scope_name:
            top = self.Scope(var_scope_name)
            self.stack.insert(0, top)

        top[name] = value

    class Scope(dict):
        def __init__(self, name):
            self.name = name

        def contains(self, var_scope_name):
            return var_scope_name.startswith(self.name)

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]

tf.app.flags.DEFINE_integer('input_size', 224, "input image size")

activation = tf.nn.relu


def bn(x, c):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if c['use_bias']:
        bias = _get_variable('bias', params_shape,
                             initializer=tf.zeros_initializer)
        return x + bias

    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.constant_initializer(0),
                         trainable=c['is_training_py'])
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.constant_initializer(1),
                          trainable=c['is_training_py'])

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.constant_initializer(0),
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.constant_initializer(1),
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        c['is_training'], lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    # x.set_shape(inputs.get_shape()) ??

    return x


def fc(x, c):
    num_units_in = x.get_shape()[1]
    num_units_out = c['fc_units_out']
    weights_initializer = tf.truncated_normal_initializer(
        stddev=FC_WEIGHT_STDDEV)

    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV,
                            trainable=c['is_training_py'])
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           initializer=tf.constant_initializer(),
                           trainable=c['is_training_py'])
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x


def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


def conv(x, c):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']

    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY,
                            trainable=c['is_training_py'])
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')


import numpy as np
IMAGE_NET_PIXEL_MEAN = 256.0*np.array([0.485, 0.456, 0.406])
IMAGE_NET_PIXEL_STD = 256.0*np.array([0.229, 0.224, 0.225])

def imagenet_preprocess(rgb):
    """Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
    red, green, blue = tf.split(rgb * 255.0, 3, 3)
    bgr = tf.concat([blue, green, red], 3)
    bgr -= IMAGENET_MEAN_BGR
    return bgr




def stack(x, c):
    for n in range(c['num_blocks']):
        s = c['stack_stride'] if n == 0 else 1
        c['block_stride'] = s
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x, c)
    return x


def block(x, c):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed.
    # That is the case when bottleneck=False but when bottleneck is
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    m = 4 if c['bottleneck'] else 1
    filters_out = m * c['block_filters_internal']

    shortcut = x  # branch 1

    c['conv_filters_out'] = c['block_filters_internal']

    if c['bottleneck']:
        with tf.variable_scope('a'):
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('b'):
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('c'):
            c['conv_filters_out'] = filters_out
            c['ksize'] = 1
            assert c['stride'] == 1
            x = conv(x, c)
            x = bn(x, c)
    else:
        with tf.variable_scope('A'):
            c['stride'] = c['block_stride']
            assert c['ksize'] == 3
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('B'):
            c['conv_filters_out'] = filters_out
            assert c['ksize'] == 3
            assert c['stride'] == 1
            x = conv(x, c)
            x = bn(x, c)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or c['block_stride'] != 1:
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            c['conv_filters_out'] = filters_out
            shortcut = conv(shortcut, c)
            shortcut = bn(shortcut, c)

    return activation(x + shortcut)