import tensorflow as tf
from tensorflow.contrib import layers as tf_layers


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def bottleneck_block(inputs, channels_out, stride=1, training=True, bottleneck_ratio=4, weights_collections=None, scale=False, activation=tf.nn.relu):
    bottleneck_channels = channels_out / bottleneck_ratio
    batch_norm_params = {'updates_collections': None, 'is_training': training, 'trainable': training, 'scale': scale}
    variable_collections = None if weights_collections is None else {'weights': weights_collections}

    with tf.variable_scope(None, default_name='ResidualBottleneckBlock'):
        with tf.variable_scope('bbc1'):
            out = tf_layers.conv2d(inputs, bottleneck_channels, 1,
                             activation_fn=activation,
                             normalizer_fn=tf_layers.batch_norm, normalizer_params=batch_norm_params,
                             variables_collections=variable_collections, trainable=training,
                             weights_initializer=tf_layers.variance_scaling_initializer(2.))
        with tf.variable_scope('bbc2'):
            out = tf_layers.conv2d(out, bottleneck_channels, 3, stride,
                                   activation_fn=activation,
                                   normalizer_fn=tf_layers.batch_norm, normalizer_params=batch_norm_params,
                                   variables_collections=variable_collections, trainable=training,
                                   weights_initializer=tf_layers.variance_scaling_initializer(2.))
        with tf.variable_scope('bbc3'):
            out = tf_layers.conv2d(out, channels_out, 1,
                                   normalizer_fn=tf_layers.batch_norm, normalizer_params=batch_norm_params,
                                   variables_collections=variable_collections, activation_fn=None, trainable=training,
                                   weights_initializer=tf_layers.variance_scaling_initializer(2.))

        if stride != 1 or inputs.get_shape().as_list()[-1] != channels_out :
            with tf.variable_scope('shortcut_projection'):
                inputs = tf_layers.conv2d(inputs, channels_out, 1, stride,
                                          variables_collections=variable_collections, activation_fn=None, trainable=training,
                                          weights_initializer=tf_layers.variance_scaling_initializer(2.))

        out = activation(inputs + out)

    return out


class DenseBlock:
    def __init__(self, growth_rate, layers, bottleneck=0.25, trainable=True, weights_collections=None, activation=tf.nn.relu, dilation_factors=None):
        self.growth_rate = [growth_rate]*layers if isinstance(growth_rate, int) else growth_rate
        self.layers = layers
        assert (0 < bottleneck <= 1) or isinstance(bottleneck, bool)
        self.bottleneck = bottleneck if bottleneck != 1 else False
        self.trainable = trainable
        self.activation = activation
        self.batch_norm_params = {'updates_collections': None, 'is_training': trainable, 'trainable': trainable}
        assert not isinstance(weights_collections, basestring), 'Must be a list of collections!'
        self.variable_collections = None if weights_collections is None else {'weights': weights_collections}
        self.dilation_factors = dilation_factors if dilation_factors is not None else [1]*self.layers
        assert len(self.dilation_factors) == layers, 'you know why'


    def __call__(self, inp):
        with tf.variable_scope(None, default_name='DenseBlock'):
            for layer in xrange(self.layers):
                original_input = inp
                input_channels = inp.get_shape().as_list()[-1]
                with tf.variable_scope(None, default_name='DenseUnit'):
                    if self.bottleneck:
                        with tf.variable_scope('bottleneck'):
                            cand = int(self.bottleneck * input_channels)
                            bottleneck_channels = max(cand, min(input_channels, 16)) # basically will not bottleneck below 16 or if input channels is smaller then 16 then no bottleneck at all
                            if bottleneck_channels < self.bottleneck*input_channels*2:
                                # BN -> RELU -> CONV
                                inp = tf_layers.batch_norm(inp, **self.batch_norm_params)
                                inp = self.activation(inp)
                                inp = tf_layers.conv2d(inp, bottleneck_channels, 1, 1,
                                                       variables_collections=self.variable_collections,
                                                       trainable=self.trainable,
                                                       activation_fn=None,
                                                       biases_initializer=None,
                                                       weights_initializer=tf_layers.variance_scaling_initializer(2.))
                    # BN -> RELU -> CONV
                    inp = tf_layers.batch_norm(inp, **self.batch_norm_params)
                    inp = self.activation(inp)
                    extra_maps = tf_layers.conv2d(inp, self.growth_rate[layer], 3, 1,
                                           rate = self.dilation_factors[layer],
                                           variables_collections=self.variable_collections,
                                           trainable=self.trainable,
                                           activation_fn=None,
                                           biases_initializer=None,
                                           weights_initializer=tf_layers.variance_scaling_initializer(2.))

                    # concat channels
                    inp = tf.concat((original_input, extra_maps), 3)
        return inp



class TrainsitionLayer:
    def __init__(self, keep_dim_fraction, trainable=True, weights_collections=None, kernel_size=2, stride=2, pool_method=tf_layers.avg_pool2d):
        assert (0 < keep_dim_fraction <= 1)
        self.keep_dim_fraction = keep_dim_fraction
        self.trainable = trainable
        self.weights_collections = weights_collections
        assert not isinstance(weights_collections, basestring), 'Must be a list of collections!'
        self.variable_collections = None if weights_collections is None else {'weights': weights_collections}
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool_method = pool_method

    def __call__(self, inp):
        with tf.variable_scope(None, default_name='DenseTransitionLayer'):
            if self.keep_dim_fraction != 1:
                reduced = DimReductionLayer(self.keep_dim_fraction, self.trainable, self.weights_collections)(inp)
            else:
                reduced = inp

            out = self.pool_method(reduced, self.kernel_size, self.stride)
        return out


class DimReductionLayer:
    def __init__(self, keep_dim_fraction, trainable=True, weights_collections=None):
        assert (0 < keep_dim_fraction < 1), 'keed_dim_fraction must be smaller than 1 and larger than 0'
        self.keep_dim_fraction = keep_dim_fraction
        self.trainable = trainable
        assert not isinstance(weights_collections, basestring), 'Must be a list of collections!'
        self.variable_collections = None if weights_collections is None else {'weights': weights_collections}

    def __call__(self, inp):
        input_channels = inp.get_shape().as_list()[-1]
        with tf.variable_scope(None, default_name='DimReductionLayer'):
            target_dims = int(self.keep_dim_fraction * input_channels)
            with tf.variable_scope('dimreduction'):
                reduced = tf_layers.conv2d(inp, target_dims, 1, 1,
                                           variables_collections=self.variable_collections,
                                           trainable=self.trainable,
                                           activation_fn=None,
                                           biases_initializer=None,
                                           weights_initializer=tf_layers.variance_scaling_initializer(2.))
        return reduced

