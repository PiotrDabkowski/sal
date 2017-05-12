import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers as tf_layers
import dense_net



class MagicBlock:
    '''
    
    '''

    def __call__(self, inp, trainable, weights_collections, **extra_params):
        raise NotImplementedError()

    def __lt__(self, other):
        raise TypeError('Magic block is not a magic container')

    def __repr__(self):
        return self.NAME

    @property
    def NAME(self):
        return self.__class__.__name__


class MagicContainer(MagicBlock):
    SUPPORTED_CONTAINERS = set([])
    def __init__(self):
        self.ops = []
        self.branches = {}

    def __le__(self, other):
        '''Add a magic block to the Container'''
        assert isinstance(other, MagicBlock), 'Must be Magic'
        assert self.NAME in other.SUPPORTED_CONTAINERS, repr(other) +' does not support %s' % self.NAME
        self.ops.append(other)
        return self

    def __call__(self, inp, trainable, weights_collections, **extra_params):
        '''Returns a MagicCapsule - a handle to the model that allows saving, specific block access and more.'''
        raise NotImplementedError()

    def __setitem__(self, key, container):
        if not isinstance(container, MagicContainer):
            raise TypeError('MagicContainer required')
        if key in self.branches:
            raise ValueError('Branch "%s" already exists' % key)
        if not isinstance(key, basestring) or key.lower() != key:
            raise ValueError('branch name must be a lowercase string to avoid confusion with blocks')
        self.branches[key] = container
        self <= FakeBlock(container, key)


    def __getitem__(self, key):
        if key not in self.branches:
            raise ValueError('No such branch "%s"' % key)
        return self.branches[key]

class FakeBlock(MagicBlock):
    SUPPORTED_CONTAINERS = {'LinearContainer', 'UNetContainer'}
    def __init__(self, container, name):
        self.container = container
        self.NAME = name + '_'

    # todo dynamically handle requirements! via @property requirements

    def __call__(self, inp, trainable, weights_collections, **extra_params):
        return self.container(inp, trainable, weights_collections, **extra_params)


class MagicCapsule:
    def __init__(self, output, resident_scope_name, weights_collections):
        self.output = output
        self.resident_scope_name = resident_scope_name
        self.weights_collections = weights_collections

    def __getitem__(self, output_type):
        if output_type in self.output:
            return self.output[output_type]
        if output_type+'_0' in self.output: # branches have 0 appended to their name...
            return self.output[output_type+'_0']
        raise TypeError('No such output '+str(output_type))

    @property
    def final_output(self):
        return self.output['final_output']

    def get_own_variables(self):
        return tf.get_collection(tf.GraphKeys().GLOBAL_VARIABLES, scope=self.resident_scope_name)

    def get_own_weights(self):
        assert self.weights_collections
        return tf.get_collection(self.weights_collections[-1])

    def get_num_params(self):
        s = 0
        for e in self.get_own_weights():
            s += np.prod(e.get_shape().as_list())
        return s

    def get_own_l2_loss(self):
        print 'Number of params in weights', self.get_num_params()
        return sum(map(tf.nn.l2_loss, self.get_own_weights()), tf.constant(0.))


def satisfy_reqs(op, out_map):
    return {}

class LinearContainer(MagicContainer):
    SUPPORTED_CONTAINERS = {'LinearContainer', }
    def __call__(self, inp, trainable, weights_collections, **extra_params):
        out = inp
        out_map = {}
        with tf.variable_scope(None, default_name=self.NAME) as scope:
            for op in self.ops:
                print out
                res = op(out, trainable, weights_collections, **satisfy_reqs(op, out_map))
                cand = op.NAME
                i = 0
                while cand+str(i) in out_map:
                    i+=1
                name = cand+str(i)
                out_map[name] = res
                if not isinstance(op, FakeBlock):
                    out = res['final_output']
            scope_name = scope.name
        out_map['final_output'] = out
        return MagicCapsule(out_map, scope_name, weights_collections)


class UNetContainer(MagicContainer):
    ''' Allows Blocks to have receive_passthrough attribute that allows the Block to receive 
       previously calculated input of specified resolution as passthrough extra_param. Based on UNet.
       receive_passthrough should be int. Assuming that the block is receiving input of resolution X 
        passthrough will have resolution equal to _passthrough_ratio*X'''
    SUPPORTED_CONTAINERS = {'LinearContainer', }
    def __call__(self, inp, trainable, weights_collections, **extra_params):
        out = inp
        out_map = {}
        resolution_map = {}
        with tf.variable_scope(None, default_name=self.NAME) as scope:
            for op in self.ops:
                print out
                deps = satisfy_reqs(op, out_map)
                if hasattr(op, 'receive_passthrough') and op.receive_passthrough:
                    required_resolution = out.get_shape().as_list()[2]*op._passthrough_ratio

                    if required_resolution not in resolution_map:
                        raise ValueError('Specified resolution %d not found for a block %s, probably too many upsamplers :)' % (required_resolution, repr(op)))
                    deps['passthrough'] = resolution_map[required_resolution]
                res = op(out, trainable, weights_collections, **deps)
                cand = op.NAME
                i = 0
                while cand + str(i) in out_map:
                    i += 1
                name = cand + str(i)
                out_map[name] = res
                if not isinstance(op, FakeBlock):
                    out = res['final_output']
                resolution = out.get_shape().as_list()[2]
                if resolution not in resolution_map:
                    resolution_map[resolution] = out
            scope_name = scope.name
        out_map['final_output'] = out
        return MagicCapsule(out_map, scope_name, weights_collections)



class DenseBlock(MagicBlock):
    SUPPORTED_CONTAINERS = {'LinearContainer', 'UNetContainer'}
    def __init__(self, layers, growth_factors, dilate_after=None, bottleneck=0.25, compress_final=0.5):
        ''' layers specifies how many layers given dense block should have
            growth_factors specifies growth_factor for each layer (if int constant for all)
            dilate_after is a list of layers after which dilation should be increased by 2, if None then no dilation
            
            compress final is whether to apply compression at the end of the block. I know that it should be in the transition 
            layer but I added it here for convenience. if compress final is 1 or false then no compression is applied.
            '''
        self.layers = layers
        self.growth_factors = growth_factors if type(growth_factors)!=int else self.layers*[growth_factors]
        self.dilations = [1]*layers
        if dilate_after is not None:
            for k in dilate_after:
                while k<len(self.dilations):
                    self.dilations[k] *= 2
                    k += 1

        self.bottleneck = bottleneck
        self.compress_final = compress_final

    def __call__(self, inp, trainable, weights_collections, **extra_params):
        with tf.variable_scope(None, default_name=self.NAME):
            out = dense_net.DenseBlock(growth_rate=self.growth_factors, layers=self.layers,
                             bottleneck=self.bottleneck, trainable=trainable,
                             weights_collections=weights_collections, dilation_factors=self.dilations)(inp)
            if self.compress_final and self.compress_final!=1:
                out = dense_net.DimReductionLayer(self.compress_final,
                                                  trainable=trainable,
                                                  weights_collections=weights_collections)(out)
        return {'final_output': out}


class TransitionBlock(MagicBlock):
    SUPPORTED_CONTAINERS = {'LinearContainer', 'UNetContainer'}
    def __init__(self, keep_dim_fraction=1, kernel_size=2, stride=2, pool_method=tf_layers.avg_pool2d):
        self.keep_dim_fraction = keep_dim_fraction
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool_method = pool_method

    def __call__(self, inp, trainable, weights_collections, **extra_params):
        with tf.variable_scope(None, default_name=self.NAME):
            out = dense_net.TrainsitionLayer(keep_dim_fraction=self.keep_dim_fraction,
                                             trainable=trainable,
                                             weights_collections=weights_collections,
                                             kernel_size=self.kernel_size,
                                             stride=self.stride,
                                             pool_method=self.pool_method)(inp)
        return {'final_output': out}

class BottleneckBlock(MagicBlock):
    SUPPORTED_CONTAINERS = {'LinearContainer', 'UNetContainer'}
    def __init__(self, num_chans, layers=1, stride=1, scale=False, activation_fn=tf.nn.relu):
        self.num_chans = num_chans
        self.stride = stride
        self.scale = scale
        self.activation_fn = activation_fn
        self.layers = layers

    def __call__(self, inp, trainable, weights_collections, **extra_params):
        with tf.variable_scope(None, default_name=self.NAME):
            for _ in xrange(self.layers):
                with tf.variable_scope(None, default_name='BottleneckUnit'):
                    out = dense_net.bottleneck_block(inp, self.num_chans, stride=self.stride, training=trainable,
                                                     weights_collections=weights_collections, scale=self.scale,
                                                     activation=self.activation_fn)
        return {'final_output': out}


class SimpleCNNBlock(MagicBlock):
    SUPPORTED_CONTAINERS = {'LinearContainer', 'UNetContainer'}
    def __init__(self, layers, num_chans, conv_kernel_size, conv_stride=1, follow_with_bn=False, activation_fn=tf.nn.relu):
        assert layers != 0
        self.layers = layers
        self.num_chans = num_chans
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.activation_fn = activation_fn
        self.follow_with_bn = follow_with_bn

    def __call__(self, inp, trainable, weights_collections, **extra_params):
        batch_norm_params = {'updates_collections': None, 'is_training': trainable, 'trainable': trainable} if self.follow_with_bn else None
        assert not isinstance(weights_collections, basestring), 'Must be a list of collections!'
        variable_collections = None if weights_collections is None else {'weights': weights_collections}
        out = inp
        with tf.variable_scope(None, default_name=self.NAME):
            for layer in xrange(self.layers):
                with tf.variable_scope(None, default_name='MagicCNNLayer'):
                    out = tf_layers.conv2d(out, self.num_chans, self.conv_kernel_size, self.conv_stride,
                                           variables_collections=variable_collections,
                                           trainable=trainable,
                                           activation_fn=self.activation_fn,
                                           normalizer_fn=tf_layers.batch_norm if self.follow_with_bn else None,
                                           normalizer_params=batch_norm_params,
                                           weights_initializer=tf_layers.variance_scaling_initializer(2.),)
        return {'final_output': out}


class SimpleLinearBlock(MagicBlock):
    SUPPORTED_CONTAINERS = {'LinearContainer',}
    def __init__(self, num_chans, layers=1, follow_with_bn=False, activation_fn=tf.nn.relu):
        assert layers != 0
        self.layers = layers
        self.num_chans = num_chans
        self.activation_fn = activation_fn
        self.follow_with_bn = follow_with_bn

    def __call__(self, inp, trainable, weights_collections, **extra_params):
        assert len(inp.get_shape().as_list()) == 2, 'input to SimpleLinearBlock must be 2D, use flatten first'

        batch_norm_params = {'updates_collections': None, 'is_training': trainable, 'trainable': trainable} if self.follow_with_bn else None
        assert not isinstance(weights_collections, basestring), 'Must be a list of collections!'
        variable_collections = None if weights_collections is None else {'weights': weights_collections}
        out = inp
        with tf.variable_scope(None, default_name=self.NAME):
            for layer in xrange(self.layers):
                with tf.variable_scope(None, default_name='MagicLinearLayer'):
                    out = tf_layers.fully_connected(out, self.num_chans,
                                           variables_collections=variable_collections,
                                           trainable=trainable,
                                           activation_fn=self.activation_fn,
                                           normalizer_fn=tf_layers.batch_norm if self.follow_with_bn else None,
                                           normalizer_params=batch_norm_params,
                                           weights_initializer=tf_layers.variance_scaling_initializer(2.),)
        return {'final_output': out}


class FlattenBlock(MagicBlock):
    SUPPORTED_CONTAINERS = {'LinearContainer',}
    def __init__(self):
        pass

    def __call__(self, inp, trainable, weights_collections, **extra_params):
        out = inp
        with tf.variable_scope(None, default_name=self.NAME):
            out = tf_layers.flatten(out)
        return {'final_output': out}


class CustomLinearBlock(MagicBlock):
    SUPPORTED_CONTAINERS = {'LinearContainer',}
    def __init__(self, func):
        self.func = func

    def __call__(self, inp, trainable, weights_collections, **extra_params):
        with tf.variable_scope(None, default_name=self.NAME):
            out = self.func(inp)
        return {'final_output': out}





class UpsamplerBlock(MagicBlock):
    SUPPORTED_CONTAINERS = {'LinearContainer', 'UNetContainer'}

    def __init__(self, num_chans, receive_passthrough, follow_up_residual_blocks, passthrough_relative_chan_size=1, deconv_kernel_size=2, deconv_stride=2, activation_fn=tf.nn.elu):
        self.num_chans = num_chans
        self.deconv_kernel_size = deconv_kernel_size
        self.deconv_stride =deconv_stride
        self.activation_fn = activation_fn
        self.receive_passthrough = receive_passthrough
        self.follow_up_residual_blocks = follow_up_residual_blocks
        self.passthrough_relative_chan_size = passthrough_relative_chan_size
        self._passthrough_ratio = self.deconv_stride


    def __call__(self, inp, trainable, weights_collections, **extra_params):
        batch_norm_params = {'updates_collections': None, 'is_training': trainable,
                             'trainable': trainable}
        assert not isinstance(weights_collections, basestring), 'Must be a list of collections!'
        variable_collections = None if weights_collections is None else {'weights': weights_collections}


        with tf.variable_scope(None, default_name=self.NAME):
            # first standard deconv
            out = tf_layers.conv2d_transpose(inp, self.num_chans, self.deconv_kernel_size, stride=self.deconv_stride,
                                             activation_fn=self.activation_fn,
                                             normalizer_fn=tf_layers.batch_norm,
                                             normalizer_params=batch_norm_params,
                                             variables_collections=variable_collections,
                                             trainable=trainable)

            if self.receive_passthrough:
                assert self.follow_up_residual_blocks, 'You must follow with at least one residual block if you use passthrough!'
                assert 'passthrough' in extra_params, 'Did not receive the passthrough'
                passthrough = extra_params['passthrough']

                # the question is: should we add the passthrough or concat as extra channels?
                # if concat then use batch_norm + activation, otherwise not but has to have the same num of channels
                # will use concat for now
                if self.passthrough_relative_chan_size != 1:
                    ext = tf_layers.conv2d(passthrough,
                                           int(passthrough.get_shape().as_list()[-1] * self.passthrough_relative_chan_size), 1,
                                           stride=1,
                                           activation_fn=None,
                                           normalizer_fn=tf_layers.batch_norm,
                                           normalizer_params=batch_norm_params,
                                           variables_collections=variable_collections,
                                           trainable=trainable)
                else:
                    ext = passthrough

                out = tf.concat((out, ext), 3)

            if self.follow_up_residual_blocks:
                if not isinstance(self.follow_up_residual_blocks, int):
                    blocks = 1
                else:
                    blocks = self.follow_up_residual_blocks
                for _ in xrange(blocks):
                    out = dense_net.bottleneck_block(out, self.num_chans, stride=1, training=trainable,
                                           weights_collections=weights_collections, scale=False,
                                           activation=self.activation_fn)
        return {'final_output': out}






def output_channel_ranges_from_mean_std(mean, std, initial_range=(0, 255)):
    ''' Each RGB channel is normalised so that its mean is 0 and std is 1. Initial range of channels is usually <0, 255>.
       This function returns the ranges of channels after normalisation as a np.array (CHANS, 2)
       You can use it with to_image_channels function. 
       '''
    new_mean = ((initial_range[0]+initial_range[1])/2. - mean)/std
    new_range = (initial_range[1]-initial_range[0]) / std
    return np.concatenate((np.expand_dims(new_mean - new_range/2., 1), np.expand_dims(new_mean + new_range/2. , 1)), 1)


class ToImageChannelsBlock(MagicBlock):
    def __init__(self):
        raise NotImplementedError()

    def to_image_channels(inp, num_channels, output_channel_ranges, trainable=True, weights_collections=None, nonlinearity=tf.nn.tanh, nonlinearity_range=(-1, 1)):
        ''' for each channel you must supply a range (min_val, max_val) as an array (CHANS, 2)'''
        assert len(output_channel_ranges) == num_channels
        assert not isinstance(weights_collections, basestring), 'Must be a list of collections!'
        variable_collections = None if weights_collections is None else {'weights': weights_collections}
        print output_channel_ranges
        with tf.variable_scope(None, default_name='ToImageChannels'):
            out = tf_layers.conv2d(inp, num_channels, 1,
                                   stride=1,
                                   activation_fn=nonlinearity,
                                   trainable=trainable,
                                   variables_collections=variable_collections)
            nonlinearity_mean = sum(nonlinearity_range) / 2.
            nonlinearity_spread = float(nonlinearity_range[1]) - nonlinearity_range[0]
            output_channel_ranges = np.array(output_channel_ranges)
            output_channel_means = np.mean(output_channel_ranges, 1)
            output_channel_spreads = output_channel_ranges[:, 1] - output_channel_ranges[:, 0]
            return (out - nonlinearity_mean) / nonlinearity_spread * output_channel_spreads + output_channel_means




def to_classification_layer(inp, num_classes, trainable=True, weights_collections=None):
    assert not isinstance(weights_collections, basestring), 'Must be a list of collections!'
    variables_collections = None if weights_collections is None else {'weights': weights_collections}
    with tf.variable_scope(None, default_name='ClassificationLayer'):
        out = tf.reduce_mean(inp, (1,2))
        out = tf_layers.fully_connected(out, num_classes,
                               activation_fn=None,
                               variables_collections=variables_collections,
                               trainable=trainable)
    return out







