import tensorflow as tf
from resnet_utils import *

CKPT = 'resnet50/ResNet-L50'

def _inference(x, is_training,
               reuse,
               num_classes=1000,
               num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
               use_bias=False,  # defaults to using batch norm
               bottleneck=True):
    scope = tf.get_variable_scope()
    old_reuse = scope.reuse
    scope._reuse = reuse

    c = Config()
    c['bottleneck'] = bottleneck
    c['is_training'] = tf.convert_to_tensor(is_training,
                                            dtype='bool',
                                            name='is_training')
    c['is_training_py'] = is_training
    c['ksize'] = 3
    c['stride'] = 1
    c['use_bias'] = use_bias
    c['fc_units_out'] = num_classes
    c['num_blocks'] = num_blocks
    c['stack_stride'] = 2

    with tf.variable_scope('scale1'):
        c['conv_filters_out'] = 64
        c['ksize'] = 7
        c['stride'] = 2
        x = conv(x, c)
        x = bn(x, c)
        x = activation(x)

    with tf.variable_scope('scale2'):
        x = max_pool(x, ksize=3, stride=2)
        c['num_blocks'] = num_blocks[0]
        c['stack_stride'] = 1
        c['block_filters_internal'] = 64
        x = stack(x, c)

    with tf.variable_scope('scale3'):
        c['num_blocks'] = num_blocks[1]
        c['block_filters_internal'] = 128
        assert c['stack_stride'] == 2
        x = stack(x, c)

    with tf.variable_scope('scale4'):
        c['num_blocks'] = num_blocks[2]
        c['block_filters_internal'] = 256
        x = stack(x, c)

    with tf.variable_scope('scale5'):
        c['num_blocks'] = num_blocks[3]
        c['block_filters_internal'] = 512
        x = stack(x, c)

    # post-net
    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
    avg_pool = x

    if num_classes != None:
        with tf.variable_scope('fc'):
            x = fc(x, c)

    scope._reuse = old_reuse
    return x, avg_pool


def inference(images, trainable, reuse, return_avg_pool=False):
    x = imagenet_preprocess((images * imagenet.IMAGE_NET_PIXEL_STD + imagenet.IMAGE_NET_PIXEL_MEAN) / 255.0)
    x, avg_pool = _inference(x, trainable, reuse, num_classes=1000, num_blocks=[3, 4, 6, 3], use_bias=False, bottleneck=True)
    return x if not return_avg_pool else (x, avg_pool)