import tensorflow as tf
import numpy as np

def get_by_labels(x, labels):
    '''X is [BS, CLASSES], labels is [BS,]
      Returns [BS,] where each entry corresponds to the value of X for given label'''
    bs, num_classes = x.get_shape().as_list()
    flat = tf.reshape(x, (-1,))
    offsets = tf.constant(np.arange(bs) * num_classes, dtype=tf.int32)
    vals = tf.gather(flat, offsets + tf.cast(labels, tf.int32))
    return vals

def abs_distance_loss(logits, labels, ref):
    return tf.abs(get_by_labels(logits, labels)-ref)
