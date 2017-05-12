import numpy as np
import tensorflow as tf

def _gaussian_kernel(kernel_size, sigma, chans):
    """Invented my own, hope it works..."""
    assert kernel_size % 2, 'Kernel size must be odd!'
    x = np.expand_dims(np.array(range(-kernel_size/2, -kernel_size/2+kernel_size, 1)), 0)
    vals = np.exp(-np.square(x)/(2.*sigma**2))
    kernel_raw = np.matmul(vals.T, vals)
    kernel = np.reshape(kernel_raw / np.sum(kernel_raw), (kernel_size, kernel_size, 1, 1))
    repeated =  np.zeros((kernel_size, kernel_size, chans, 1), dtype=np.float32) + kernel
    return tf.constant(repeated, tf.float32)

def gaussian_blur(imgs, kernel_size=21, sigma=5):
    ''' Applies gaussian blur to the imgs. Imgs must be [N, H, W, C] and remember that kernel size should be larger than 4*sigma'''
    chans = imgs.get_shape().as_list()[-1]
    return tf.nn.depthwise_conv2d(imgs, _gaussian_kernel(kernel_size, sigma, chans), [1,1,1,1], 'SAME')

