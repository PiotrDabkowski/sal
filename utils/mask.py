import tensorflow as tf


def smoothness_loss(mask, power=2, border_penalty=True):
    n, x, y = mask.get_shape().as_list()
    if border_penalty:
        mx1 = tf.concat((mask, tf.zeros((n, 1, y))), 1)
        mx2 = tf.concat((tf.zeros((n, 1, y)), mask), 1)
        x_loss = tf.reduce_mean(tf.abs(mx1-mx2) ** power)
        my1 = tf.concat((mask, tf.zeros((n, x, 1))), 2)
        my2 = tf.concat((tf.zeros((n, x, 1)), mask), 2)
        y_loss = tf.reduce_mean(tf.abs(my1 - my2) ** power)
    else:
        x_loss = tf.reduce_mean((tf.abs(mask[:, 1:, :] - mask[:, :-1, :])) ** power)
        y_loss = tf.reduce_mean((tf.abs(mask[:, :, 1:] - mask[:, :, :-1])) ** power)
    return (x_loss + y_loss) / 2.


def area_loss(mask, power=1., satisfactory_reduction=0.):
    assert 0.<=satisfactory_reduction<=1.
    assert len(mask.get_shape().as_list()) == 3, 'Mask must have shape (BATCHES, H, W)'
    adjusted_reduction = satisfactory_reduction ** power if satisfactory_reduction else 0.
    if power != 1:
        mask = (mask+0.0005)**power
    return tf.reduce_mean(tf.nn.relu(tf.reduce_mean(mask, (1,2))-adjusted_reduction)+adjusted_reduction)


def apply_mask(images, mask, noise=False, random_colors=False, noise_std=0.11, color_range=0.33, bypass=0., boolean=False):
    assert 0.<=bypass<0.9
    if boolean:
        print 'Warning using boolean mask, it\'s just for validation!'
        return tf.expand_dims(tf.cast(tf.greater(mask, 0.5), tf.float32), 3) *images
    if bypass > 0:
        mask = (1.-bypass)*mask + bypass
    if noise and noise_std:
        alt = tf.random_normal(images.get_shape().as_list(), 0., noise_std, dtype=tf.float32)
    else:
        alt = tf.zeros_like(images, dtype=tf.float32)
    if random_colors:
        n, _, _, c = images.get_shape().as_list()
        alt += tf.random_uniform((n, 1, 1, c), -color_range, color_range, dtype=tf.float32)
    expanded_mask = tf.expand_dims(mask, 3)
    return (expanded_mask*images) + (1. - expanded_mask)*alt


