import tensorflow as tf
import gaussian

def smoothness_loss(mask, power=2, border_penalty=True):
    ''' For a given image this loss should be more or less invariant to image resize when using power=2...
        let L be the length of a side
        EdgesLength ~ L
        EdgesSharpness ~ 1/L, easy to see if you imagine just a single vertical edge in the whole image'''
    n, x, y = mask.get_shape().as_list()
    if border_penalty:
        mx1 = tf.concat((mask, tf.zeros((n, 1, y))), 1)
        mx2 = tf.concat((tf.zeros((n, 1, y)), mask), 1)
        x_loss = tf.reduce_sum(tf.abs(mx1-mx2) ** power)
        my1 = tf.concat((mask, tf.zeros((n, x, 1))), 2)
        my2 = tf.concat((tf.zeros((n, x, 1)), mask), 2)
        y_loss = tf.reduce_sum(tf.abs(my1 - my2) ** power)
    else:
        x_loss = tf.reduce_sum((tf.abs(mask[:, 1:, :] - mask[:, :-1, :])) ** power)
        y_loss = tf.reduce_sum((tf.abs(mask[:, :, 1:] - mask[:, :, :-1])) ** power)
    return (x_loss + y_loss) / float(power * n)  # watch out, normalised by batch size!


def area_loss(mask, power=1., satisfactory_reduction=0.):
    assert 0.<=satisfactory_reduction<=1.
    assert len(mask.get_shape().as_list()) == 3, 'Mask must have shape (BATCHES, H, W)'
    adjusted_reduction = satisfactory_reduction ** power if satisfactory_reduction else 0.
    if power != 1:
        mask = (mask+0.0005)**power # prevent nan (derivative of sqrt at 0 is inf)
    return tf.reduce_mean(tf.nn.relu(tf.reduce_mean(mask, (1,2))-adjusted_reduction)+adjusted_reduction)


def apply_mask(images, mask, noise=False, random_colors=False, blurred_version_prob=0, noise_std=0.11, color_range=0.33, blur_sigma=8, blur_kernel=33, bypass=0., boolean=False):
    assert 0.<=bypass<0.9
    n, _, _, c = images.get_shape().as_list()
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
        alt += tf.random_uniform((n, 1, 1, c), -color_range, color_range, dtype=tf.float32)
    if blurred_version_prob: # <- it can be a scalar between 0 and 1
        when = tf.cast(tf.random_uniform((n, 1, 1, 1), 0., 1., dtype=tf.float32) < float(blurred_version_prob), tf.float32)
        cand = gaussian.gaussian_blur(images, blur_kernel, blur_sigma)
        alt = alt*(1.-when) + cand*when
    expanded_mask = tf.expand_dims(mask, 3)
    return (expanded_mask*images) + (1. - expanded_mask)*alt


