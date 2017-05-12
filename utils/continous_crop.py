import tensorflow as tf
import  numpy as np
import cv2

# Nasty piece of code, differentiable crop

def get_crop_coords(x, y, w, h):
    th2 = .999
    x2 = th2 - th2 * th2 * tf.nn.relu(1. - x - w)
    y2 = th2 - th2 * th2 * tf.nn.relu(1. - y - h)

    th1 = th2 ** 2
    x1 = th1 * x
    y1 = th1 * y
    return x1, y1, x2, y2


def continous_crop(img, x, y, w, h, target_size=None):
    ''' everything must be a tensor.
        img has to have a fixed shape (H, W, C)
        x, y, w, h values must in [0, 1]

        Uses bilinear interpolation.
        returned shape is at least 2x2 and is as close to the requested crop shape as possible. no guarantees though

        Differentiable vs x, y, w, h!

        Probably would be best if implemented as a native op...
        # optional target_size is (h, w) tuple where h, w can be a tensor of any type.
        '''
    im_h, im_w, im_c = img.get_shape().as_list()

    # bilinear interpolation params
    a0 = img[:-1, :-1, :]
    ax = img[:-1, 1:, :] - img[:-1, :-1, :]
    ay = img[1:, :-1, :] - img[:-1, :-1, :]
    axy = img[1:, 1:, :] - a0 - ax - ay

    # quite complex but ensures that range is well behaved.
    x1, y1, x2, y2 = get_crop_coords(x, y, w, h)


    if target_size is None:
        req_w = tf.cast(tf.ceil((x2 - x1)*im_w), tf.int32)  # at least 2x2
        req_h = tf.cast(tf.ceil((y2 - y1)*im_h), tf.int32)
    else:    # BONUS: this also implements bilinear resize, you can choose any required size and specify it as a tuple
        assert len(target_size) == 2, 'target_size must be a tuple'
        req_h = tf.cast(target_size[0], tf.int32)
        req_w = tf.cast(target_size[1], tf.int32)

    # now the fun begins...
    # we just have to create a mask with x, y coords

    shape = tf.stack((req_h, req_w))


    step_x = (x2 - x1)*(im_w-1) / tf.cast(req_w-1, tf.float32) # this cast will make the result not differentiable vs target size as required.
    step_y = (y2 - y1)*(im_h-1) / tf.cast(req_h-1, tf.float32)

    _x_coords = tf.cast(tf.range(req_w), tf.float32) * step_x
    _y_coords = tf.cast(tf.range(req_h), tf.float32) * step_y

    x_coords = x1*(im_w-1)*tf.ones(shape) + tf.expand_dims(_x_coords, 0)
    y_coords = y1*(im_h-1)*tf.ones(shape) + tf.expand_dims(_y_coords, 1)

    # now calculate integer coords and their offsets
    _base_x = tf.floor(x_coords)
    _base_y = tf.floor(y_coords)
    off_x = tf.expand_dims(x_coords - _base_x, 2)
    off_y = tf.expand_dims(y_coords - _base_y, 2)
    base_x = tf.cast(_base_x, tf.int32)
    base_y = tf.cast(_base_y, tf.int32)

   # a0 = tf.Print(a0, [req_h, req_w, x1+step_x*3, x1*(im_w-1)])
    # now use base_x and base_y to create a matrix of this shape from a0, ax, ay and axy...
    _a0, _ax, _ay, _axy = (index_by_matrices(a0, base_x, base_y, shape),
                           index_by_matrices(ax, base_x, base_y, shape),
                           index_by_matrices(ay, base_x, base_y, shape),
                           index_by_matrices(axy, base_x, base_y, shape))


    # they all have the same shape as the required crop
    return _a0 + off_x*_ax + off_y*_ay + off_x*off_y*_axy


def index_by_matrices(tensor, x_coords, y_coords, shape):
    # tensor must be 3d, indexing is 2d based
    # I will use tf.gather so we have to flatten everything...
    size_y, size_x, c = tensor.get_shape().as_list()

    gatherable = tf.reshape(tensor, (size_x*size_y, c))
    indices = tf.reshape(y_coords, (-1,))*size_x + tf.reshape(x_coords, (-1,))

    g = tf.gather(gatherable, indices)
    new_shape = tf.concat((shape, tf.expand_dims(c, 0)), 0)
    return tf.reshape(g, new_shape)



def crop_and_resize_and_place_on_square_matrix(img, x, y, w, h, matrix_side=300):
    x1, y1, x2, y2 = get_crop_coords(x, y, w, h)
    dx = (x2 - x1)
    dy = (y2 - y1)
    matrix_side = tf.convert_to_tensor(matrix_side, dtype=tf.float32)
    sx = tf.cond(dx > dy, lambda: matrix_side, lambda: matrix_side * dx / dy)
    sy = tf.round(sx * dy / dx)
    sx = tf.round(sx)
    sx = tf.nn.relu(sx - 2) + 2
    sy = tf.nn.relu(sy - 2) + 2

    chans = float(img.get_shape().as_list()[-1])

    def x_larger():
        yy_left = tf.floor((matrix_side - sy)/2.)
        yy_right = matrix_side - yy_left - sy
        left_shape = tf.cast(tf.stack((yy_left, matrix_side, chans)), tf.int32)
        right_shape = tf.cast(tf.stack((yy_right, matrix_side, chans)), tf.int32)
        return tf.concat((tf.zeros(left_shape), resized, tf.zeros(right_shape)), 0)

    def y_larger():
        xx_left = tf.floor((matrix_side - sx)/2.)
        xx_right = matrix_side - xx_left - sx
        left_shape = tf.cast(tf.stack((matrix_side, xx_left, chans)), tf.int32)
        right_shape = tf.cast(tf.stack((matrix_side, xx_right, chans)), tf.int32)
        return tf.concat((tf.zeros(left_shape), resized, tf.zeros(right_shape)), 1)


    # sx = tf.Print(sx, [sy, sx])
    resized = continous_crop(img, x, y, w, h, target_size=(sy, sx))
    return tf.cond(dx > dy,
                   x_larger,
                   y_larger)


def batch_crop_and_resize_and_place_on_square_matrix(imgs, crop_params, matrix_side):
    res = []
    for im, params in zip(tf.unstack(imgs), tf.unstack(crop_params)):
        print params.get_shape()
        x, y, w, h = tf.unstack(params)
        res.append(crop_and_resize_and_place_on_square_matrix(im, x, y, w, h, matrix_side))
    return tf.stack(res)

