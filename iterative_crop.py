# A very simple, iterative solution to the masking problem
# Given an image and a trained classifier produces a mask for the selected class
# Here we take images from ImageNet validation set.


# Following params were found to work OK:
# Adam, learning rate 0.1,  450 steps ( use smaller learning rates, like 0.02 when applying loss to resized mask, bilinear grad a bit unstable)
# Mean area loss = 0.3
# Total variation loss = 0.15 (when applied to the unresized mask)
# Preservation loss factor = 1  (calculated as a abs distance from 1 softmax prob)
# Destroyer loss factor = 2 (calculated as a third root of abs distance from 0 softmax prob)
# Unresized mask size = 25
# applying blurring to the resized mask is not necessary

# These were all found for batch size of 16 so you have to divide by (16/NEW_BS) when using other batch size or adjust the learning rate.
# The ratio between params will stay the same...

import random
import numpy as np
np.random.seed(11)
random.seed(11)
import tensorflow as tf
from datasets import imagenet
import utils.continous_crop
import utils.gaussian
import utils.meta_restorer
import utils.bounding_box
from blocks import resnet, alexnet, vgg16
import cv2


# You can choose either resnet, vgg16 or alexnet (you must have their checkpoints though)
CHOSEN_MODEL = resnet
CKPT = CHOSEN_MODEL.CKPT

# How many images to optimise for at the given time
MASK_SIZE = 25   # if lower it will be resized to 224 anyway...
BATCH_SIZE = 1
assert  BATCH_SIZE == 1

images = tf.get_variable('images', (BATCH_SIZE, 224, 224, 3), dtype=tf.float32, trainable=False)
labels = tf.get_variable('labels', (BATCH_SIZE,), dtype=tf.int32, trainable=False)



X = tf.get_variable('x', (), tf.float32, trainable=True)
Y = tf.get_variable('y', (), tf.float32, trainable=True)

W = tf.get_variable('w', (), tf.float32, trainable=True)
H = tf.get_variable('h', (), tf.float32, trainable=True)


noisy_images = utils.gaussian.gaussian_blur(tf.random_normal(images.get_shape().as_list(), 0, 0.01) + images, 15, 1 )
preserved_images = tf.expand_dims(utils.continous_crop.continous_crop(noisy_images[0], X, Y, W, H, (224, 224)), 0)

# feed the resulting images to the model
with tf.variable_scope('TrainedClassifierModel'):
    all_scores = CHOSEN_MODEL.inference(preserved_images, False, False)
preserved_scores = all_scores
init_op = utils.meta_restorer.restore_in_scope(CKPT, 'TrainedClassifierModel')


# now the losses...
area_loss = tf.reduce_sum(W*H)

# not sure how these 2 should exactly be defined...
#preservation_loss = tf.reduce_mean(utils.loss_calc.abs_distance_loss(logits=tf.nn.softmax(preserved_scores), labels=labels, ref=1.))
preservation_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preserved_scores, labels=labels))
#destroyer_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=-destroyed_scores, labels=labels))

# compose the full loss, area and smoothness coefs are quite important

full_loss =  2*area_loss + preservation_loss


with tf.variable_scope('adam'):
    opt = tf.train.AdamOptimizer(0.011, 0.9)

    grads = opt.compute_gradients(full_loss, var_list=[X, Y, W, H])
    clipped_grads = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in grads]
    opt_op = opt.apply_gradients(clipped_grads)





assert len(tf.trainable_variables())==4

# this is how you reset the images, just run reset_all with new_images and new_labels in feed_dict
new_images = tf.placeholder(tf.float32, (BATCH_SIZE, 224, 224, 3))
new_labels = tf.placeholder(tf.int32, (BATCH_SIZE,))
reset_all = tf.group(
    tf.assign(images, new_images),
    tf.assign(labels, new_labels),
    tf.assign(X, 0.01),
    tf.assign(Y, 0.01),
    tf.assign(W, 0.98),
    tf.assign(H, 0.98),
)

reposition_params = tf.group(
    tf.assign(X, tf.clip_by_value(X, 0.006, 0.994)),
    tf.assign(Y, tf.clip_by_value(Y, 0.006, 0.994)),
    tf.assign(W, tf.clip_by_value(W, 0.05, 0.995-X)),
    tf.assign(H, tf.clip_by_value(H, 0.05, 0.995-Y)),

)

boxes = imagenet.get_boxes_by_img_num()

with tf.Session() as sess:
    sess.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'adam')))
    sess.run(init_op)
    val_bm = imagenet.get_val_bm(BATCH_SIZE)
    tot = 0
    for path in imagenet.VAL_IMAGES:
        raw_img = imagenet.load_normalized_without_resize(path)
        im_h, im_w, c = raw_img.shape
        if im_w < im_h:
            scale = 224. / im_h
            im_w = int(scale*im_w)
            im_h = 224
            img = np.concatenate((cv2.resize(raw_img, (im_w, im_h), interpolation=cv2.INTER_LINEAR),
                                  np.zeros((im_h, 224-im_w, c))), 1)
        else:
            scale = 224. / im_w
            im_h = int(scale * im_h)
            im_w = 224
            img = np.concatenate((cv2.resize(raw_img, (im_w, im_h), interpolation=cv2.INTER_LINEAR),
                                  np.zeros((224-im_h, im_w, c))), 0)
        label = imagenet.LABEL_VAL_PIPELINE(path)
        sess.run(reset_all, {
            new_images: np.expand_dims(img, 0),
            new_labels: label
        })
        i = 0
        min_loss = float('inf')
        while i<451:
            _, al, pl, x, y, w, h, pi, di, fl = sess.run([opt_op, area_loss, preservation_loss, X, Y, W, H, images, preserved_images, full_loss])
            print i, 'A: %.2f   P: %.2f   X: %.3f    Y: %.3f    W: %.3f    H: %.3f' % (al, pl, x, y, w, h)
            if fl < min_loss:
                x0 = int(x * 224)
                y0 = int(y * 224)
                w0 = int(w * 224) - 1
                h0 = int(h * 224) - 1
                min_loss = fl
            if not i%30:
                pi0 = imagenet.to_bgr_img(pi[0])

                pi0 = cv2.rectangle(pi0.copy(), (x0,y0), (x0+w0, y0+h0), (0,255,0), 2)
                for box in boxes[imagenet.img_num_from_path(path)].values()[0]:
                    box = map(lambda x: int(x*scale), box)
                    pi0 = cv2.rectangle(pi0.copy(), (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                    print 'IOU', utils.bounding_box.intersection_over_union(box, utils.bounding_box.box_from_xywh(x0, y0, w0, h0))
                di0 = imagenet.to_bgr_img(di[0])

                cv2.imwrite('iter.jpg', np.concatenate((pi0, di0), 0))
            i+=1
            sess.run(reposition_params)

        raw_input("done")






