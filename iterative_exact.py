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
import utils.mask
import utils.meta_restorer
import utils.bounding_box
import utils.loss_calc
import utils.gaussian
from blocks import resnet, alexnet, vgg16
import cv2


# You can choose either resnet, vgg16 or alexnet (you must have their checkpoints though)
CHOSEN_MODEL = resnet
CKPT = resnet.CKPT

# How many images to optimise for at the given time
BATCH_SIZE = 2
MASK_SIZE = 28   # if lower it will be resized to 224 anyway...

images = tf.get_variable('images', (BATCH_SIZE, 224, 224, 3), dtype=tf.float32, trainable=False)
labels = tf.get_variable('labels', (BATCH_SIZE,), dtype=tf.int32, trainable=False)



vals = tf.get_variable('mvals', (BATCH_SIZE, MASK_SIZE, MASK_SIZE), dtype=tf.float32, trainable=True)
masks =   tf.clip_by_value(vals, 0., 1.) # tf.nn.tanh(vals)/2. + 0.5
if MASK_SIZE != 224:
    masks = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(masks, 3), (224, 224)))
    masks = tf.squeeze(utils.gaussian.gaussian_blur(tf.image.resize_bilinear(tf.expand_dims(masks, 3), (224, 224)), 15, 2))
showable_masks = tf.concat((255*tf.expand_dims(masks, 3), tf.zeros((BATCH_SIZE, 224, 224, 2))), 3)


preserved_images = utils.mask.apply_mask(images, masks, random_colors=True, noise=True, blurred_version_prob=1., blur_sigma=10, blur_kernel=41)
destroyed_images = utils.mask.apply_mask(images, 1.-masks, random_colors=True, noise=True, blurred_version_prob=1., blur_sigma=10, blur_kernel=41)

# feed the resulting images to the model
all_images = tf.concat((preserved_images, destroyed_images), 0)
with tf.variable_scope('TrainedClassifierModel'):
    all_scores = resnet.inference(all_images, False, False)
preserved_scores, destroyed_scores = tf.split(all_scores, 2)
init_op = utils.meta_restorer.restore_in_scope(resnet.CKPT, 'TrainedClassifierModel')


def satisfactory_reduction(x, sat):
    return tf.nn.relu(x-sat)+sat

# now the losses...
area_loss = utils.mask.area_loss(vals)
smoothness_loss = utils.mask.smoothness_loss(vals)

# not sure how these 2 should exactly be defined...
preservation_loss = 0*tf.reduce_mean(utils.loss_calc.abs_distance_loss(logits=tf.nn.softmax(preserved_scores), labels=labels, ref=1.))
destroyer_loss = tf.reduce_mean((utils.loss_calc.abs_distance_loss(logits=tf.nn.softmax(destroyed_scores), labels=labels, ref=0.)+0.0005)**1)
#preservation_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preserved_scores, labels=labels))
#destroyer_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=-destroyed_scores, labels=labels))

# compose the full loss, area and smoothness coefs are quite important

full_loss =  0.01*smoothness_loss + 0.05*area_loss + preservation_loss + destroyer_loss
opt_op = tf.train.AdamOptimizer(0.05).minimize(full_loss, var_list=[vals])
assert len(tf.trainable_variables())==1

# this is how you reset the images, just run reset_all with new_images and new_labels in feed_dict
new_images = tf.placeholder(tf.float32, (BATCH_SIZE, 224, 224, 3))
new_labels = tf.placeholder(tf.int32, (BATCH_SIZE,))
reset_all = tf.group(
    tf.assign(images, new_images),
    tf.assign(labels, new_labels),
    tf.assign(vals, 0.85*tf.ones((BATCH_SIZE, MASK_SIZE, MASK_SIZE), tf.float32))
)

blur_vals = tf.assign(vals, tf.squeeze(utils.gaussian.gaussian_blur(tf.expand_dims(vals, 3), kernel_size=3, sigma=0.1)))
reposition_vals = tf.assign(vals, tf.clip_by_value(vals, 0.01, 0.99))

GT_BOXES = imagenet.get_boxes_by_img_num()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(init_op)
    val_bm = imagenet.get_val_bm(BATCH_SIZE)
    too = 0
    all_ious = []
    for batch in xrange(len(imagenet.VAL_IMAGES)/BATCH_SIZE):
        too += 1
        if too < 9:
            continue
        paths = imagenet.VAL_IMAGES[batch*BATCH_SIZE:batch*BATCH_SIZE+BATCH_SIZE]

        sess.run(reset_all, {
            new_images: np.concatenate(map(imagenet.IMAGE_VAL_PIPELINE, paths), 0),
            new_labels: np.concatenate(map(imagenet.LABEL_VAL_PIPELINE, paths), 0)
        })
        sess.run(reposition_vals)
        i = 1
        while i<301:
            _, al, sl, pl, dl, pi, di, mi = sess.run([opt_op, area_loss, smoothness_loss, preservation_loss, destroyer_loss, preserved_images, destroyed_images, showable_masks])
            sess.run(reposition_vals)
            print i, 'A: %.2f   S: %.2f   P: %.3f    D: %.3f' % (al, sl, pl, dl)
            if not i%300:
                for example, path in enumerate(paths):
                    pi0 = imagenet.to_bgr_img(pi[example])
                    di0 = imagenet.to_bgr_img(di[example])
                    box = utils.bounding_box.box_from_mask(mi[example, :, :, 0])
                    gt_boxes, (width, height) = GT_BOXES[imagenet.img_num_from_path(path)]
                    scale = 224./min(width, height)
                    ious = []
                    for gt_box in gt_boxes:
                        x_offset, y_offset = (width - min(width, height))/2.,  (height - min(width, height))/2.
                        _gt_box_in_local = int((gt_box[0]-x_offset)*scale), int((gt_box[1]-y_offset)*scale), int((gt_box[2]-x_offset)*scale), int((gt_box[3]-y_offset)*scale)
                        gt_box_in_local = tuple(map(lambda x: min(max(x, 0), 223), _gt_box_in_local))
                        if gt_box_in_local!=_gt_box_in_local:
                            print 'Box overfull!', _gt_box_in_local
                        pi0 = utils.bounding_box.draw_box(pi0, gt_box_in_local, text='GT', color='red')
                        # calculate intersection over union. we have to use the original box, does not if overfull!
                        ious.append(utils.bounding_box.intersection_over_union(_gt_box_in_local, box))
                    all_ious.append(max(ious))
                    print 'IOU', max(ious)
                    pi0 = utils.bounding_box.draw_box(pi0, box, text='%.3f' % dl)

                    mi0 = mi[example].astype(np.uint8)
                    #mi0[mi0<0.6*255] = 0
                    cv2.imwrite('iter%d.jpg'%example, np.concatenate((pi0, di0, mi0), 0))
            i+=1
        print 'Cumulative accuracy from %d examples' % len(all_ious), np.mean(np.array(all_ious)>0.5)





