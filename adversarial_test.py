import tensorflow as tf
import numpy as np
from blocks import resnet
import utils.meta_restorer
import utils.fixed_image
from datasets import imagenet
import sal
import cv2

dog = imagenet.IMAGE_VAL_PIPELINE('dog.jpg')[0]

img_var = tf.get_variable('chuj', (224, 224, 3))
expanded_img_var = tf.expand_dims(img_var, 0)
with tf.variable_scope('resneeval'):
    ev_probs = tf.nn.softmax(resnet.inference(expanded_img_var, False, False))[0]
to_init = utils.meta_restorer.restore_in_scope(resnet.CKPT, 'resneeval')

new_image = tf.placeholder(tf.float32, (224, 224, 3))
set_img_var = tf.assign(img_var, new_image)
reset_img_var = tf.assign(img_var, tf.random_uniform((224, 224, 3), -0.1, 0.1))

adv_target = tf.placeholder(tf.int32, ())
adv_loss = ev_probs[adv_target]
adv_opt = tf.train.AdamOptimizer(0.05).minimize(adv_loss)

def get_crop_probs(sess, im, crop_box=None):
    if crop_box is None:
        cropped = im
    else:
        if crop_box[0]==crop_box[2] or crop_box[1]==crop_box[3]:
            print 'Invalid size!!!'
            raise
        crop = im[int(crop_box[1]):int(crop_box[3]),int(crop_box[0]):int(crop_box[2]),:]
        # sometimes labels are wrong...
        if not crop.shape[0] or not crop.shape[1]:
            print 'Incorrect label...'
            return [0.001]*1000
        cropped = utils.fixed_image.ImageResizer(crop, 224).get_resulting_img()
    sess.run(set_img_var, {new_image: cropped})
    return sess.run(ev_probs)

def prepare_adv(sess, target):
    for n in xrange(10):
        _, p = sess.run([adv_opt, ev_probs[adv_target]], {adv_target: target})
        print p



sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(to_init)

print 'Initial prob', get_crop_probs(sess, dog)[203]
sess.run(set_img_var, {new_image: dog})

cv2.imshow('Initial image', imagenet.to_bgr_img(sess.run(img_var)))
cv2.waitKey(1000000)

prepare_adv(sess, 203)
best = np.argmax(sess.run(ev_probs))
print 'Highest prob class', best, imagenet.CLASS_ID_TO_NAME[best], sess.run(ev_probs)[best]
cv2.imshow('Adversarial image', imagenet.to_bgr_img(sess.run(img_var)))
cv2.waitKey(1000000)

b, mask = sal.get_bbs(sess, sess.run(expanded_img_var), (440,))
print b
cv2.imshow('Mask for random (beer bottle) label', mask[0])
cv2.waitKey(1000000)

b, mask = sal.get_bbs(sess, sess.run(expanded_img_var), (203,))
print b
cv2.imshow('Mask for correct label', mask[0])
cv2.waitKey(1000000)

b, mask = sal.get_bbs(sess, sess.run(expanded_img_var), (best,))
print b
cv2.imshow('Mask for adversarial label', mask[0])
cv2.waitKey(1000000)

advim = sess.run(img_var)
fixed_advim = advim*(1-np.expand_dims(mask[0], 2))
print 'Adversarial before masking prob', get_crop_probs(sess, advim)[best]
print 'Adversarial after masking prob', get_crop_probs(sess, fixed_advim)[best]
print 'Adversarial after masking correct prob', get_crop_probs(sess, advim)[203]
cv2.imshow('Adversarial after masking', imagenet.to_bgr_img(fixed_advim))
cv2.waitKey(1000000)

print 'Real before masking', get_crop_probs(sess, dog)[203]
b, mask = sal.get_bbs(sess, sess.run(expanded_img_var), (203,))
masked_real = dog*(1-np.expand_dims(mask[0], 2))
print 'Real after masking', get_crop_probs(sess, masked_real)[203]

cv2.imshow('Real after masking', imagenet.to_bgr_img(masked_real))
cv2.waitKey(1000000)
