import tensorflow as tf
import numpy as np
import cv2
from blocks import inception2
from utils.meta_restorer import restore_in_scope

input_tensor = tf.placeholder(tf.float32, (1, 299, 299, 3))
with tf.variable_scope('aaa'):
    logits = inception2.inference(input_tensor, False, False)
init_op = restore_in_scope(inception2.CKPT, 'aaa')

sess = tf.Session()
sess.run(init_op)

img = cv2.imread('zebra.jpg')
img = cv2.resize(img, (299, 299), cv2.INTER_LINEAR)[:,:,[2,1,0]]
img = np.reshape(img, (-1, 299, 299, 3)) / 255.*2 - 1.
print logits
r, s = sess.run([logits, tf.nn.softmax(logits)], {input_tensor: img})
print r.shape
print np.max(s), np.argmax(r)
