import vgg16
import tensorflow as tf
from meta_restorer import restore_in_scope
import imagenet
from tfutils import *

sess = tf.Session()

# 1 crop - 89.5%  , 10 crop - 90.5%
images = tf.placeholder(tf.float32, (None, 224, 224, 3), name='i1')
images2 = tf.placeholder(tf.float32, (None, 224, 224, 3), name='i2')


with tf.variable_scope('r1'):
    p1 = vgg16.inference(images, False, False)
    probs = tf.nn.softmax(p1)

with tf.variable_scope('r1'):
    p2 = vgg16.inference(images2, False, True)
    probs2 = tf.nn.softmax(p2)



sess.run(tf.global_variables_initializer())
restore_op = restore_in_scope(vgg16.CKPT, 'r1')
sess.run(restore_op)


assert not tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

for e in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    print e.name



val_bm = imagenet.get_val_bm(150)

for imgs,pro in zip((images, images2), (probs, probs2)):
    nt = NiceTrainer(sess,
                     None,
                     [imgs],
                     11,
                     val_bm,
                     extra_variables={'probs': pro},
                     computed_variables={
                                        'acc-5': accuracy_calc_op(n=5, avg_preds_over=imagenet.NUM_VALIDATION_IMGS_PER_EXAMPLE),
                                        'acc-1': accuracy_calc_op(n=1, avg_preds_over=imagenet.NUM_VALIDATION_IMGS_PER_EXAMPLE)
                     },
                     printable_vars=['acc-1', 'acc-5'],
                     )

    nt.validate()







