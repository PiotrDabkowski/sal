import resnet
import tensorflow as tf
from meta_restorer import restore_in_scope
import alexnet
from tfutils import *
import imagenet

# 80.5 percent top 5 with 10 crop validation and 79 % with single centre crop. AlexNet is pretty bad. Loss 2
sess = tf.Session()


images = tf.placeholder(tf.float32, (200, 224, 224, 3), name='i1')
images2 = tf.placeholder(tf.float32, (200, 224, 224, 3), name='i2')


with tf.variable_scope('r1'):
    p1 = alexnet.inference(images, False, False)
    probs = tf.nn.softmax(p1)

with tf.variable_scope('r1', reuse=True):
    p2 = alexnet.inference(images2, False, True)
    probs2 = tf.nn.softmax(p2)


restore_op = restore_in_scope('alexnet/model.ckpt', 'r1')
sess.run(restore_op)


assert not tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

for e in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    print e.name


BS = 200
ins = tf.placeholder(tf.int32, (BS,))
val_bm = imagenet.get_val_bm(BS)

for imgs,pro, p in zip((images, images2), (probs, probs2), (p1, p2)):
    nt = NiceTrainer(sess,
                     None,
                     [imgs, ins],
                     11,
                     val_bm,
                     extra_variables={'probs': pro,
                                      'loss': tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=p, labels=ins))},
                     computed_variables={
                                        'acc-5': accuracy_calc_op(n=5, avg_preds_over=imagenet.NUM_VALIDATION_IMGS_PER_EXAMPLE),
                                        'acc-1': accuracy_calc_op(n=1, avg_preds_over=imagenet.NUM_VALIDATION_IMGS_PER_EXAMPLE),
                     },
                     printable_vars=['acc-1', 'acc-5', 'loss'],
                     )

    nt.validate()







