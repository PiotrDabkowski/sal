import tensorflow as tf
import numpy as np

X = tf.random_uniform((1000,), 0, 7, dtype=tf.int32)

sess = tf.Session()
print sess.run(tf.reduce_mean(X))