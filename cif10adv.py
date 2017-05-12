# Cifar10 experiments with adversarial setup
# Validation accuracy goes up to 91%, typical for this architecture.
# Preserved area is about 55% with smoothness of 0.02
# Classifier network is able to reconstruct its original prediction from masked image in about 90% of cases, 0.38 loss (quite good)
# Results are quite good and can be seen in examples folder.

from blocks.std import *
from datasets import cifar10
from tfutils import NiceTrainer, accuracy_calc_op
import tensorflow as tf
import utils.mask
import utils.rep



BATCH_SIZE = 64

# ------------------------- MODEL SECTION ------------------------------
# Classifier
c = LinearContainer()
c <= SimpleCNNBlock(3, 32, 3, follow_with_bn=True)
c <= SimpleCNNBlock(2, 48, 3, follow_with_bn=True)
c <= TransitionBlock()
c <= SimpleCNNBlock(5, 80, 3, follow_with_bn=True)
c <= TransitionBlock()
c <= SimpleCNNBlock(5, 128, 3, follow_with_bn=True)
c <= CustomLinearBlock(lambda x: tf.reduce_mean(x, (1, 2)))
c <= SimpleLinearBlock(10, activation_fn=None)

# Mask generator + classifier
m = UNetContainer()
m <= SimpleCNNBlock(3, 32, 3, follow_with_bn=True)
m <= SimpleCNNBlock(2, 48, 3, follow_with_bn=True)
m <= TransitionBlock()
m <= SimpleCNNBlock(5, 90, 3, follow_with_bn=True)
m <= TransitionBlock()
m <= SimpleCNNBlock(5, 140, 3, follow_with_bn=True)
m['classification_branch'] = LinearContainer()
# continue with mask generation as a default branch
m <= UpsamplerBlock(64, True, 2)
m <= UpsamplerBlock(32, True, 2)
m <= SimpleCNNBlock(1, 1, 1, 1, False, tf.nn.tanh)
# finish classification branch
m['classification_branch'] <= CustomLinearBlock(lambda x: tf.reduce_mean(x, (1, 2)))
m['classification_branch'] <= SimpleLinearBlock(10, activation_fn=None)


# ------------------------- COMPOSITION SECTION (the longest part, have to generalise it) ------------------------------

# data input
images = tf.placeholder(tf.float32, (BATCH_SIZE, 32, 32, 3))
labels = tf.placeholder(tf.int32, (BATCH_SIZE,))


# classification
with tf.variable_scope('cif10model'):
    model = c(images, True, ['x134a'])
initial_classification_scores = model.final_output


initial_classification_labels = tf.argmax(initial_classification_scores, 1)
initial_classification_probs = tf.nn.softmax(initial_classification_scores)
classification_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=initial_classification_scores, labels=labels))
classifier_l2_loss = model.get_own_l2_loss()



# now the masking model
with tf.variable_scope('cif10masker'):
    masker = m(images, True, ['a42tt'])
masker_classification_scores = masker['classification_branch'].final_output

masker_classification_probs = tf.nn.softmax(masker_classification_scores)
masker_classification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=masker_classification_scores, labels=initial_classification_probs))
masker_l2_loss = masker.get_own_l2_loss()

generated_mask = tf.squeeze(masker.final_output/2. + 0.5)

masker_area_loss = utils.mask.area_loss(generated_mask, satisfactory_reduction=0.1)
masker_smoothness_loss = utils.mask.smoothness_loss(generated_mask)

preserved_imgs = utils.mask.apply_mask(images, generated_mask, noise=True, random_colors=True)
destroyed_imgs = utils.mask.apply_mask(images, 1.-generated_mask, noise=True, random_colors=True)

secondary_inp = tf.concat((preserved_imgs, destroyed_imgs), 0)
with tf.variable_scope('cif10model', reuse=True):
    model2 = c(secondary_inp, True, None)
preserved_scores, destroyed_scores = tf.split(model2.final_output, 2)
print 'Preserved', preserved_scores, destroyed_scores

preservation_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preserved_scores, labels=initial_classification_labels))
destroyer_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=-destroyed_scores, labels=initial_classification_labels))

masker_full_loss = 0.0005*masker_l2_loss + 5*masker_area_loss + 33*masker_smoothness_loss + preservation_loss + destroyer_loss + masker_classification_loss
masker_train_op = tf.train.MomentumOptimizer(0.003, 0.9, use_nesterov=True).minimize(masker_full_loss, var_list=masker.get_own_variables())



secondary_classification_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=destroyed_scores, labels=initial_classification_labels))
classifier_full_loss = classification_loss + 0.0005*classifier_l2_loss + 0.5*secondary_classification_loss +0.1*preservation_loss
classifier_train_op = tf.train.MomentumOptimizer(0.003, 0.9, use_nesterov=True).minimize(classifier_full_loss, var_list=model.get_own_variables())


train_op = tf.group(masker_train_op, classifier_train_op)

# ------------------------- TRAIN SECTION ------------------------------

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    nt = NiceTrainer(sess,
                     bm_train=cifar10.get_train_bm(BATCH_SIZE),
                     feed_keys=[images, labels],
                     train_op=train_op,
                     bm_val=cifar10.get_val_bm(BATCH_SIZE),
                     extra_variables={'loss': classification_loss,
                                      'probs': initial_classification_probs,
                                      'rec': masker_classification_loss,
                                      'sm': masker_smoothness_loss,
                                      'ar': masker_area_loss,
                                      'pres': preservation_loss,
                                      'dest': destroyer_loss,
                                      'mask': generated_mask,
                                      'resulting_img': preserved_imgs,
                                      'probs_cif1': masker_classification_probs,
                                      'probs_cif2': tf.nn.softmax(preserved_scores)},
                     printable_vars=['loss', 'acc', 'rec', 'sm', 'ar', 'pres', 'dest'],
                     computed_variables={'acc': accuracy_calc_op(),
                                         'repr': utils.rep.rep_op(32, cifar10.to_bgr_img, validation_only=True)},
                     saver=tf.train.Saver(tf.global_variables()),
                     save_every=100000,
                     save_dir='temp_ckpts/cif10adv'
                     )
    nt.restore()
    while True:
        nt.train()
        nt.validate()
        nt.save()
