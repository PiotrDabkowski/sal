import tensorflow as tf
import numpy as np
import cv2
from blocks import resnet, alexnet
from blocks.std import *
from utils.meta_restorer import restore_in_scope
import utils.mask
import utils.loss_calc
import utils.rep
from tensorflow.contrib import layers as tf_layers
from datasets import imagenet
from tfutils import *




CLASS_EMBEDDING_SIZE = 80
FAKE_LABEL_CHANCE = 0.4
RESIDUALS = 2

PARENT_MODEL = alexnet
BS = 24

TRAINABLE = True
WEIGHT_COLLECTIONS = ['chuje433314']


def get_tensor_as_constant(name):
    return tf.stop_gradient(tf.get_default_graph().get_tensor_by_name(name))


to_init = []

images = tf.placeholder(tf.float32, (BS, 224, 224, 3))
labels = tf.placeholder(tf.int32, (BS,))

is_class_present = tf.cast(tf.random_uniform((BS,), 0, 1, dtype=tf.float32) > FAKE_LABEL_CHANCE, tf.int32)
fake_labels= tf.random_uniform((BS,), 0, 1000, dtype=tf.int32)
class_selector = labels*is_class_present + (1-is_class_present)*fake_labels

# 1 in a 1000 chance that fake label is actually real :) Therefore:
is_class_present = tf.cast(tf.equal(labels, class_selector), tf.int32)

# Create masker model, U-Net like architecture with ResNet-50 extractor. Generated mask will be 112x112
with tf.variable_scope('masker'):
    # use resnet to extract features from the image
    # thats the first part of the U-Net
    with tf.variable_scope('extractor'):
        resnet.inference(images, False, False)  # we will get the tensors manually
    to_init.append(restore_in_scope(resnet.CKPT, 'masker/extractor'))

    class_embedding = tf.get_variable('clsemb', (1000, CLASS_EMBEDDING_SIZE),
                                      dtype=tf.float32,
                                      trainable=True,
                                      collections=WEIGHT_COLLECTIONS+[tf.GraphKeys.GLOBAL_VARIABLES],
                                      initializer=tf_layers.variance_scaling_initializer())
    embedded_classes = tf.nn.embedding_lookup(class_embedding, class_selector)
    replicated_embedded_classes = tf.zeros((BS, 7, 7, CLASS_EMBEDDING_SIZE)) + tf.reshape(embedded_classes, (BS, 1, 1, CLASS_EMBEDDING_SIZE))

    _res7 = get_tensor_as_constant('masker/extractor/scale5/block3/Relu:0')
    res7 = tf.concat((replicated_embedded_classes, _res7), 3)
    res14 = get_tensor_as_constant('masker/extractor/scale4/block6/Relu:0')
    res28 = get_tensor_as_constant('masker/extractor/scale3/block4/Relu:0')
    res56 = get_tensor_as_constant('masker/extractor/scale2/block3/Relu:0')
    res112 = get_tensor_as_constant('masker/extractor/scale1/Relu:0')

    # now construct the remaining, upsampler part of the U-Net
    up1 = UpsamplerBlock(512, True, RESIDUALS) # 14x14
    up2 = UpsamplerBlock(256, True, RESIDUALS) # 28x28
    up3 = UpsamplerBlock(128, True, RESIDUALS) # 56x56
    up4 = UpsamplerBlock(64, True, RESIDUALS) # 112x112
    to_mask = SimpleCNNBlock(1, 2, 1, 1, False, tf.abs)  # converts to the mask

    out14 = up1(res7, trainable=TRAINABLE, weights_collections=WEIGHT_COLLECTIONS, passthrough=res14)['final_output']
    out28 = up2(out14, trainable=TRAINABLE, weights_collections=WEIGHT_COLLECTIONS, passthrough=res28)['final_output']
    out56 = up3(out28, trainable=TRAINABLE, weights_collections=WEIGHT_COLLECTIONS, passthrough=res56)['final_output']
    out112 = up4(out56, trainable=TRAINABLE, weights_collections=WEIGHT_COLLECTIONS, passthrough=res112)['final_output']
    final_out = out56

    _mask = to_mask(final_out, trainable=TRAINABLE, weights_collections=WEIGHT_COLLECTIONS)['final_output']
    _mask = tf.expand_dims(_mask[:,:,:,0]/(_mask[:,:,:,0] + _mask[:,:,:,1]), 3)
    _mask = tf.image.resize_bilinear(_mask, (224, 224))
    mask = tf.squeeze(_mask)
    print mask.get_shape()



# here quickly computer mask losses
mask_area_loss = utils.mask.area_loss(mask, satisfactory_reduction=0.)
mask_smoothness_loss = utils.mask.smoothness_loss(mask)

preserved_imgs = utils.mask.apply_mask(images, mask, True, True, 0.4, blur_kernel=41, blur_sigma=10)
destroyed_imgs = utils.mask.apply_mask(images, 1.-mask, True, True, 0.4, blur_kernel=41, blur_sigma=10)


all_imgs = tf.concat((preserved_imgs, destroyed_imgs), 0)
with tf.variable_scope('parent'):
    all_scores = PARENT_MODEL.inference(all_imgs, False, False)  # we will get the tensors manually
to_init.append(restore_in_scope(PARENT_MODEL.CKPT, 'parent'))
preserved_scores, destroyed_scores = tf.split(all_scores, 2, 0)
preserved_probs = tf.nn.softmax(preserved_scores)
destroyed_probs = tf.nn.softmax(destroyed_scores)

is_class_present_f32 = tf.cast(is_class_present, tf.float32)

preservation_loss = tf.reduce_mean(is_class_present_f32*tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preserved_scores, labels=class_selector))
destroyer_loss = tf.reduce_mean(is_class_present_f32*((utils.loss_calc.abs_distance_loss(logits=destroyed_probs, labels=labels, ref=0.)+0.0005)**0.33))

l2_loss = sum(map(tf.nn.l2_loss, tf.get_collection(WEIGHT_COLLECTIONS[0], 'masker')))


full_loss = 0.0005*l2_loss + 12*mask_area_loss + 0.0001*mask_smoothness_loss + (preservation_loss + 4*destroyer_loss)/(1.-FAKE_LABEL_CHANCE)

train_op = tf.train.MomentumOptimizer(0.03, 0.9, use_nesterov=True).minimize(full_loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.group(*to_init))

    bm_train = imagenet.get_train_bm(BS)
    bm_val = imagenet.get_val_bm(BS)

    nt = NiceTrainer(sess, bm_train, [images, labels], train_op, bm_val,
                     extra_variables={
                         'a': mask_area_loss,
                         's': mask_smoothness_loss,
                         'p': preservation_loss,
                         'd': destroyer_loss,
                         'resulting_img': preserved_imgs,
                         'mask': mask,
                         'probs': preserved_probs,
                         'class_selector': class_selector,
                         'is_class_present': is_class_present,
                         'pr': tf.reduce_mean(100*is_class_present),
                     },
                     printable_vars=['a', 's', 'p', 'd', 'pr'],
                     computed_variables={
                         'rep': utils.rep.rep_op2(224, imagenet.to_bgr_img, imagenet.CLASS_ID_TO_NAME, validation_only=False)
                     })

    nt.train()


