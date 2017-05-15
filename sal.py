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
import utils.bounding_box



CLASS_EMBEDDING_SIZE = 2048
FAKE_LABEL_CHANCE = 0.
RESIDUALS = 3
DECONV_KERNEL = 2
OPTIMIZE_CLS_EMBEDDING = False

PARENT_MODEL = resnet
BS = 1

TRAINABLE = False
WEIGHT_COLLECTIONS = ['chuje433314']


def get_tensor_as_constant(name):
    return tf.stop_gradient(tf.get_default_graph().get_tensor_by_name(name))


to_init = []

images = tf.placeholder(tf.float32, (BS, 224, 224, 3))
labels = tf.placeholder(tf.int32, (BS,))

is_class_present = tf.cast(tf.random_uniform((BS,), 0, 1, dtype=tf.float32) > FAKE_LABEL_CHANCE, tf.int32)
fake_labels= tf.random_uniform((BS,), 0, 1000, dtype=tf.int32)
class_selector = tf.stop_gradient(labels*is_class_present + (1-is_class_present)*fake_labels)

# 1 in a 1000 chance that fake label is actually real :) Therefore:
is_class_present = tf.cast(tf.equal(labels, class_selector), tf.int32)
is_class_present_f32 = tf.cast(is_class_present, tf.float32)

# Create masker model, U-Net like architecture with ResNet-50 extractor. Generated mask will be 112x112
with tf.variable_scope('masker'):
    # use resnet to extract features from the image
    # thats the first part of the U-Net
    with tf.variable_scope('extractor'):
        resnet.inference(images, False, False)  # we will get the tensors manually
    to_init.append(restore_in_scope(resnet.CKPT, 'masker/extractor'))

    class_embedding = tf.get_variable('clsemb', (1000, CLASS_EMBEDDING_SIZE),
                                      dtype=tf.float32,
                                      trainable=TRAINABLE and OPTIMIZE_CLS_EMBEDDING,
                                      collections=[tf.GraphKeys.GLOBAL_VARIABLES],
                                      initializer=tf_layers.variance_scaling_initializer())

    embedded_classes = tf.nn.embedding_lookup(class_embedding, class_selector)
    print 'Embedded classes', embedded_classes

    resnet_emb = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'masker/extractor/fc')[0]

    res7 = get_tensor_as_constant('masker/extractor/scale5/block3/Relu:0')
    res14 = get_tensor_as_constant('masker/extractor/scale4/block6/Relu:0')
    res28 = get_tensor_as_constant('masker/extractor/scale3/block4/Relu:0')
    res56 = get_tensor_as_constant('masker/extractor/scale2/block3/Relu:0')
    res112 = get_tensor_as_constant('masker/extractor/scale1/Relu:0')

    params = tf.get_variable('paramsss', (2,),
                                      dtype=tf.float32,
                                      trainable=TRAINABLE,
                                      collections=[tf.GraphKeys.GLOBAL_VARIABLES],
                                      initializer=tf.constant_initializer(np.array([1., 0])))

    # allow to select a specific class. use a gater to gate the top level mask
    _gater7 = tf.nn.sigmoid(params[0]*tf.reduce_sum(res7 * tf.reshape(embedded_classes, (BS, 1, 1, -1)), 3)+params[1])
    gater7 = tf.reshape(_gater7, (BS, 7, 7, 1))
    # if not OPTIMIZE_CLS_EMBEDDING:
    #     gater7 = tf.stop_gradient(gater7)

    check = _gater7
    print gater7
    print check
    disp_check = tf.reduce_mean(tf.abs(tf.reshape(is_class_present_f32, (BS, 1, 1, 1))-tf.reduce_max(gater7, (1,2), keep_dims=True)))

    # now construct the remaining, upsampler part of the U-Net
    #digester = SimpleCNNBlock(2, 768, 1, follow_with_bn=True)
    up1 = UpsamplerBlock(768, True, RESIDUALS, deconv_kernel_size=DECONV_KERNEL) # 14x14
    up2 = UpsamplerBlock(384, True, RESIDUALS, deconv_kernel_size=DECONV_KERNEL) # 28x28
    up3 = UpsamplerBlock(192, True, RESIDUALS, deconv_kernel_size=DECONV_KERNEL) # 56x56
    up4 = UpsamplerBlock(64, True, RESIDUALS, deconv_kernel_size=DECONV_KERNEL) # 112x112
    to_mask = SimpleCNNBlock(1, 2, 1, 1, False, tf.abs)  # converts to the mask

    # digester(res7, trainable=TRAINABLE, weights_collections=WEIGHT_COLLECTIONS)['final_output']
    out7 = res7 * gater7 # tf.reshape(is_class_present_f32, (BS, 1, 1, 1))
    out14 = up1(out7, trainable=TRAINABLE, weights_collections=WEIGHT_COLLECTIONS, passthrough=res14)['final_output']
    out28 = up2(out14, trainable=TRAINABLE, weights_collections=WEIGHT_COLLECTIONS, passthrough=res28)['final_output']
    out56 = up3(out28, trainable=TRAINABLE, weights_collections=WEIGHT_COLLECTIONS, passthrough=res56)['final_output']
    out112 = up4(out56, trainable=TRAINABLE, weights_collections=WEIGHT_COLLECTIONS, passthrough=res112)['final_output']
    final_out = out56

    # lets use _res7 to calculate whether selected class exists in the image at all, we can use that to produce an extra loss term
    raw_exists_logits = tf.reduce_sum(tf.reduce_mean(res7, (1, 2)) * embedded_classes, 1, keep_dims=True)
    exists_logits = tf.concat((tf.zeros((BS, 1)), raw_exists_logits), 1)  # we calculate logits for is_class_present==1 and fix another logit to 0
    exists_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=exists_logits, labels=is_class_present))


    _mask = to_mask(final_out, trainable=TRAINABLE, weights_collections=WEIGHT_COLLECTIONS)['final_output']
    _mask = tf.expand_dims(_mask[:,:,:,0]/(_mask[:,:,:,0] + _mask[:,:,:,1]), 3)
    _mask = tf.image.resize_bilinear(_mask, (224, 224))
    mask = tf.squeeze(_mask, axis=3)
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


preservation_loss = tf.reduce_mean(is_class_present_f32*tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preserved_scores, labels=class_selector))
destroyer_loss = tf.reduce_mean(is_class_present_f32*((utils.loss_calc.abs_distance_loss(logits=destroyed_probs, labels=class_selector, ref=0.)+0.0005)**0.33))

l2_loss = sum(map(tf.nn.l2_loss, tf.get_collection(WEIGHT_COLLECTIONS[0], 'masker')))


full_loss = 0.0005*l2_loss + 10*mask_area_loss + 0.005*mask_smoothness_loss + (preservation_loss + 4*destroyer_loss)/(1.-FAKE_LABEL_CHANCE) #+ exists_loss
train_op = tf.train.MomentumOptimizer(0.03, 0.9, use_nesterov=True).minimize(full_loss) if TRAINABLE else 11

pretrained_embedding = tf.placeholder(tf.float32, (1000, CLASS_EMBEDDING_SIZE))
set_embedding = tf.assign(class_embedding, pretrained_embedding)

DID_INIT=False
def get_bbs(sess, imgs_, labels_, return_center_crop_on_failure=True):
    global DID_INIT
    if not DID_INIT:
        #sess.run(tf.global_variables_initializer())
        tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'masker')).restore(sess, tf.train.get_checkpoint_state('temp_ckpts/masker1/').model_checkpoint_path)
        sess.run(tf.group(*to_init))
        sess.run(set_embedding, {pretrained_embedding: np.load('temp_ckpts/clsemb.npy')})
        print 'Finished inits...'
        DID_INIT = True
    masks = sess.run(mask, {images: imgs_, labels: labels_})
    for e in xrange(len(masks)):
        r = np.concatenate((imagenet.to_bgr_img(imgs_[e]), (np.zeros((224, 224, 3)) + np.expand_dims((255 * masks[e]), 2)).astype(np.uint8)), 0)
        cv2.imwrite('ss%d.jpg'%e, r)
    # exit()
    bbs = []
    for m in masks:
        # for each generated mask perform density based clustering and return...
        bbs.append(utils.bounding_box.box_from_mask(m, threshold=0.6, min_members=300, return_center_crop_on_failure=return_center_crop_on_failure))
    return bbs, masks


if __name__=='__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.group(*to_init))
        #emb2 = sess.run(resnet_emb).T
        sess.run(set_embedding, {pretrained_embedding: np.load('temp_ckpts/clsemb.npy')})

        bm_train = imagenet.get_train_bm(BS)
        bm_val = imagenet.get_val_bm(BS)

        nt = NiceTrainer(sess, bm_val, [images, labels], train_op, bm_val,
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
                             'e': exists_loss,
                             'disp': disp_check,
                             'p0': params[0],
                             'p1': params[1],
                         },
                         printable_vars=['a', 's', 'p', 'd', 'e', 'disp', 'p0', 'p1'],
                         computed_variables={
                             'rep': utils.rep.rep_op2(224, imagenet.to_bgr_img, imagenet.CLASS_ID_TO_NAME, validation_only=False),
                         },
                         saver=tf.train.Saver(tf.global_variables()),
                         save_every=500,
                         save_dir='temp_ckpts/globmask')
        nt.restore()

        tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'masker')).save(sess, 'temp_ckpts/masker1/model', global_step=1)
        nt.validate()
       # nt.save()

