# Evaluator provides a batch of images and labels to the box generator which for each (image, label) pair provides coords of the box
# Evaluator later calculates GT IOU and other metrics
from datasets import imagenet
import tensorflow as tf
import numpy as np
import cPickle
import utils.fixed_image
import utils.bounding_box
import utils.meta_restorer
from blocks import resnet
import cv2
import sal

# Some params
FAKE_PROVIDER = lambda sess, ims, labels: (0, 0, 200, 200)
BATCH_SIZE = 25
PROVIDER = sal.get_bbs

VAL_IMGS = cPickle.load(open('ckpts/VAL_IMGS.cpi', 'rb'))[:2500]
assert len(VAL_IMGS)%BATCH_SIZE == 0, len(VAL_IMGS)
print 'Loading boxes'
ALL_GT_BOXES = imagenet.get_boxes_by_img_num()
GT_BOXES = [ALL_GT_BOXES[imagenet.img_num_from_path(path)] for path in VAL_IMGS]
# load all images, will fit in the memory
print 'Loading images'
LOADED_IMGS = [utils.fixed_image.FixedAspectRatioNoCropping(imagenet.load_normalized_without_resize(path), 224) for path in VAL_IMGS]
LABELS = map(lambda x: imagenet.LABEL_VAL_PIPELINE(x)[0], VAL_IMGS)


# resnet evaluation model
ev_images = tf.placeholder(tf.float32, (224, 224, 3), 'evimgs')
with tf.variable_scope('resneeval'):
    ev_probs = tf.nn.softmax(resnet.inference(tf.expand_dims(ev_images, 0), False, False))
to_init = utils.meta_restorer.restore_in_scope(resnet.CKPT, 'resneeval')

def get_crop_probs(sess, im, crop_box=None):
    if crop_box is None:
        cropped = im
    else:
        if crop_box[0]==crop_box[2]:
            print 'Invalid size!!!'
            return 1000*[0.]
        crop = im[int(crop_box[1]):int(crop_box[3]),int(crop_box[0]):int(crop_box[2]),:]
        cropped = utils.fixed_image.FixedAspectRatioNoCropping(crop, 224).get_resulting_img()
    return sess.run(ev_probs, {ev_images: cropped})[0]


sess = tf.Session()
sess.run(to_init)

hits = []

for i in xrange(0, len(VAL_IMGS), BATCH_SIZE):
    imcs = LOADED_IMGS[i:i+BATCH_SIZE]
    labels = LABELS[i:i+BATCH_SIZE]
    gt_boxes_sizes = GT_BOXES[i:i+BATCH_SIZE]
    _local_boxes = PROVIDER(sess, np.array(map(lambda x: x.get_resulting_img(), imcs)), np.array(labels))
    k = 0
    for imc, label, gt_boxes_size, _local_box in zip(imcs, labels, gt_boxes_sizes, _local_boxes):
        gt_boxes, size = gt_boxes_size
        im = imc.get_resulting_img()
        box = imc.from_local_coords(_local_box)
        # local box may be invalid (outside actual image area)
        local_box = imc.to_local_coords(box)  # this one is valid
        ious = []
        showable = imagenet.to_bgr_img(im.copy())
        for gt_box in gt_boxes:
            p = get_crop_probs(sess, imc.original_image, gt_box)[label]
            showable = utils.bounding_box.draw_box(showable, imc.to_local_coords(gt_box), color='red', text='P: %.2f'% p)
            ious.append(utils.bounding_box.intersection_over_union(gt_box, box))
        iou = max(ious)
        hits.append(float(iou>=0.5))
        area_fraction = utils.bounding_box.intersection_over_union((0.,0.)+size, box)
        prob = get_crop_probs(sess, imc.original_image, box)
        showable = utils.bounding_box.draw_box(showable, local_box, color='green', text='IOU: %.2f P: %.2f' % (iou, prob[label]))
        cv2.imwrite('ev.jpg', showable)
        print 'ok', k, np.mean(hits)
        k+=1
        #raw_input()

