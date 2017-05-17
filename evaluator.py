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
import random
import cv2
import iterative_crop
import math

# Some params
FAKE_PROVIDER = lambda sess, ims, labels: (0, 0, 200, 200), 9
BATCH_SIZE = 1
PROVIDER = iterative_crop.get_bb

VAL_IMGS = cPickle.load(open('ckpts/VAL_IMGS.cpi', 'rb'))[7500:8500]
random.shuffle(VAL_IMGS)

assert len(VAL_IMGS)%BATCH_SIZE == 0, len(VAL_IMGS)
print 'Loading boxes'
ALL_GT_BOXES = imagenet.get_boxes_by_img_num()
GT_BOXES = [ALL_GT_BOXES[imagenet.img_num_from_path(path)] for path in VAL_IMGS]
# load all images, will fit in the memory
print 'Loading images'
LOADED_IMGS = [utils.fixed_image.ImageResizer(imagenet.load_normalized_without_resize(path), 224) for path in VAL_IMGS]
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
        if crop_box[0]==crop_box[2] or crop_box[1]==crop_box[3]:
            print 'Invalid size!!!'
            raise
        crop = im[int(crop_box[1]):int(crop_box[3]),int(crop_box[0]):int(crop_box[2]),:]
        # sometimes labels are wrong...
        if not crop.shape[0] or not crop.shape[1]:
            print 'Incorrect label...'
            return [0.001]*1000
        cropped = utils.fixed_image.ImageResizer(crop, 224).get_resulting_img()
    return sess.run(ev_probs, {ev_images: cropped})[0]


sess = tf.Session()
sess.run(to_init)

hits = []
gt_scores= []
our_scores = []
gt_a = []
our_a = []


single_hits = []
single_gt_scores = []
single_our_scores = []

def calc_score(a, p):
    return math.log(max(a, 0.05)) - math.log(p)


for i in xrange(0, len(VAL_IMGS), BATCH_SIZE):
    imcs = LOADED_IMGS[i:i+BATCH_SIZE]
    labels = LABELS[i:i+BATCH_SIZE]
    gt_boxes_sizes = GT_BOXES[i:i+BATCH_SIZE]
    _local_boxes, masks = PROVIDER(sess, np.array(map(lambda x: x.get_resulting_img(), imcs)), np.array(labels))
    k = 0
    for imc, label, gt_boxes_size, _local_box in zip(imcs, labels, gt_boxes_sizes, _local_boxes):
        gt_boxes, size = gt_boxes_size
        total_area = float(size[0]*size[1])
        im = imc.get_resulting_img()
        box = imc.from_local_coords(_local_box)
        # local box may be invalid (outside actual image area)
        local_box = imc.to_local_coords(box)  # this one is valid
        ious = []
        gt_scores_ = []
        gt_a_ = []
        showable = imagenet.to_bgr_img(im.copy())
        for gt_box in gt_boxes:
            p = get_crop_probs(sess, imc.original_image, gt_box)[label]
            a = utils.bounding_box.area(gt_box)/total_area
            showable = utils.bounding_box.draw_box(showable, imc.to_local_coords(gt_box), color='red', text='P: %.2f'% p)
            ious.append(utils.bounding_box.intersection_over_union(gt_box, box))
            gt_scores_.append(calc_score(a, p))
            gt_a_.append(math.log(max(a, 0.05)))
        gt_a.append(np.mean(gt_a_))
        iou = max(ious)
        p = get_crop_probs(sess, imc.original_image, box)[label]
        a = utils.bounding_box.area(box) / total_area
        our_scores.append(calc_score(a, p))
        hits.append(float(iou >= 0.5))
        gt_scores.append(np.mean(gt_scores_))
        our_a.append(math.log(max(a, 0.05)))
        if len(gt_boxes)==1:
            single_our_scores.append(calc_score(a, p))
            single_hits.append(float(iou >= 0.5))
            single_gt_scores.append(np.mean(gt_scores_))
        showable = utils.bounding_box.draw_box(showable, local_box, color='green', text='IOU: %.2f P: %.2f' % (iou, p))
        cv2.imwrite('ev.jpg', showable)
        print 'ok Acc:', np.mean(hits), 'GT_M:', np.mean(gt_scores),np.mean(gt_a), 'OUR_M', np.mean(our_scores), np.mean(our_a)
        k+=1
        #raw_input()

print 'Done'
print 'All Acc:', np.mean(hits), 'GT_M:', np.mean(gt_scores), 'OUR_M', np.mean(our_scores)
print 'Single Acc:', np.mean(single_hits), 'GT_M:', np.mean(single_gt_scores), 'OUR_M', np.mean(single_our_scores)
print len(single_our_scores)
print len(hits)

