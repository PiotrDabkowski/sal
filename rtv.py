import sal
from datasets import imagenet
import utils.fixed_image
import utils.bounding_box
import numpy as np
import cv2
import tensorflow as tf


def get_bbs(sess, im, label):
    conv = utils.fixed_image.FixedAspectRatioNoCropping(imagenet.from_bgr_normalize_only(im), 224)
    xy_, mask_ = sal.get_bbs(sess, np.expand_dims(conv.get_resulting_img(), 0), [label], return_center_crop_on_failure=False)
    xy = xy_[0]
    if xy is None:
        return None
    return conv.from_local_coords(xy)

sess = tf.Session()
im = cv2.imread('tests/catdog.jpg')

while 1:
    sel = int(raw_input('? => '))
    box = get_bbs(sess, im, sel)
    print 'Looking for', imagenet.CLASS_ID_TO_NAME[sel]
    if box is not None:
        print 'Found'
        show = im.copy()/255.
        cv2.imshow('aa', utils.bounding_box.draw_box(show, box))
        cv2.waitKey(30000)
        cv2.destroyWindow('aa')
    else:
        print 'Not found'

