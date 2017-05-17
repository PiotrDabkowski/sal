import sal
from datasets import imagenet
import utils.fixed_image
import utils.bounding_box
import numpy as np
import cv2
import tensorflow as tf


def get_bbs(sess, im, label):
    conv = utils.fixed_image.ImageResizer(imagenet.from_bgr_normalize_only(im), (224, 224))
    xy_, mask_ = sal.get_bbs(sess, np.expand_dims(conv.get_resulting_img(), 0), [label], return_center_crop_on_failure=False)
    xy = xy_[0]
    if xy is None:
        return None
    return conv.from_local_coords(xy), mask_

sess = tf.Session()
im = cv2.imread('tests/catdog.jpg')
showable = utils.fixed_image.ImageResizer(imagenet.from_bgr_normalize_only(im), (224, 224)).get_resulting_img()

while 1:
    sel = int(raw_input('? => '))
    box, mask = get_bbs(sess, im, sel)
    mask = np.reshape(mask, (224, 224, 1))
    print 'Looking for', imagenet.CLASS_ID_TO_NAME[sel]
    if box is not None:
        print 'Found'
        cv2.imwrite('mask.png', np.concatenate((255*mask, np.zeros((224,224,2), dtype=np.float32)), 2).astype(np.uint8))
        cv2.imwrite('original.png', imagenet.to_bgr_img(showable))
        cv2.imwrite('preserved.png', imagenet.to_bgr_img(mask*showable))
        cv2.imwrite('destroyed.png', imagenet.to_bgr_img((1-mask) * showable))
        show = im.copy()/255.
        cv2.imshow('aa', utils.bounding_box.draw_box(show, box))
        cv2.waitKey(30000)
        cv2.destroyWindow('aa')
    else:
        print 'Not found'

