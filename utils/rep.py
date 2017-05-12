import numpy as np
import time
import cv2


LAST_REP = time.time()

def rep_op(resolution, to_bgr_img, show_every=30, validation_only=True):
    ''' Required extra vars:
    mask
    resulting_img
    probs
    probs_cif1
    probs_cif2
    '''
    def rep(extra_vars, batch):
        global LAST_REP
        if time.time() - LAST_REP < show_every:
            return
        if extra_vars['is_training'] and validation_only:
            return
        LAST_REP = time.time()
        mask = extra_vars['mask']
        real_imgs = batch[0]
        res_imgs = extra_vars['resulting_img']
        probs = extra_vars['probs']

        labels_by_cif1 = np.argmax(extra_vars['probs_cif1'], 1)
        labels = batch[1]

        to_show = int(1000/resolution)
        show = []
        for e in xrange(to_show):
            m = np.expand_dims((mask[e]*255.0).astype(np.uint8), 2)
            real_img = to_bgr_img(real_imgs[e])
            res_img = to_bgr_img(res_imgs[e])
            understanding = np.ones((resolution, resolution, 3), np.uint8)*color_scale(probs[e][labels_by_cif1[e]])

            secondary_understanding = np.ones((resolution, resolution, 3), np.uint8) * color_scale(extra_vars['probs_cif2'][e][labels_by_cif1[e]])
            cif_1_correctness = np.ones((resolution, resolution, 3), np.uint8) * color_scale(extra_vars['probs_cif1'][e][labels[e]])

            img_row = (real_img, np.concatenate((m, 0*m, 0*m), 2), res_img, understanding, secondary_understanding, cif_1_correctness)

            show.append(np.concatenate(img_row, 0))
            # separator :)
            show.append(np.zeros((len(img_row)*resolution, 2, 3)))
        fin = np.concatenate(tuple(show), 1)
        cv2.imwrite('res.jpg', fin)
    return rep


def rep_op2(resolution, to_bgr_img, class_id_to_name, show_every=30, validation_only=True):
    ''' Required extra vars:
    mask
    resulting_img
    class_selector
    '''
    def rep(extra_vars, batch):
        global LAST_REP
        if time.time() - LAST_REP < show_every:
            return
        if extra_vars['is_training'] and validation_only:
            return
        LAST_REP = time.time()
        mask = extra_vars['mask']
        real_imgs = batch[0]
        res_imgs = extra_vars['resulting_img']
        class_selector = extra_vars['class_selector']
        is_class_present = extra_vars['is_class_present']


        to_show = int(1000/resolution)
        show = []
        for e in xrange(to_show):
            m = np.expand_dims((mask[e]*255.0).astype(np.uint8), 2)
            real_img = to_bgr_img(real_imgs[e])
            res_img = to_bgr_img(res_imgs[e])

            info = 255*np.ones((resolution, resolution, 3), dtype=np.uint8)
            cv2.putText(info, 'T: %s' % class_id_to_name[class_selector[e]], (5, 30), fontScale=0.6, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=color_scale(is_class_present[e]))


            img_column = (real_img, np.concatenate((m, 0*m, 0*m), 2), res_img, info)

            show.append(np.concatenate(img_column, 0))
            # separator :)
            show.append(np.zeros((len(img_column)*resolution, 2, 3)))
        fin = np.concatenate(tuple(show), 1)
        cv2.imwrite('res.jpg', fin)
    return rep

def color_scale(fraction):
    R = 255 * (1.-fraction)
    G = 255 * fraction
    B = 0
    return int(B), int(G), int(R)