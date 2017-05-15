import cv2
from Queue import Queue
import time
import numpy as np
import threading
import sal
from datasets import imagenet
import utils.fixed_image
import utils.bounding_box
import cv2
import tensorflow as tf


def get_bbs(sess, im, label):
    conv = utils.fixed_image.FixedAspectRatioNoCropping(imagenet.from_bgr_normalize_only(im), 224)
    xy_, mask_ = sal.get_bbs(sess, np.expand_dims(conv.get_resulting_img(), 0), [label], return_center_crop_on_failure=False)
    xy = xy_[0]
    if xy is None:
        return None
    return conv.from_local_coords(xy)

POLL_DELAY = 0.01


class RT:
    DELAY_SMOOTH = 0.85

    def __init__(self, processor, batch_size=1, view_size=(1280, 720)):
        self.batch_size = batch_size
        self.processor = processor
        self.cam = None
        self.req_queue = Queue()
        self.display_queue = Queue()
        self.delay = 0.
        self.time_per_frame = 0.
        self.view_size = view_size


    def start(self):
        if self.cam is None:
            self.cam = cv2.VideoCapture(0)
        self._stop = False
        # start transformer and display services
        threading.Thread(target=self.transform).start()
        threading.Thread(target=self.display).start()
        cv2.imshow('rtsal', np.zeros((10,10, 3)))
        while not self._stop:
            ret_val, img = self.cam.read()
            img = cv2.flip(img, 1)
            self.req_queue.put((time.time(), img))
            if cv2.waitKey(10) == 27:
                self._stop = True
        time.sleep(0.5)
        cv2.destroyAllWindows()

    def transform(self):
        while not self._stop:
            if self.req_queue.qsize()< self.batch_size:
                time.sleep(POLL_DELAY)
                continue
            to_proc = []
            while not self.req_queue.empty():
                to_proc.append(self.req_queue.get(timeout=0.1))

            if len(to_proc) > self.batch_size:
                # usual case, take self.batch_size equally separated items
                sep = int(len(to_proc) / self.batch_size)
                old = len(to_proc)
                to_proc = to_proc[:sep*self.batch_size:sep]
                assert len(to_proc) == self.batch_size

            imgs = np.concatenate(tuple(np.expand_dims(e[1], 0) for e in to_proc), 0)
            done_imgs = self.processor(imgs)

            for e in xrange(len(done_imgs)):
                im = done_imgs[e]
                t = to_proc[e][0]
                self.display_queue.put((t, im))

    def display(self):
        last_frame = time.time()
        while not self._stop:
            if self.display_queue.empty():
                time.sleep(POLL_DELAY)
                continue
            creation_time, im = self.display_queue.get(timeout=11)
            self.delay = self.DELAY_SMOOTH*self.delay + (1.-self.DELAY_SMOOTH)*(time.time() - creation_time)
            while time.time() < creation_time + self.delay:
                time.sleep(POLL_DELAY)
            self.time_per_frame = 0.9*self.time_per_frame + 0.1*(time.time() - last_frame)
            if self.view_size is not None:
                im = cv2.resize(im, self.view_size, interpolation=cv2.INTER_LINEAR)
            cv2.imshow('rtsal', im)
            last_frame = time.time()
            #print 'Delay %f  |  FPS %f' % (self.delay, self.fps)

    @property
    def fps(self):
        return 1./self.time_per_frame




class Saliency:
    def __init__(self, sess, batch_size, initial_looking_for=440):
        assert batch_size == 1, 'Only BS of 1 supported'
        self.sess = sess
        self.looking_for = initial_looking_for
        try:
            sal.get_bbs(sess, None, None)
        except:
            print 'Initialized'
        self.report_looking()

    def report_looking(self):
        print 'Looking for ', imagenet.CLASS_ID_TO_NAME[self.looking_for]


    def __call__(self, imgs):
        imgs = np.array(imgs)
        box = get_bbs(sess, imgs[0], self.looking_for)
        if box is None:
            return imgs
        else:
            print 'Found', imagenet.CLASS_ID_TO_NAME[self.looking_for]
            return np.array([utils.bounding_box.draw_box(imgs[0], box)])




sess = tf.Session()

a = RT(Saliency(sess, 1))

a.start()