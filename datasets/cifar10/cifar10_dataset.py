import numpy as np
from tfutils import *
from config import *
import time

def split_synset_line(s):
    b = s.split()[0]
    r = ' '.join(s.split()[1:])
    return b, r

# -------------------------------------------
synset = open(os.path.join(os.path.dirname(__file__), 'synset.txt')).read()
SYNSET_TO_NAME= dict(split_synset_line(e) for e in synset.splitlines())
SYNSET_TO_CLASS_ID = dict((split_synset_line(e)[0], i) for i, e in enumerate(synset.splitlines()))

CLASS_ID_TO_SYNSET = {v:k for k,v in SYNSET_TO_CLASS_ID.items()}
CLASS_ID_TO_NAME = {i:SYNSET_TO_NAME[CLASS_ID_TO_SYNSET[i]] for i in CLASS_ID_TO_SYNSET}

# these values are in RGB format
PIXEL_MEAN = 1. * np.array([124.8473, 121.17274, 112.0139])  # to find these just set mean to 0 and std to 1 and run to see the required values
PIXEL_STD = 1. * np.array([59.826111, 58.889648, 62.99823])

IMG_NET_PIXEL_EIG_VEC = np.array([
      [ -0.5675,  0.7192,  0.4009 ],
      [ -0.5808, -0.0045, -0.8140 ],
      [ -0.5836, -0.6948,  0.4203 ]])
IMG_NET_PIXEL_EIG_VAL = np.array([0.2175, 0.0188, 0.0045])

NUM_VALIDATION_IMGS_PER_EXAMPLE = len(VALIDATION_CROPS) * (bool(VALIDATION_USE_FLIPS) + 1)  # <- how many patches to evaluate per example
print 'NUM VAL' , NUM_VALIDATION_IMGS_PER_EXAMPLE
assert NUM_VALIDATION_IMGS_PER_EXAMPLE >= 1

TRAIN_IMAGES = get_images_op(IMAGE_NET_TRAIN_PATH)
VAL_IMAGES = get_images_op(IMAGE_NET_VAL_PATH)
random.shuffle(VAL_IMAGES)
#VAL_IMAGES = VAL_IMAGES[:5000]

# -------------------------------------------

print SYNSET_TO_CLASS_ID
#   ---- TRAIN PIPELINE ----

IMAGE_TRAIN_PIPELINE = compose_ops([
    load(),
    random_crop(IMAGE_SIZE, AREA_RANGE, ASPECT_RATIO_RANGE),
    probabilistic_op(0.5, horizontal_flip()),
    normalize(PIXEL_MEAN, PIXEL_STD),
    random_lighting(LIGHTNING_APLHA_STD, IMG_NET_PIXEL_EIG_VAL, IMG_NET_PIXEL_EIG_VEC)
])

LABEL_TRAIN_PIPELINE = compose_ops([
    folder_name(),
    key_to_element(SYNSET_TO_CLASS_ID)
])

TRAIN_PIPELINE = for_each(parallelise_ops([
    IMAGE_TRAIN_PIPELINE,
    LABEL_TRAIN_PIPELINE
]))


#   ---- VAL PIPELINE ----

val_imgs = [
        crop_at(IMAGE_SIZE, position=pos) for pos in VALIDATION_CROPS
    ] + ([] if not VALIDATION_USE_FLIPS else [
        compose_ops([crop_at(IMAGE_SIZE, position=pos), horizontal_flip()]) for pos in VALIDATION_CROPS
    ])

assert len(val_imgs) == NUM_VALIDATION_IMGS_PER_EXAMPLE

IMAGE_VAL_PIPELINE = compose_ops([
    load(),
    square_centrer_crop_resize(VALIDATION_CROP_FROM_SIZE),
    normalize(PIXEL_MEAN, PIXEL_STD),
    parallelise_ops(val_imgs)
])

LABEL_VAL_PIPELINE = compose_ops([
    folder_name(),
    key_to_element(SYNSET_TO_CLASS_ID),
    parallelise_ops(NUM_VALIDATION_IMGS_PER_EXAMPLE*[lambda x: x])
])

def flatten():
    def flatten_op(inp, typ=tuple):
        return sum(inp, typ())
    return flatten_op

VAL_PIPELINE = for_each(parallelise_ops([
    IMAGE_VAL_PIPELINE,
    LABEL_VAL_PIPELINE
]))



def val_batch_composer(type_data, type_label):
    def generic_batch_composer_op(examples):
        data = [k for e in examples for k in e[0]]
        labels = [k for e in examples for k in e[1]]
        return parallelise_ops([
            lambda x: (np.concatenate(tuple(np.expand_dims(e, 0) for e in data), 0)).astype(type_data),
            lambda x: (np.concatenate(tuple(np.expand_dims(e, 0) for e in labels), 0)).astype(type_label),
        ])(examples)
    return generic_batch_composer_op


def get_train_bm(batch_size, float16=False):
    return BatchManager(
        TRAIN_PIPELINE,
        TRAIN_IMAGES,
        generic_batch_composer(np.float32 if not float16 else np.float16, np.int32),
        batch_size,
        shuffle_examples=True,
        num_workers=1
    )


def get_val_bm(batch_size, float16=False):
    assert batch_size % NUM_VALIDATION_IMGS_PER_EXAMPLE == 0, 'batch_size must be divisible by number of validations tries per image'
    return BatchManager(
        VAL_PIPELINE,
        VAL_IMAGES,
        val_batch_composer(np.float32 if not float16 else np.float16, np.int32),
        batch_size/NUM_VALIDATION_IMGS_PER_EXAMPLE,
        shuffle_examples=True,
        num_workers=1
    )

def to_rgb_img(im):
    return np.clip(im * PIXEL_STD + PIXEL_MEAN, 0, 255).astype(np.uint8)

def to_bgr_img(im):
    return to_rgb_img(im)[:,:,[2,1,0]]


if __name__=='__main__':
    bm = get_train_bm(30)
    means = []
    stds = []
    for e in bm:
        means.append(np.mean(e[0], (0,1,2)))
        stds.append(np.std(e[0], (0,1,2)))
    print 'Mean', list(np.mean(np.stack(means), axis=0))
    print 'Stds', list(np.mean(np.stack(stds), axis=0))
    for im in xrange(len(e[0])):
        cv2.imshow('test', to_bgr_img(e[0][im]))
        cv2.waitKey(100000)
