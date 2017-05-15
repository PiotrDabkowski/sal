import numpy as np
from tfutils import *
from config import *
import time
import warnings
import random


# -------------------------------------------
synset = open(os.path.join(os.path.dirname(__file__), 'synset.txt')).read()
SYNSET_TO_NAME= dict((e[:9], e[10:]) for e in synset.splitlines())
SYNSET_TO_CLASS_ID = dict((e[:9], i) for i, e in enumerate(synset.splitlines()))

CLASS_ID_TO_SYNSET = {v:k for k,v in SYNSET_TO_CLASS_ID.items()}
CLASS_ID_TO_NAME = {i:SYNSET_TO_NAME[CLASS_ID_TO_SYNSET[i]] for i in CLASS_ID_TO_SYNSET}

# these values are in RGB format
IMAGE_NET_PIXEL_MEAN = 256.0*np.array([0.485, 0.456, 0.406])
IMAGE_NET_PIXEL_STD = 256.0*np.array([0.229, 0.224, 0.225])

IMG_NET_PIXEL_EIG_VEC = np.array([
      [ -0.5675,  0.7192,  0.4009 ],
      [ -0.5808, -0.0045, -0.8140 ],
      [ -0.5836, -0.6948,  0.4203 ]])
IMG_NET_PIXEL_EIG_VAL = np.array([0.2175, 0.0188, 0.0045])

NUM_VALIDATION_IMGS_PER_EXAMPLE = len(VALIDATION_CROPS) * (bool(VALIDATION_USE_FLIPS) + 1)  # <- how many patches to evaluate per example
print 'NUM VAL' , NUM_VALIDATION_IMGS_PER_EXAMPLE
assert NUM_VALIDATION_IMGS_PER_EXAMPLE >= 1

try:
    TRAIN_IMAGES = get_images_op(IMAGE_NET_TRAIN_PATH)
    VAL_IMAGES = get_images_op(IMAGE_NET_VAL_PATH)
except:
    warnings.warn(Warning('Specified ImageNet folders not found!'))
    TRAIN_IMAGES = []
    VAL_IMAGES = []
random.shuffle(VAL_IMAGES)
#VAL_IMAGES = VAL_IMAGES[:5000]

# -------------------------------------------


#   ---- TRAIN PIPELINE ----

IMAGE_TRAIN_PIPELINE = compose_ops([
    load(),
    random_crop(IMAGE_SIZE, AREA_RANGE, ASPECT_RATIO_RANGE),
    probabilistic_op(0.5, horizontal_flip()),
    normalize(IMAGE_NET_PIXEL_MEAN, IMAGE_NET_PIXEL_STD),
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
    normalize(IMAGE_NET_PIXEL_MEAN, IMAGE_NET_PIXEL_STD),
    #lambda x: x[:,:,[2,1,0]],
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



def load_normalized_without_resize(path):
    return compose_ops([
        load(),
        normalize(IMAGE_NET_PIXEL_MEAN, IMAGE_NET_PIXEL_STD)
    ])(path)


def from_rgb(rgb):
    return compose_ops([
        square_centrer_crop_resize(IMAGE_SIZE),
        normalize(IMAGE_NET_PIXEL_MEAN, IMAGE_NET_PIXEL_STD)
    ])(rgb.astype(np.float32))

def from_bgr(bgr):
    return from_rgb(bgr[:,:,[2,1,0]])

def from_bgr_normalize_only(bgr):
    return normalize(IMAGE_NET_PIXEL_MEAN, IMAGE_NET_PIXEL_STD)(bgr[:,:,[2,1,0]].astype(np.float32))

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
        num_workers=6
    )


def get_val_bm(batch_size, float16=False):
    assert batch_size % NUM_VALIDATION_IMGS_PER_EXAMPLE == 0, 'batch_size must be divisible by number of validations tries per image'
    return BatchManager(
        VAL_PIPELINE,
        VAL_IMAGES,
        val_batch_composer(np.float32 if not float16 else np.float16, np.int32),
        batch_size/NUM_VALIDATION_IMGS_PER_EXAMPLE,
        shuffle_examples=True,
        num_workers=5
    )

def to_rgb_img(im):
    return np.clip(im*IMAGE_NET_PIXEL_STD + IMAGE_NET_PIXEL_MEAN, 0, 255).astype(np.uint8)

def to_bgr_img(im):
    if len(im.shape)==3:
        return to_rgb_img(im)[:,:,[2,1,0]]
    else:
        assert len(im.shape)==4
        return to_rgb_img(im)[:,:,:,[2,1,0]]



if __name__=='__main__':
    example_batch = iter(get_train_bm(20)).next()
    i = 0
    for im, label in zip(*example_batch):
        name = 'imgs/'+CLASS_ID_TO_NAME[label]+'- %d.jpg' % i
        cv2.imwrite(name, to_bgr_img(im))
        i+=1
