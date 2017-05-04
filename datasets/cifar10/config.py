IMAGE_NET_TRAIN_PATH = '/home/piter/cifar10/train/'
IMAGE_NET_VAL_PATH = '/home/piter/cifar10/test/'

IMAGE_SIZE = (32, 32)

# Data augmentation
ASPECT_RATIO_RANGE = (3./4, 4./3)
AREA_RANGE = (0.5, 1.)
LIGHTNING_APLHA_STD = 0.33

# Validation
use_complex_val = 0
if use_complex_val:  # 10 crop validation
    VALIDATION_CROP_FROM_SIZE = (36, 36)
    VALIDATION_CROPS = ['CC', 'RL', 'LR', 'RR', 'LL']
    VALIDATION_USE_FLIPS = 1
else:
    VALIDATION_CROP_FROM_SIZE = IMAGE_SIZE
    VALIDATION_CROPS = ['CC']
    VALIDATION_USE_FLIPS = 0


