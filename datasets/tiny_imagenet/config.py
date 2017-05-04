IMAGE_NET_TRAIN_PATH = '/home/piter/ImageNetTiny/train/'
IMAGE_NET_VAL_PATH = '/home/piter/ImageNetTiny/val/'

IMAGE_SIZE = (64, 64)

# Data augmentation
ASPECT_RATIO_RANGE = (3./4, 4./3)
AREA_RANGE = (0.66, 1.)
LIGHTNING_APLHA_STD = 0.33

# Validation
use_complex_val = 0
if use_complex_val:  # 10 crop validation
    VALIDATION_CROP_FROM_SIZE = (80, 80)
    VALIDATION_CROPS = ['CC', 'RL', 'LR', 'RR', 'LL']
    VALIDATION_USE_FLIPS = 1
else:
    VALIDATION_CROP_FROM_SIZE = IMAGE_SIZE
    VALIDATION_CROPS = ['CC']
    VALIDATION_USE_FLIPS = 0





