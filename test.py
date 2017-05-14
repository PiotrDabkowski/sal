import tensorflow as tf
import numpy as np
import cv2
import utils.fixed_image
import utils.bounding_box

im = cv2.imread('dog2.jpg').astype(np.float32)/255.

c = utils.fixed_image.FixedAspectRatioNoCropping(im, 500)
im2 = c.get_resulting_img()

im3 = utils.bounding_box.draw_box(im2, box=c.to_local_coords(11,121,67,235))

print c.from_local_coords(-1000, -1000, 1000, 1000)


cv2.imshow('aaa', im3)
cv2.waitKey(100000)