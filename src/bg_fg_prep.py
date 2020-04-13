import matplotlib.image as img
import glob

import numpy as np
from scipy.ndimage import filters, measurements, interpolation
from math import pi
import cv2

def image_histogram_equalization(image):
    return cv2.equalizeHist(image)


for saliency in glob.glob("input/images/segmentation/*.jpg"):
    print(saliency)
    # s = img.imread(saliency)
    s = cv2.imread(saliency,0)
    # print(s.shape)
    image = image_histogram_equalization(s)
    print(image.max())
    image[image>255 - 15.5] = 255
    image[image<=255 - 15.5] = 0
    print(r"input/images/segmentation_fg/" + saliency[len("input/images/segmentation//"):])
    cv2.imwrite(r"input/images/segmentation_fg/" + saliency[len("input/images/segmentation/"):], image)

    image = image_histogram_equalization(s)
    print(image.max())
    v = np.zeros_like(image)
    v[image>15.5] = 0
    v[image<=15.5] = 255
    print(r"input/images/segmentation_bg/" + saliency[len("input/images/segmentation/"):])
    cv2.imwrite(r"input/images/segmentation_bg/" + saliency[len("input/images/segmentation/"):], v)