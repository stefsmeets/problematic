from scipy import ndimage
import numpy as np


def find_beam_center(img, sigma=30):
    "Find position of the central beam using gaussian filter"
    blurred = ndimage.gaussian_filter(img, sigma)
    center = np.unravel_index(blurred.argmax(), blurred.shape)
    return np.array(center)
