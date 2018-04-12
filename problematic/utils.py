import numpy as np
from scipy import ndimage as ndi
from skimage import morphology, filters


def find_beam_position_blur(z, sigma=30):
    """Estimate direct beam position by blurring the image with a large
    Gaussian kernel and finding the maximum.

    Parameters
    ----------
    sigma : float
        Sigma value for Gaussian blurring kernel.

    Returns
    -------
    center : np.array
        np.array containing indices of estimated direct beam positon.
    """
    blurred = ndi.gaussian_filter(z, sigma, mode='wrap')
    center = np.unravel_index(blurred.argmax(), blurred.shape)
    return np.array(center)


def subtract_background_median(z, footprint=19, implementation='scipy'):
    """Remove background using a median filter.

    Parameters
    ----------
    footprint : int
        size of the window that is convoluted with the array to determine
        the median. Should be large enough that it is about 3x as big as the
        size of the peaks.
    implementation: str
        One of 'scipy', 'skimage'. Skimage is much faster, but it messes with
        the data format. The scipy implementation is safer, but slower.

    Returns
    -------
        Pattern with background subtracted as np.array
    """

    if implementation == 'scipy':
        bg_subtracted = z - ndi.median_filter(z, size=footprint)
    elif implementation == 'skimage':
        selem = morphology.square(footprint)
        # skimage only accepts input image as uint16
        bg_subtracted = z - filters.median(z.astype(np.uint16), selem).astype(z.dtype)
    else:
        raise ValueError("Unknown implementation `{}`".format(implementation))

    return np.maximum(bg_subtracted, 0)

