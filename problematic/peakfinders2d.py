import numpy as np
import scipy.ndimage as ndi

NO_PEAKS = np.array([[np.nan, np.nan]])


def clean_peaks(peaks):
    if len(peaks) == 0:
        return NO_PEAKS
    else:
        return peaks


def find_peaks_regionprops(z, min_sigma=4, max_sigma=5, threshold=1,
                           min_size=50, return_props=False):
    """
    Finds peaks using regionprops.
    Uses the difference of two gaussian convolutions to separate signal from
    background, and then uses the skimage.measure.regionprops function to find
    connected islands (peaks). Small blobs can be rejected using `min_size`.

    Parameters
    ----------
    z : numpy.ndarray
        Array of image intensities.
    min_sigma : int, float
        Standard deviation for the minimum gaussian convolution
    max_sigma : int, float
        Standard deviation for the maximum gaussian convolution
    threshold : int, float
        Minimum difference in intensity
    min_size : int
        Minimum size in pixels of blob
    return_props : bool
        Return skimage.measure.regionprops

    Returns
    -------
    numpy.ndarray
        (n_peaks, 2)
        Array of peak coordinates.

    """
    from skimage import morphology, measure

    difference = ndi.gaussian_filter(z, min_sigma) - ndi.gaussian_filter(z, max_sigma)

    labels, numlabels = ndi.label(difference > threshold)
    labels = morphology.remove_small_objects(labels, min_size)

    props = measure.regionprops(labels, z)

    if return_props:
        return props
    else:
        peaks = np.array([prop.centroid for prop in props])
        return clean_peaks(peaks)
