import pycrystem as pc
from hdf5_to_hyperspy import hdf5_to_hyperspy
from pycrystem.utils.peakfinders2D import find_peaks_regionprops
import os, sys
import numpy as np
import scipy.ndimage as ndi


def apply_transform_to_image(img, transform, center=None):
    """Applies transformation matrix to image and recenters it
    http://docs.sunpy.org/en/stable/_modules/sunpy/image/transform.html
    http://stackoverflow.com/q/20161175
    """

    if center is None:
        center = (np.array(img.shape)[::-1]-1)/2.0
    # shift = (center - center.dot(transform)).dot(np.linalg.inv(transform))
    
    displacement = np.dot(transform, center)
    shift = center - displacement
    
    # linear interpolation (order=1) is required to avoid negative intensities in corrected image
    img_tf = ndi.interpolation.affine_transform(img, transform, offset=shift, mode="constant", order=1, cval=0.0)
    return img_tf


def affine_transform_ellipse_to_circle(azimuth, amplitude, inverse=False):
    """Usage: 

    e2c = circle_to_ellipse_affine_transform(azimuth, amplitude):
    np.dot(arr, e2c) # arr.shape == (n, 2)
       or
    apply_transform_to_image(img, e2c)

    http://math.stackexchange.com/q/619037
    """
    sin = np.sin(azimuth)
    cos = np.cos(azimuth)
    sx    = 1 - amplitude
    sy    = 1 + amplitude
    
    # apply in this order
    rot1 = np.array((cos, -sin,  sin, cos)).reshape(2,2)
    scale = np.array((sx, 0, 0, sy)).reshape(2,2)
    rot2 = np.array((cos,  sin, -sin, cos)).reshape(2,2)
    
    composite = rot1.dot(scale).dot(rot2)
    
    if inverse:
        return np.linalg.inv(composite)
    else:
        return composite


def apply_stretch_correction(z, center=None, azimuth=0, amplitude=0):
    azimuth_rad = np.radians(azimuth)    # go to radians
    amplitude_pc = amplitude / (2*100)   # as percentage
    tr_mat = affine_transform_ellipse_to_circle(azimuth_rad, amplitude_pc)
    z = apply_transform_to_image(z, tr_mat, center=center)
    return z


def im_reconstruct(props, shape=None):
    z = np.zeros(shape)
    for prop in props:
        x0,y0,x1,y1 = prop.bbox
        z[x0:x1, y0:y1] = prop.intensity_image
    return z


def load(filepat):
    if os.path.exists(filepat):
        ed = pc.load(filepat)
    else:
        fns = glob.glob(filepat)
        ed = hdf5_to_hyperspy(fns)
    return serialED(ed)


class serialED(pc.ElectronDiffraction):

    _get_direct_beam_position = pc.ElectronDiffraction.get_direct_beam_position
    _remove_background = pc.ElectronDiffraction.remove_background
    _props_collection = []
    _centers = []
    _raw_orientation_collection = []
    _orientation_collection = []

    def get_direct_beam_position(self, sigma=10):
        centers = self._get_direct_beam_position(method="blur", sigma=sigma, inplace=False)
        self._centers = centers
        return centers

    def apply_stretch_correction(self, centers, azimuth=-6.61, amplitude=2.43, inplace=False):
        return self.map(apply_stretch_correction, azimuth=azimuth, amplitude=amplitude, center=centers, inplace=inplace)

    def remove_background(self, footprint=19, inplace=False):
        return self._remove_background(method="median", footprint=footprint, inplace=inplace)

    def find_peaks_and_clean_images(self, min_sigma=4, max_sigma=5, threshold=1, min_size=50, inplace=False):
        props_collection = self.map(find_peaks_regionprops, 
                             min_sigma=min_sigma, max_sigma=max_sigma, 
                             threshold=threshold, min_size=min_size, return_props=True, inplace=False)

        self._props_collection = props_collection
        
        def im_reconstruct_func(z, props):
            return im_reconstruct(props, shape=z.shape)

        return self.map(im_reconstruct_func, props=props_collection, inplace=inplace)

    def find_orientations(self, indexer, centers, nsolutions=25, filter1d=False, nprojs=100):
        orientation_collection = self.map(indexer.find_orientation, center=centers, nsolutions=nsolutions,
                                          filter1d=filter1d, nprojs=nprojs, 
                                          parallel=False, inplace=False)
        self._raw_orientation_collection = orientation_collection
        return orientation_collection

    def refine_orientations(self, indexer, orientation_collection, sort=True, method="powell", 
                            vary_scale=True, vary_center=True, vary_alphabeta=True):
        orientation_collection = self.map(indexer.refine_all, results=orientation_collection, sort=sort, method=method, 
                                          vary_scale=vary_scale, vary_center=vary_center, vary_alphabeta=vary_alphabeta, 
                                          parallel=False, inplace=False)
        self._orientation_collection = orientation_collection
        return orientation_collection
