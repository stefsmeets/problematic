import pycrystem as pc
import hyperspy.api as hs
from hdf5_to_hyperspy import hdf5_to_hyperspy
from pycrystem.utils.peakfinders2D import find_peaks_regionprops
import os, sys
import numpy as np
import glob
import datetime
from stretch_correction import apply_stretch_correction
import io_utils


def im_reconstruct(props, shape=None, clip=True):
    """Takes a list of regionprops and reconstructs an image with the given shape"""
    z = np.zeros(shape)
    for prop in props:
        x0,y0,x1,y1 = prop.bbox
        z[x0:x1, y0:y1] = prop.intensity_image
    if clip:
        z = np.clip(z, 0, z.max())
    return z


def load(filepat):
    """Takes a file pattern (*.h5) to load a list of instamatic h5 files,
    or a hdf5 file in hyperspy format or a list of files

    returns serialED object
    """
    if isinstance(filepat, (list, tuple)):
        signal = hdf5_to_hyperspy(filepat)
    elif os.path.exists(filepat):
        root, ext = os.path.splitext(filepat)
        if ext.lower() == ".ycsv":
            df, d = io_utils.read_ycsv(filepat)
            fns = df.index.tolist()
            signal = hdf5_to_hyperspy(fns)
        elif ext.lower() in (".hspy", ".hdf5", ".h5"):
            signal = hs.load(filepat)
        else:
            f = open(filepat, "r")
            fns = [line.split("#")[0].strip() for line in f if not line.startswith("#")]
            signal = hdf5_to_hyperspy(fns)
    else:
        fns = glob.glob(filepat)
        signal = hdf5_to_hyperspy(fns)
    return serialED(**signal._to_dictionary())


def serialmerge_intensities(intensities, orientations, n=25):
    """Use serialmerge to merge intensities from best n orientations"""
    import pandas as pd
    from serialmerge import serialmerge
    
    best_orientations = [ori[0] for ori in orientations.data]
    
    scores = [ori.score for ori in best_orientations]
    ix = np.argsort(scores)[:-(n+1):-1]
    
    hklies = np.take(intensities.data, ix, axis=0).tolist()
    
    m = serialmerge(hklies, verbose=True)
    return m


class serialED(pc.ElectronDiffraction):

    _props_collection = []
    _centers = []
    _raw_orientations = []
    _orientations = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata.add_node("Processing")
        self.load_extras()

    def get_direct_beam_position(self, sigma=10):
        """Get the position of the direct beam in all frames using the blurring method

        Sigma: int
            Standard deviation for the gaussian kernel

        Returns:
            BaseSignal1D"""

        method = "blur"
        d = {"sigma": sigma,
        "method": method,
        "date": str(datetime.datetime.now()),
        "func": "pycrystem.ElectronDiffraction.get_direct_beam_position" }
        self.metadata.Processing["get_direct_beam_position"] = d

        centers = super().get_direct_beam_position(method=method, sigma=sigma, inplace=False)
        self._centers = centers
        return centers

    def apply_stretch_correction(self, centers, azimuth=-6.61, amplitude=2.43, inplace=False):
        """Applies a stretch correction to all images

        azimuth: float
            Direction of the azimuth in degrees with respect to the vertical axis (TODO: check this)
        amplitude: float
            The difference in percent between the long and short axes

        Returns:
            serialED(BaseSignal2D)"""

        d = {"azimuth": azimuth,
        "amplitude": amplitude,
        "date": str(datetime.datetime.now()),
        "func": "serialED.apply_stretch_correction" }
        self.metadata.Processing["apply_stretch_correction"] = d

        return self.map(apply_stretch_correction, azimuth=azimuth, amplitude=amplitude, center=centers, inplace=inplace)

    def remove_background(self, footprint=19, inplace=False):
        """Removes the background of each frame using a median filter

        footprint:
            size of the window for determining the median

        Returns:
            serialED(BaseSignal2D)"""

        method = "median"
        d = {"footprint": footprint,
        "method": method,
        "date": str(datetime.datetime.now()),
        "func": "pycrystem.ElectronDiffraction.remove_background" }
        self.metadata.Processing["remove_background"] = d

        return super().remove_background(method=method, footprint=footprint, inplace=inplace)

    def find_peaks_and_clean_images(self, min_sigma=4, max_sigma=5, threshold=1, min_size=50, inplace=False):
        """Find regions of connected pixels using regionprops method, and use those to remove noise
        (i.e. clean the images). This is done by setting the intensities in areas areas that do not 
        belong to the connected regions are set to zero.

        min_sigma : int, float
            Standard deviation for the minimum gaussian convolution
        max_sigma : int, float
            Standard deviation for the maximum gaussian convolution
        threshold : int, float
            Minimum difference in intensity
        min_size : int
            Minimum size in pixels of blob

        Returns:
            serialED(BaseSignal2D)

        regionprops are stored to self._props_collection
        """
        d = {
        "min_sigma": min_sigma,
        "max_sigma": max_sigma,
        "threshold": threshold,
        "min_size": min_size,
        "date": str(datetime.datetime.now()),
        "func": "pycrystem.utils.peakfinders2D.find_peaks_regionprops" }

        props_collection = self.map(find_peaks_regionprops, 
                             min_sigma=min_sigma, max_sigma=max_sigma, 
                             threshold=threshold, min_size=min_size, return_props=True, inplace=False)

        self._props_collection = props_collection
        
        def im_reconstruct_func(z, props):
            return im_reconstruct(props, shape=z.shape)

        d["nprops"] = len(props_collection)
        self.metadata.Processing["find_peaks_and_clean_images"] = d

        return self.map(im_reconstruct_func, props=props_collection, inplace=inplace)

    def find_orientations(self, indexer, centers, nsolutions=25, filter1d=False, nprojs=100):
        """Find the orientations for all the images using a brute force forward projection method

        indexer: indexer.Indexer
             See `indexer.Indexer` documention for how to set this up
        centers: BaseSignal1D
            List of centers of the diffraction patterns
        nsolutions: int
            Number of best solutions to keep for each fram
        filter1d: bool
            Use a powder pattern as a first selection criterion to determine
            if a frame is good or nto
        nprojs: int
            Number of projections to keep, corresponding to filter1d

        Returns:
            BaseSignal1D containing a list of `nsolutions` namedtuples

        Example:
            pixelsize = 0.00433
            dmin, dmax = 1.0, 10.0
            params = (24.3450,)
            spgr = "Fd-3m"
            name = "FAU"
            projector = Projector.from_parameters(params, spgr=spgr, name=name, dmin=dmin, dmax=dmax, thickness=thickness)
            indexer = Indexer.from_projector(projector, pixelsize=pixelsize)
            orientations = ed.find_orientations(indexer, centers)

            # visualize results
            ed.orientation_explorer(indexer, orientations)

            # save results
            serialED.io_utils.save_orientations(orientations)
        """
        d = indexer.to_dict()
        d["date"] = str(datetime.datetime.now())
        d["func"] = "indexer.find_orientation"
        self.metadata.Processing["find_orientations"] = d
        orientation_collection = self.map(indexer.find_orientation, center=centers, nsolutions=nsolutions,
                                          filter1d=filter1d, nprojs=nprojs, 
                                          parallel=False, inplace=False)
        self._raw_orientations = orientation_collection
        return orientation_collection

    def refine_orientations(self, indexer, orientation_collection, sort=True, method="powell", 
                            vary_scale=True, vary_center=True, vary_alphabeta=True):
        """Refine the orientations obtained using serialED.find_orientations

        indexer: indexer.Indexer
             See `indexer.Indexer` documention for how to set this up
        orientation_collection: BaseSignal1D
            Collection of orientations
        sort: bool
            Sort the orientations
        method: str
            Method to use for the least-squares minimization
        vary_scale: bool
            Refine the scale (1/pixelsize)
        vary_center: bool
            Refine the center of the beam
        vary_alphabeta: bool
            Refine the orientation of the crystals

        Returns:
            BaseSignal1D containing a list of namedtuples

        Example:
            orientations = ed.find_orientations(indexer, centers)

            # refine the orientations
            refined_orientations = ed.refine_orientations(indexer, orientations)

            # visualize results
            ed.orientation_explorer(indexer, refined_orientations)

            # save results
            serialED.io_utils.save_orientations(refined_orientations)
        """
        d = indexer.to_dict()
        d["vary_scale"] = vary_scale
        d["vary_center"] = vary_center
        d["vary_alphabeta"] = vary_alphabeta
        d["method"] = method
        d["date"] = str(datetime.datetime.now())
        d["func"] = "indexer.refine_all"
        self.metadata.Processing["refine_orientations"] = d

        orientation_collection = self.map(indexer.refine_all, results=orientation_collection, sort=sort, method=method, 
                                          vary_scale=vary_scale, vary_center=vary_center, vary_alphabeta=vary_alphabeta, 
                                          parallel=False, inplace=False)
        self._orientations = orientation_collection
        return orientation_collection

    def orientation_explorer(self, indexer, orientations, imshow_kwargs={}):
        """Visualize a collection of orientations

        indexer: indexer.Indexer
             See `indexer.Indexer` documention for how to set this up
        orientation_collection: BaseSignal1D
            Collection of orientations
        imshow_kwargs: dict
            List of keyword arguments to pass to plt.imshow      

        Example
            orientations = ed.find_orientations(indexer, centers)

            # visualize results
            ed.orientation_explorer(indexer, orientations, imshow_kwargs={"vmax":100, "cmap":"gray"})"""
        from peakexplorer import PeakExplorer
        default_imshow_kwargs = {"cmap":"gray"}
        default_imshow_kwargs.update(imshow_kwargs)
        peakexp = PeakExplorer(indexer=indexer, orientations=orientations, imshow_kwargs=default_imshow_kwargs)
        peakexp.interactive(self)

    def plot_orientations(self, indexer, orientation_collection, n=25, vmin=None, vmax=None, ax=None):
        """plot orientations using matplotlib

        indexer: indexer.Indexer
             See `indexer.Indexer` documention for how to set this up
        orientation_collection: BaseSignal1D
            Collection of orientations
        n: int
            Number of best orientations to show
        vmax, vmin: float
            Passed to imshow
        ax: matplotlib.Axes
        """

        import matplotlib.pyplot as plt
        scores = np.array([ori[0].score for ori in orientation_collection.data])
        ix = np.argsort(scores)[::-1]

        for i in ix:
            indexer.plot(self.data[i], orientation_collection.data[i][0], title="Image #"+ str(i), 
                         vmin=vmin, vmax=vmax, ax=ax)
        
        if not ax:
            plt.show()

    def extract_intensities(self, indexer, orientations, outdir="intensities"):
        """Extracts the intensities at the predicted spots

        indexer: indexer.Indexer
             See `indexer.Indexer` documention for how to set this up
        orientation_collection: BaseSignal1D
            Collection of orientations
        outdir: str
            Directory where to store the hkl files with the extracted intensities

        Returns:
            BaseSignal1D: Contains lists of extracted intensities"""
        d = {}
        d["outdir"] = outdir
        d["date"] = str(datetime.datetime.now())
        d["func"] = "indexer.get_intensities"
        self.metadata.Processing["extract_intensities"] = d

        def func(z, orientations, indexer):
            orientation = orientations[0]
            intensities = indexer.get_intensities(z, orientation)
            return intensities
        
        intensities_collection = self.map(func, orientations=orientations, indexer=indexer, inplace=False)

        if outdir:
            for i, intensities in enumerate(intensities_collection.data):
                out = os.path.join(outdir, "intensities_{:05d}.hkl".format(i))
                np.savetxt(out, intensities, fmt="%4d%4d%4d %7.1f %7.1f")

        return intensities_collection

    def export_indexing_results(s, fname="orientations.ycsv"):
        """Export the indexing results to a ycsv file"""
        import pandas as pd
        import yaml
        d = s.metadata.Processing["find_orientations"].as_dictionary()
        d["title"] = s.metadata.General["title"]
        try:
            d["data"] = {"outdir": s.metadata.Processing["extract_intensities"]["outdir"] }
        except (KeyError, AttributeError):
            d["data"] = None
        df = pd.DataFrame([ori[0] for ori in s._orientations.data])
        io_utils.write_ycsv(fname, data=df, metadata=d)

    def load_extras(self):
        if hasattr(self.metadata, "Processing"):
            try:
                d = self.metadata.Processing["datfiles"].as_dictionary()
            except (KeyError, AttributeError):
                return False

            f_centers = d.get("centers", None)
            if f_centers and os.path.exists(f_centers):
                self._centers = hs.load(f_centers)
 
            f_orientations = d.get("orientations", None)
            if f_orientations and os.path.exists(f_orientations):
                self._orientations = hs.load(f_orientations)
 
            f_raw_orientations = d.get("raw_orientations", None)
            if f_raw_orientations and os.path.exists(f_raw_orientations):
                self._raw_orientations = hs.load(f_raw_orientations)
            return True
        else:
            return False

    def save(self, filename, *args, **kwargs):
        """Save current progress"""
        root, ext = os.path.splitext(filename)
        d = {}
        if self._centers:
            f_centers = root + "_centers" + ext
            self._centers.save(f_centers)
            d["centers"] = f_centers
        if self._orientations:
            f_orientations = root + "_orientations" + "npy"
            io_utils.save_orientations(self._orientations, out=f_orientations)
            d["orientations"] = f_orientations
        if self._raw_orientations:
            f_raw_orientations = root + "_raw_orientations" + "npy"
            io_utils.save_orientations(self._raw_orientations, out=f_raw_orientations)
            d["raw_orientations"] = f_raw_orientations
        self.metadata.Processing["datfiles"] = d
        super().save(filename=filename)

    def deepcopy(self, *args, **kwargs):
        import copy
        new = super().deepcopy(*args, **kwargs)
        new._centers = copy.copy(self._centers)
        new._raw_orientations = copy.copy(self._raw_orientations)
        new._orientations = copy.copy(self._orientations)
        new._props_collection = copy.copy(self._props_collection)
        return new

    def select(self, i, j):
        """Select part of the data set for processing, from row i to j"""
        signal_dict = self._to_dictionary(add_learning_results=False)
        signal_dict['data'] = self.data[i:j]
        signal_dict["axes"][0]["size"] = j-i
        signal_dict["axes"][0]["offset"] = i
        return self.__class__(**signal_dict)
