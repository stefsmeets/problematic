# problematic

Program for processing serial electron diffracton data collected using [instamatic](http://github.com/stefsmeets/instamatic).

## Usage

Imports:

    %matplotlib tk
    from problematic import Indexer, Projector
    from problematic import serialED

Load some data:

    ed = serialED.load("data/*.h5")

Show data:

    ed.plot(vmax=300)

Find the position of the direct beam:

    beam_center_sigma = 10
    centers = ed.get_direct_beam_position(sigma=beam_center_sigma)

Remove the background:

    background_footprint = 19
    processed = ed.remove_background(footprint=background_footprint)

Correct for the lens distortion:

    stretch_azimuth = -6.61
    stretch_amplitude = 2.43
    processed = processed.apply_stretch_correction(azimuth=stretch_azimuth, amplitude=stretch_amplitude, centers=centers)

Use interactive peak finder (use `regionprops` to find ideal parameters for next function):

    processed.find_peaks_interactive(imshow_kwargs={"vmax":300, "cmap":"gray"})

Find regions of connected pixels and clean images:

    min_sigma=2
    max_sigma=5
    threshold=1
    min_size=30
    processed = processed.find_peaks_and_clean_images(min_sigma=min_sigma, max_sigma=max_sigma, 
                                                      threshold=threshold, min_size=min_size)

Generate indexer object:

    name = "FAU"
    pixelsize = 0.00433
    dmin, dmax = 1.0, 10.0
    params = (24.3450,)
    spgr = "Fd-3m"
    projector = Projector.from_parameters(params, spgr=spgr, name=name, dmin=dmin, dmax=dmax, thickness=thickness)
    indexer = Indexer.from_projector(projector, pixelsize=pixelsize)

Find orientations:

    orientations = processed.find_orientations(indexer, centers)

Refine and save orientations:

    refined = processed.refine_orientations(indexer, orientations)
    serialED.io_utils.save_orientations(refined)

Show best orientations:

    ed.orientation_explorer(indexer, refined, imshow_kwargs={"vmax":300, "cmap":"gray"})

Export indexing results to ycsv file (yaml+csv):

    processed.export_indexing_results(fname="orientations.ycsv")

Load orientations and extract intensities:
    
    orientations = serialED.io_utils.load_orientations()
    intensities = processed.extract_intensities(orientations=orientations, indexer=indexer)
    
Merge intensities from best 50 orientations using serialmerge algorithm:

    m = serialED.serialmerge_intensities(intensities, orientations, n=50)

Save all data in `hdf5` format:

    processed.save("processed.hdf5")

Load all data:

    processed = serialED.load("processed.hdf5")

## Requirements

- Python3.6
- PyCrystEM
- HyperSpy
- ...

## Install using Conda

    Get miniconda from https://conda.io/miniconda.html (Python3.6)

    conda install hyperspy -c conda-forge
    conda install --channel matsci pymatgen
	conda install cython
    pip install transforms3d
    pip install https://github.com/pycrystem/pycrystem/archive/master.zip
    pip install https://github.com/stefsmeets/problematic/archive/master.zip


## Installation

Using pip:

    pip install https://github.com/stefsmeets/problematic/archive/master.zip

