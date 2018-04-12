# problematic

Collection of routines for processing serial electron diffracton data collected using [instamatic](http://github.com/stefsmeets/instamatic).

## Usage

### Command line

Visualize the data collected (both images and diffraction data) in a serial ED experiment:

    problematic.browser images/image*.h5

Data processing is done using IPython notebooks. To generate a template input file (named indexing_template.ipynb):

    problematic.index --template

The notebook can then be opened after the IPython notebook server has been started:

    jupyter notebook

### Python

Imports:

```python
%matplotlib tk
from problematic import Indexer, Projector
from problematic import serialED
```

Load some data:

```python
ed = serialED.load("data/*.h5")
```

Show data:

```python
ed.plot(vmax=300)
```

Find the position of the direct beam:

```python
beam_center_sigma = 10       # sigma of the gaussian kernel
centers = ed.get_direct_beam_position(sigma=beam_center_sigma)
```

Remove the background:

```python
background_footprint = 19    # window for the median filter
processed = ed.remove_background(footprint=background_footprint)
```

Correct for the (elliptical) lens distortion:

```python
stretch_azimuth = -6.61      # orientation of the major axis of the ellipse
stretch_amplitude = 2.43     # percent difference between the major/minor axes
processed = processed.apply_stretch_correction(azimuth=stretch_azimuth, amplitude=stretch_amplitude, centers=centers)
```

Use interactive peak finder (use `regionprops` to find ideal parameters for next function):

```python
processed.find_peaks_interactive(imshow_kwargs={"vmax":300, "cmap":"gray"})
```

Find regions of connected pixels and clean images:

```python
min_sigma=2               # sigma of the minimum gaussian filter
max_sigma=5               # sigma of the maximum gaussian filter
threshold=1               # minimum intensity threshold for a peak
min_size=30               # minimum number of pixels for a peak
processed = processed.find_peaks_and_clean_images(min_sigma=min_sigma, max_sigma=max_sigma, 
                                                  threshold=threshold, min_size=min_size)
```

Generate indexer object:

```python
name = "FAU"              # name of the phase
pixelsize = 0.00433       # pixel per Angstrom
dmin, dmax = 1.0, 10.0    # Angstrom
thickness = 100           # nm used to estimate the width of the reflections (max. excitation error)
params = (24.3450,)       # cell parameters
spgr = "Fd-3m"            # space group
projector = Projector.from_parameters(params, spgr=spgr, name=name, 
                                      dmin=dmin, dmax=dmax, thickness=thickness)
indexer = Indexer.from_projector(projector, pixelsize=pixelsize)
```

Find orientations:

```python
orientations = processed.find_orientations(indexer, centers)
```

Refine and save orientations:

```python
refined = processed.refine_orientations(indexer, orientations)
serialED.io_utils.save_orientations(refined)
```

Show best orientations:

```python
ed.orientation_explorer(indexer, refined, imshow_kwargs={"vmax":300, "cmap":"gray"})
```

Export indexing results to ycsv file (yaml+csv):

```python
processed.export_indexing_results(fname="orientations.ycsv")
```

Load orientations and extract intensities:
    
```python
orientations = serialED.io_utils.load_orientations()
intensities = processed.extract_intensities(orientations=orientations, indexer=indexer)
```
    
Merge intensities from best 50 orientations using serialmerge algorithm:

```python
m = serialED.serialmerge_intensities(intensities, orientations, n=50)
```

Save all data in `hdf5` format:

```python
processed.save("processed.hdf5")
```

Load all data:

```python
processed = serialED.load("processed.hdf5")
```

## Requirements

- Python3.6
- [HyperSpy](http://hyperspy.org/)
- [sginfo](http://cci.lbl.gov/sginfo/) must be available as `sginfo` on the search path
- ...

## Install using Conda

Get miniconda from https://conda.io/miniconda.html (Python3.6)

    conda install hyperspy -c conda-forge
    pip install https://github.com/stefsmeets/problematic/archive/master.zip

## Installation

Using pip:

    pip install https://github.com/stefsmeets/problematic/archive/master.zip
