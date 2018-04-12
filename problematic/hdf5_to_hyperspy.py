import glob
import h5py
from hyperspy.signals import Signal2D
import os, sys
import numpy as np


def hdf5_to_hyperspy(fns):
    if isinstance(fns, str):
        fns = glob.glob(fns)

    dat = []
    for fn in fns:
        f = h5py.File(fn)
        dat.append(np.array(f["data"]))
        f.close()
    
    try:
        f = h5py.File(fn)
        pixelsize = f["data"].attrs["ImagePixelsize"]
    except KeyError:
        pixelsize = 1
    finally:
        f.close()

    ed = Signal2D(dat)
    ed.metadata["General"]["title"] = "serialED"
    ed.axes_manager[0].name = "frame"
    ed.axes_manager[1].name = "X"
    ed.axes_manager[2].name = "Y"
    ed.axes_manager[1].units = "$A^{-1}$"
    ed.axes_manager[2].units = "$A^{-1}$"
    ed.axes_manager[1].scale = pixelsize
    ed.axes_manager[2].scale = pixelsize
    return ed


def main():
    filename = "serialED.hdf5"
    patt = sys.argv[1]
    fns = glob.glob(patt)
    ed = main(fns)
    ed.save(filename)


if __name__ == '__main__':
    main()

