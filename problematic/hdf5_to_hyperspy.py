import glob
import h5py as h5
import hyperspy.api as hs
import pyxem as pxm
import os, sys
import numpy as np

cameralength2pixelsize = {
    150: 0.0087845,
    200: 0.0065884,
    250: 0.0052707,
    300: 0.0043922,
    400: 0.0032942,
    500: 0.0026353 }

def hdf5_to_hyperspy(fns):
    if isinstance(fns, str):
        fns = glob.glob(fns)

    dat = []
    for fn in fns:
        f = h5.File(fn)
        dat.append(np.array(f["data"]))
        f.close()
    
    try:
        f = h5.File(fn)
        pixelsize = f["data"].attrs["ImagePixelsize"]
    except KeyError:
        try:
            cl = f["data"].attrs["Magnification"]
            pixelsize = cameralength2pixelsize[cl]
        except KeyError:
            pixelsize = 1
    finally:
        f.close()

    ed = pxm.ElectronDiffraction(dat)
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

