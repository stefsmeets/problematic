from scipy import ndimage
import numpy as np


def find_beam_center(img, sigma=30):
    "Find position of the central beam using gaussian filter"
    blurred = ndimage.gaussian_filter(img, sigma)
    center = np.unravel_index(blurred.argmax(), blurred.shape)
    return np.array(center)


def get_files(file_pat):
    """Grab files from globbing pattern or stream file"""
    from instamatic.formats import read_ycsv
    if os.path.exists(file_pat):
        root, ext = os.path.splitext(file_pat)
        if ext.lower() == ".ycsv":
            df, d = read_ycsv(file_pat)
            fns = df.index.tolist()
        else:
            f = open(file_pat, "r")
            fns = [line.split("#")[0].strip() for line in f if not line.startswith("#")]
    else:
        fns = glob.glob(file_pat)

    if len(fns) == 0:
        raise IOError("No files matching '{}' were found.".format(file_path))

    return fns

