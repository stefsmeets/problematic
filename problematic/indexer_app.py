import os, sys, glob

import argparse
import pandas as pd

import time

from .indexer import *
from tqdm import tqdm
import logging

from .io_utils import get_files

import yaml

import serialED

from hyperspy.defaults_parser import preferences
preferences.General.nb_progressbar = False


def copy_template():
    """Copy template ipython file to current directory"""

    from shutil import copyfile

    fn = "indexing_template.ipynb"
    src = os.path.join(os.path.split(os.path.abspath(__file__))[0], fn)
    dst = os.path.join(os.path.abspath(os.path.curdir), fn)

    if not os.path.exists(src):
        raise IOError("No such file: {}".format(src))
        sys.exit()
    if os.path.exists(dst):
        raise IOError("File already exists: {}".format(dst))
        sys.exit()

    copyfile(src, dst)

    return True


def merge_csv(csvs):
    """Read amd combine csv files `csvs` into one df,

    Returns: pd.DataFrame with combined items"""
    combined = pd.concat((read_csv(csv) for csv in csvs))

    for csv in csvs:
        os.unlink(csv)

    return combined


def multi_run(arg, procs=1, dry_run=False, logfile=None):
    import subprocess as sp
    from multiprocessing import cpu_count

    d = yaml.load(open(arg, "r"))

    path    = d["data"]["path"]
    csv_out = d["data"]["csv_out"]

    ed = serialED.load(path)
    nfiles = ed.data.shape[0]

    print("Found {} files".format(nfiles))

    n = nfiles // procs + 1

    cores = cpu_count()

    assert procs >= 0, 'Expected a positive number of processors'
    if procs > cores:
        print('{} cores detected, are you sure you want to run {} processes? [y/n]'.format(cores, procs))
        if not input(' >> [y] ').lower() == 'y':
            sys.exit()

    processes = []

    t1 = time.time()

    print('Starting processes...')
    for i in range(procs):
        cmd = "problematic.index.exe {} -c {} {}".format(arg, i, n)
        if logfile:
            cmd += " --logfile {}".format(logfile)

        print("     >>", cmd, end=' ')

        if not dry_run:
            # CREATE_NEW_CONSOLE is windows only
            p = sp.Popen(cmd, creationflags=sp.CREATE_NEW_CONSOLE)
            processes.append(p)
            print(';; started (PID={})'.format(p.pid))
        else:
            print(';; not started')

    if dry_run:
        return
    
    from collections import deque
    d = deque(("|", "\\", "-", "/"))
    while any(p.poll() == None for p in processes):
        d.rotate()
        print("Running... {}".format(d[0]), end="\r")
        time.sleep(1)

    t2 = time.time()

    print("Time taken: {:.0f} s / {:.1f} s per image".format(t2-t1, (t2-t1)/nfiles))
    print()
    print(" >> Done << ")


def run(arg, chunk=None, dry_run=False, log=None):
    log = log or logging.getLogger(__name__)

    if len(sys.argv) == 1:
        print("Usage: problematic.index indexing.inp")
        print()
        print("Example input file:")
        print() 
        print(TEMPLATE)
        exit()

    d = yaml.load(open(arg, "r"))
    
    path     = d["data"]["path"]
    csv_out   = d["data"]["csv_out"]
    drc_out   = d["data"]["drc_out"]

    beam_center_sigma = 19

    if not "instructions" in d:
        d["instructions"] = {}

    method      = d["instructions"].get("method", "powell")
    radius      = d["instructions"].get("radius", 3)
    nsolutions  = d["instructions"].get("nsolutions", 25)
    filter1d    = d["instructions"].get("filter1d", False)
    nprojs      = d["instructions"].get("nprojs", 100)
    vary_scale  = d["instructions"].get("vary_scale", True)
    vary_center = d["instructions"].get("vary_center", True)

    d["instructions"]["method"]     = method
    d["instructions"]["radius"]     = radius
    d["instructions"]["nsolutions"] = nsolutions
    d["instructions"]["filter1d"]   = filter1d
    d["instructions"]["nprojs"]     = nprojs
    d["instructions"]["vary_scale"] = vary_scale
    d["instructions"]["vary_center"]= vary_center

    pixelsize = d["experiment"]["pixelsize"]
    dmin = d["projections"]["dmin"]
    dmax = d["projections"]["dmax"]
    thickness = d["projections"]["thickness"]
    
    if isinstance(d["cell"], (tuple, list)):
        pixelsize = d["experiment"]["pixelsize"]
        indexer = IndexerMulti.from_cells(d["cell"], pixelsize=pixelsize, **d["projections"])
    else:
        spgr = d["cell"]["spgr"]
        name = d["cell"]["name"]
        params = d["cell"]["params"]

        projector = Projector.from_parameters(params, spgr=spgr, name=name, dmin=dmin, dmax=dmax, thickness=thickness)
        indexer = Indexer.from_projector(projector, pixelsize=pixelsize)

    ed = serialED.load(path)
    centers = ed._centers

    if chunk:
        i, n = chunk
        offset = i*n
        print("Chunk #{}: from {} to {}".format(i, offset, offset+n))
        ed = ed.select(offset, offset+n)
        centers = centers.__class__(centers.data[offset:offset+n])

    nfiles = ed.data.shape[0]

    if not os.path.exists(drc_out):
        os.mkdir(drc_out)

    orientations = ed.find_orientations(indexer, centers)
    
    t0 = time.time()

    refined_orientations = ed.refine_orientations(indexer, orientations)
    
    t1 = time.time()

    intensities = ed.extract_intensities(orientations=refined_orientations, indexer=indexer, outdir=drc_out)

    ed.export_indexing_results()

    time_taken1 = t1 - t0
    time_taken2 = time.time() - t1
    msg1 = "Orientation finding:  {:.0f} s / {:.1f} s per image".format(time_taken1, (time_taken1)/nfiles)
    msg2 = "Orientation refining: {:.0f} s / {:.1f} s per image".format(time_taken2, (time_taken2)/nfiles)
    print()
    print(msg1)
    print(msg2)
    print()
    print(" >> DONE <<")

    log.info(projector._get_projection_alpha_beta_cache.cache_info())
    log.info(msg1)
    log.info(msg2)


def main():
    usage = """instamatic.index indexing.inp"""

    description = """
Program for indexing electron diffraction images.

""" 
    
    parser = argparse.ArgumentParser(#usage=usage,
                                    description=description,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("args", 
                        type=str, metavar="FILE", nargs="?",
                        help="Path to input file.")

    parser.add_argument("-j", "--procs", metavar='N',
                        action="store", type=int, dest="procs",
                        help="Number of processes to use (default = 1)")

    parser.add_argument("-c", "--chunk", metavar='N', nargs=2,
                        action="store", type=int, dest="chunk",
                        help="Used internally to specify the chunk number to process.")

    parser.add_argument("-d", "--dry",
                        action="store_true", dest="dry_run",
                        help="Runs the program, but doesn't start any processes.")

    parser.add_argument("-l", "--logfile",
                        action="store", type=str, dest="logfile",
                        help="Specify logfile (default=indexing.log).")

    parser.add_argument("-t", "--template",
                        action="store_true", dest="template",
                        help="Copy template notebook for indexing and exit.")

    parser.set_defaults(procs=1,
                        chunk=None,
                        dry_run=False,
                        resize=False,
                        template=None,
                        logfile="indexing.log"
                        )
    
    options = parser.parse_args()
    arg = options.args

    if options.template:
        copy_template()
        sys.exit()

    if not arg:
        parser.print_help()
        sys.exit()

    if options.procs > 1:
        multi_run(arg, procs=options.procs, dry_run=options.dry_run, logfile=options.logfile)
    else:
        logging.basicConfig(format="%(asctime)s | %(module)s:%(lineno)s | %(levelname)s | %(message)s", filename=options.logfile, level=logging.DEBUG)
        log = logging.getLogger(__name__)
        log.info("Start indexing")
        run(arg, options.chunk, log=log)


if __name__ == '__main__':
    main()
