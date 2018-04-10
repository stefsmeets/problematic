#!/usr/bin/env python

from setuptools import setup, find_packages, Extension
from os import path
import sys

from Cython.Build import cythonize
import numpy as np

if sys.platform == "win32":
    extensions = [
        # Extension('problematic.radialprofile', ['src/radialprofile_cy.pyx'], include_dirs=[np.get_include()]),
        Extension('problematic.get_score_cy', ['src/get_score_cy.pyx'], include_dirs=[np.get_include()]),
        ]
else:
    extensions = [
        # Extension('problematic.radialprofile', ['src/radialprofile_cy.pyx'], include_dirs=[np.get_include()]),
        Extension('problematic.get_score_cy', ['src/get_score_cy.pyx'], include_dirs=[np.get_include()]),

    ]
ext_modules = cythonize(extensions)

setup(
    name="problematic",
    version="0.1.0",
    description="Program for data analysis of serial electron diffraction data",

    author="Stef Smeets",
    author_email="stef.smeets@mmk.su.se",
    license="GPL",
    url="https://github.com/stefsmeets/problematic",

    ext_modules = ext_modules,

    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],

    packages=["problematic",],

    install_requires=["numpy", "scipy", "pandas", "scikit-image", "pyyaml", "lmfit", "cython", "h5py"],

    package_data={
        "": ["LICENCE",  "readme.md", "setup.py"],
        "problematic.orientations": ["*"]
    },

    include_package_data=True,

    entry_points={
        'console_scripts': [
            'problematic.index   = problematic.indexer_app:main',
            'problematic.browser = problematic.browser:main',
        ]
    }

)

