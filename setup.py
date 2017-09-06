#!/usr/bin/env python

from setuptools import setup, find_packages
from os import path

# www.pythonhosted.org/setuptools/setuptools.html

setup(
    name="problematic",
    version="0.1.0",
    description="Program for data analysis of serial electron diffraction data",

    author="Stef Smeets",
    author_email="stef.smeets@mmk.su.se",
    license="GPL",
    url="https://github.com/stefsmeets/problematic",

    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],

    packages=["problematic",],

    install_requires=["numpy", "scikit-image", "pyyaml", "lmfit"],

    package_data={
        "": ["LICENCE",  "readme.md", "setup.py"],
    },

)

