#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import str
from builtins import map
from builtins import range
from builtins import object
from past.utils import old_div
import os
import sys
import re

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

ABSENT = -1
CENTRIC = 1

reflection_conditions = {
    "00l": lambda h, k, l: abs(h)+abs(k) == 0,
    "0k0": lambda h, k, l: abs(h)+abs(l) == 0,
    "h00": lambda h, k, l: abs(k)+abs(l) == 0,
    "0kl": lambda h, k, l: h == 0,
    "h0l": lambda h, k, l: k == 0,
    "hk0": lambda h, k, l: l == 0,
    "hkl": lambda h, k, l: np.ones_like(h),
    "hhl": lambda h, k, l: (h-k) == 0,
    "hkh": lambda h, k, l: (h-l) == 0,
    "hkk": lambda h, k, l: (k-l) == 0,
    "h-hl": lambda h, k, l: (h+k) == 0,
    "hk-h": lambda h, k, l: (h+l) == 0,
    "hk-k": lambda h, k, l: (k+l) == 0,
    "-2kkl": lambda h, k, l: (h == -2*k),
    "h-2hl": lambda h, k, l: (k == -2*h),
    "l=6n": lambda h, k, l: (l % 6) == 0,
    "l=4n": lambda h, k, l: (l % 4) == 0,
    "l=3n": lambda h, k, l: (l % 3) == 0,
    "l=2n": lambda h, k, l: (l % 2) == 0,
    "k=4n": lambda h, k, l: (k % 4) == 0,
    "k=2n": lambda h, k, l: (k % 2) == 0,
    "k+l=4n": lambda h, k, l: ((k+l) % 4) == 0,
    "k+l=2n": lambda h, k, l: ((k+l) % 2) == 0,
    "h=4n": lambda h, k, l: (h % 4) == 0,
    "h=2n": lambda h, k, l: (h % 2) == 0,
    "h-2k=4n": lambda h, k, l: ((h-2*k) % 4) == 0,
    "h+l=4n": lambda h, k, l: ((h+l) % 4) == 0,
    "h+l=2n": lambda h, k, l: ((h+l) % 2) == 0,
    "h+k=4n": lambda h, k, l: ((h+k) % 4) == 0,
    "h+k=2n": lambda h, k, l: ((h+k) % 2) == 0,
    "h+k+l=2n": lambda h, k, l: ((h+k+l) % 2) == 0,
    "h+2k=4n": lambda h, k, l: ((h+2*k) % 4) == 0,
    "2k-l=6n": lambda h, k, l: ((2*k-l) % 6) == 0,
    "2h-l=6n": lambda h, k, l: ((2*h-l) % 6) == 0,
    "2h-l=4n": lambda h, k, l: ((2*h-l) % 4) == 0,
    "2h+l=6n": lambda h, k, l: ((2*h+l) % 6) == 0,
    "2h+l=4n": lambda h, k, l: ((2*h+l) % 4) == 0,
    "2h+k=4n": lambda h, k, l: ((2*h+k) % 4) == 0,
    "-h+k+l=3n": lambda h, k, l: ((-h+k+l) % 3) == 0,
    "No Condition": lambda h, k, l: True
}


def asym_1(xxx_todo_changeme):
    (h,k,l) = xxx_todo_changeme
    ret = False
    if (h >= 0):
        ret = True

    if (k < 0) and (h == 0):
        ret = False
    if (l < 0) and (h == k == 0):
        ret = False
    return ret


def asym_2a(xxx_todo_changeme1):
    (h,k,l) = xxx_todo_changeme1
    ret = False
    if (h >= 0) and (l >= 0):
        ret = True
    
    if (l < 0) and (k == 0):
        ret = False
    return ret

def asym_2b(xxx_todo_changeme2):
    (h,k,l) = xxx_todo_changeme2
    ret = False
    if (k >= 0) and (l >= 0):
        ret = True

    if (h < 0) and (l == 0):
        ret = False
    return ret

def asym_2c(xxx_todo_changeme3):
    (h,k,l) = xxx_todo_changeme3
    ret = False
    if (k >= 0) and (l >= 0):
        ret = True

    if (h < 0) and (k == 0):
        ret = False
    return ret

def asym_3(xxx_todo_changeme4):
    (h,k,l) = xxx_todo_changeme4
    ret = False
    if (h >= 0) and (k >= 0) and (l >= 0):
        ret = True
    return ret

def asym_4a(xxx_todo_changeme5):
    (h,k,l) = xxx_todo_changeme5
    ret = False
    if (h >= 0) and (k >= 0) and (l >= 0):
        ret = True

    if (k > 0) and (h == 0):
        ret = False
    return ret

def asym_4b(xxx_todo_changeme6):
    (h,k,l) = xxx_todo_changeme6
    ret = False
    if (k >= h >= 0) and (l >= 0):
        ret = True
    return ret

def asym_5a(xxx_todo_changeme7):
    (h,k,l) = xxx_todo_changeme7
    ret = False
    if (k >= 0) and (l >= 0) and (h <= k) and (h <= l):
        ret = True

    if (h < k) and (h == l):
        ret = False
    if (l < 0) and (h ==0):
        ret = False
    return ret

def asym_5b(xxx_todo_changeme8):
    (h,k,l) = xxx_todo_changeme8
    ret = False
    if (h >= 0) and (k >= 0):
        ret = True

    if (l < 0) and (k ==0):
        ret = False
    if (l <= 0) and (h == 0):
        ret = False
    return ret

def asym_5c(xxx_todo_changeme9):
    (h,k,l) = xxx_todo_changeme9
    ret = False
    if (l >= k >= 0) and (k >= h):
        ret = True

    if (h < 0) and (abs(h) > l) and (l == 0):
        ret = True
    return ret

def asym_5d(xxx_todo_changeme10):
    (h,k,l) = xxx_todo_changeme10
    ret = False
    if (k >= h >= 0):
        ret = True

    if (l < 0) and (h == 0):
        ret = False
    return ret

def asym_5e(xxx_todo_changeme11):
    (h,k,l) = xxx_todo_changeme11
    ret = False
    if (h >= 0) and (k >= 0) and (l >= 0):
        ret = True
    
    if (h < k) and (l == 0):
        ret = False
    return ret

def asym_6a(xxx_todo_changeme12):
    (h,k,l) = xxx_todo_changeme12
    ret = False
    if (h >= 0) and (k >= 0) and (l >= 0):
        ret = True

    if (k > 0) and (h == 0):
        ret = False
    return ret

def asym_6b(xxx_todo_changeme13):
    (h,k,l) = xxx_todo_changeme13
    ret = False
    if (k >= h >= 0) and (l >= 0):
        ret = True
    return ret

def asym_7a(xxx_todo_changeme14):
    (h,k,l) = xxx_todo_changeme14
    ret = False
    if (k >= h >= 0) and (l >= h >= 0):
        ret = True

    if (h < k) and (h == l):
        ret = False
    return ret

def asym_7b(xxx_todo_changeme15):
    (h,k,l) = xxx_todo_changeme15
    ret = False
    if (l >= k >= h >= 0):
        ret = True
    return ret

# http://scripts.iucr.org/cgi-bin/paper?se0073
laue_asymmetry = {
    # Triclinic
    '-1': asym_1,  # 1/2
    # Monoclinic
    '2/m:a': asym_2a,  # 1/4
    '2/m:b': asym_2b,  # 1/4
    '2/m:c': asym_2c,  # 1/4
    # Orthorhombic
    'mmm': asym_3,  # 1/8
    # Tetragonal
    '4/m': asym_4a,  # 1/8
    '4/mmm': asym_4b,  # 1/16
    # Trigonal, Rhombohedral setting
    '-3:R': asym_5a,  # 1/6
    '-3m:R': asym_5b,  # 1/12
    # Trigonal, Hexagonal setting
    '-3:H': asym_5c,  # 1/6
    '-31m:H': asym_5d,  # 1/12
    '-3m1:H': asym_5e,  # 1/12
    # Hexagonal
    '6/m': asym_6a,  # 1/12
    '6/mmm': asym_6b,  # 1/24
    # Cubic
    'm-3': asym_7a,  # 1/24
    'm-3m': asym_7b  # 1/48
}

laue_fraction = {
    '-1': "1/2",
    '2/m:a': "1/4",
    '2/m:b': "1/4",
    '2/m:c': "1/4",
    'mmm': "1/8",
    '4/m': "1/8",
    '4/mmm': "1/16",
    '-3:R': "1/6",
    '-3m:R': "1/12",
    '-3:H': "1/6",
    '-31m:H': "1/12",
    '-3m1:H': "1/12",
    '6/m': "1/12",
    '6/mmm': "1/24",
    'm-3': "1/24",
    'm-3m': "1/48"
}

map_pointgroup2lauegroup = {
"C1" : ( "1"    , "-1"     ,  "Triclinic"),
"Ci" : ( "-1"   , "-1"     ,  "Triclinic"),
"C2" : ( "2"    , "2/m"    ,  "Monoclinic"),
"Cs" : ( "m"    , "2/m"    ,  "Monoclinic"),
"C2h": ( "2/m"  , "2/m"    ,  "Monoclinic"),
"C2v": ( "mm2"  , "mmm"    ,  "Orthorhombic"),
"D2" : ( "222"  , "mmm"    ,  "Orthorhombic"),
"D2h": ( "mmm"  , "mmm"    ,  "Orthorhombic"),
"C3" : ( "3"    , "-3"     ,  "Trigonal"),
"C3i": ( "-3"   , "-3"     ,  "Trigonal"),
"C3v": ( "3m"   , "-3m"    ,  "Trigonal"),
"D3" : ( "32"   , "-3m"    ,  "Trigonal"),
"D3d": ( "-3m"  , "-3m"    ,  "Trigonal"),
"C4" : ( "4"    , "4/m"    ,  "Tetragonal"),
"S4" : ( "-4"   , "4/m"    ,  "Tetragonal"),
"C4h": ( "4/m"  , "4/m"    ,  "Tetragonal"),
"C4v": ( "4mm"  , "4/mmm"  ,  "Tetragonal"),
"D2d": ( "-42m" , "4/mmm"  ,  "Tetragonal"),
"D4" : ( "422"  , "4/mmm"  ,  "Tetragonal"),
"D4h": ( "4/mmm", "4/mmm"  ,  "Tetragonal"),
"C6" : ( "6"    , "6/m"    ,  "Hexagonal"),
"C3h": ( "-6"   , "6/m"    ,  "Hexagonal"),
"C6h": ( "6/m"  , "6/m"    ,  "Hexagonal"),
"D6" : ( "622"  , "6/mmm"  ,  "Hexagonal"),
"C6v": ( "6mm"  , "6/mmm"  ,  "Hexagonal"),
"D6h": ( "6/mmm", "6/mmm"  ,  "Hexagonal"),
"D3h": ( "-6m2" , "6/mmm"  ,  "Hexagonal"),
"T"  : ( "23"   , "m3"     ,  "Cubic"),
"Th" : ( "m3"   , "m3"     ,  "Cubic"),
"Td" : ( "-43m" , "m-3m"   ,  "Cubic"),
"O"  : ( "432"  , "m-3m"   ,  "Cubic"),
"Oh" : ( "m-3m" , "m-3m"   ,  "Cubic") }


def find_hall(s, spacegrouptxt='spacegroups.txt'):
    print(os.path.join(os.path.dirname(__file__), spacegrouptxt))
    f = open(os.path.join(os.path.dirname(__file__), spacegrouptxt), 'r')
    spacegrouptxt = f.readlines()
    f.close()
    s = str(s).replace(" ", "").lower()
    for i, line in enumerate(spacegrouptxt):
        m = re.search(" "+s+"[: ]+", line)
        if m:
            hall = line[55:]
            return hall
    raise ValueError("Cannot find space group {}".format(s))


def get_random_cell(spgr):
    import random
    a = old_div(float(random.randrange(500, 2000)), 100)
    b = old_div(float(random.randrange(500, 2000)), 100)
    c = old_div(float(random.randrange(500, 2000)), 100)
    al = float(random.randrange(60, 120))
    be = float(random.randrange(60, 120))
    ga = float(random.randrange(60, 120))

    system = spgr.crystal_system
    lauegr = spgr.laue_group
    setting = spgr.setting

    if system == "Triclinic":
        return (a, b, c, al, be, ga)
    elif system == "Monoclinic":
        return (a, b, c, be)
    elif system == "Orthorhombic":
        return (a, b, c)
    elif system == "Tetragonal":
        return (a, c)
    elif system == "Trigonal":
        if setting == "R":
            return (a, al)
        else:
            return (a, c)
    elif system == "Hexagonal":
        return (a, c)
    elif system == "Cubic":
        return (a,)
    else:
        raise ValueError("Invalid system {}".format(system))


def symm2str(r,t=np.zeros([3, 1], dtype=float)):
    xyz = "xyz"
    string_all = []
    for i in range(3):
        string_row = ""
        for j in range(3):
            rval = r[i,j]
            if rval == -1:
                string_row += "-{}".format(xyz[j])
            if rval == 1:
                if string_row != "":
                    string_row += "+"
                string_row += "{}".format(xyz[j])
        tval = t[i,0]
        if tval%1 == 0.5:
            tval = "+1/2"
        elif tval%1 == 0.25:
            tval = "+1/4"
        elif tval%1 == 0.75:
            tval = "+3/4"
        elif tval%1 == 1.0/3.0:
            tval = "+1/3"
        elif tval%1 == 2.0/3.0:
            tval = "+2/3"
        elif tval%1 == 1.0/6.0:
            tval = "+1/6"
        elif tval%1 == 5.0/6.0:
            tval = "+5/6"
        else:
            tval = ""
        string_row += "{}".format(tval)
        if string_row == "":
            string_row = "0"
        string_all.append(string_row)
    return ", ".join(string_all)


def str2symm(s):
    """Parses symmetry strings and returns a rotation matrix and translation vector."""
    rotmat = np.zeros([3, 3], dtype=int)
    transvec = np.zeros([3, 1], dtype=float)
    split = s.split(',')
    i = 0       # determines the row in the matrix
    for symeq in split:
        # separate x,y,z,+,-,float,int,fraction
        q = re.findall('[\+\-xXyYzZ]|[0-9]*[\.\/][0-9]+|[0-9]', symeq)
        coefficient = 1
        for r in q:
            if r == '-':
                coefficient = -1
            elif r == '+':
                coefficient = 1
            elif r.lower() == 'x':
                rotmat[i, 0] += coefficient
            elif r.lower() == 'y':
                rotmat[i, 1] += coefficient
            elif r.lower() == 'z':
                rotmat[i, 2] += coefficient
            elif '/' in r:          # check for fraction
                frac = re.findall('[0-9]+', r)
                num = float(frac[0])
                denom = float(frac[1])
                transvec[i, 0] = num/denom
            else:
                try:                # check for float or integer
                    r = float(r)
                    transvec[i, 0] = coefficient*r
                except:
                    pass
        i += 1
    return rotmat, transvec


class SymOp(object):

    """Generate symmetry operations from string"""

    def __init__(self, Mx):
        super(SymOp, self).__init__()
        self.Mx = Mx

        self._r, self._t = str2symm(self.Mx)

    def __repr__(self):
        return "{}".format(self.Mx)

    def __get__(self):
        return self.r, self.t

    @property
    def r(self):
        return self._r

    @property
    def t(self):
        return self._t

    @classmethod
    def from_str(cls, s):
        return cls(s)

    @classmethod
    def from_mat(cls, r, t):
        return cls(symm2str(r, t))

    def getsymm(self, op=None):
        return str2symm(op)

    def inverse(self):
        r = np.dot(self.r, -np.eye(3,3))
        t = self.t
        return SymOp(symm2str(r,t))


class ReflCond(object):

    """Generate reflection conditions from string
    Uses lookup table to find corresponding operations"""

    def __init__(self, s):
        super(ReflCond, self).__init__()
        self._s = s
        self._ref, self._cond = self.build_reflection_condition(self._s)

    def __repr__(self):
        return self._s

    def build_reflection_condition(self, string=None):
        if not string:
            string = self._s

        ref, cond = string.split(":")
        ref = reflection_conditions[ref.strip()]
        cond = reflection_conditions[cond.strip()]

        return ref, cond

    def test(self, index):
        # return self._ref(*index) and self._cond(*index)
        return self._ref(*index) & self._cond(*index)

    def is_absent(self, index):
        # return self._ref(*index) and not self._cond(*index)
        return self._ref(*index) & ~self._cond(*index)


class SpaceGroup(object):

    """SpaceGroup class that takes a spgr dict (SpaceGroup)
    Stores all space group info that could be extracted from sginfo"""

    def __init__(self, symbol):
        super(SpaceGroup, self).__init__()

        from parse_sginfo import parse

        dct = parse(symbol)

        self._dct = dct

        self.number = dct["number"]
        self.hall = dct["hall"]
        self.spgr_name = self.hermann_mauguin = dct["hm"].replace(" ", "")
        self.schoenflies = dct["schoenfliess"]
        self.qualif = dct["qualif"]   # R/H
        self.setting = dct["setting"] # a/b/c

        self.isChiral = dct["chiral"]
        self.isEnantiomorphic = dct["enantiomorphic"]
        # self.isCentric = dct["centrosymmetric"]  # defined by Hall symbol

        self.order = dct["nsym"]
        self.order_p = dct["nsymp"]
        # self.n_symop = dct["nsymp"]/2 if self.isCentric else dct["nsym"]

        self.point_group, self.laue_group, self.crystal_system = map_pointgroup2lauegroup[self.schoenflies.split("^")[0]]

        if self.crystal_system == "Monoclinic":
            self.unique_axis = {"a":"x","b":"y", "c":"z"}[self.setting.replace("-", "")[0]]
        else:
            self.unique_axis = ""

        self._symops = dct["symops"]
        self._cvecs = dct["centering_vectors"]

        # self.reflection_conditions = [
        #     ReflCond(rc) for rc in kwargs["reflection_conditions"]]


    def __repr__(self):
        return "SpaceGroup('{}')".format(self.spgr_name)

    @property
    def space_group(self):
        if self.setting:
            return "{}:{}".format(self.number, self.setting)
        elif self.qualif:
            return "{}:{}".format(self.number, self.qualif)
        else:
            return "{}".format(self.number)

    @property
    def centering_symbol(self):
        return self.hermann_mauguin[0]

    def _symmetry_operations(self, p=False, verbose=False):
        """Returns a generator with symmetry operations"""
        for i, op in enumerate(self._symops):
            if verbose and i % self.order_p == 0:
                print("# +({})".format(self._cvecs[i // self.order_p]))
            if p and (i == self.order_p):
                break
            yield SymOp(op)
    
    @property
    def symmetry_operations(self):
        """Returns a generator with symmetry operations"""
        for item in self._symmetry_operations():
            yield item

    @property
    def symmetry_operations_p(self):
        """Returns a generator with primitive symmetry operations
        (excluding centering vectors)"""
        for item in self._symmetry_operations(p=True):
            yield item

    @property
    def laue_symmetry_operations(self):
        """Returns a generator with laue symmetry operations"""
        from laue_symops import symops
        lauegr = self.laue_group

        if self.setting:
            lauegr += f":{self.setting[0]}"

        return (SymOp.from_str(op) for op in symops[lauegr])

    @property
    def centering_vectors(self):
        """Returns generator with centering vectors"""
        for item in self._symmetry_operations[::self.order_p]:
            yield item.t

    @property
    def isCentrosymmetric(self):
        return self.hall[0] == "-"

    def sginfo(self):
        print("Space group", self.spgr_name)
        print()
        print("    Number      ", self.space_group)
        print("    Schoenflies ", self.schoenflies)
        print("    Hall        ", self.hall)
        if self.setting:
            print("    H-M symbol   {}:{}".format(self.hermann_mauguin, self.setting))
        else:
            print("    H-M symbol  ", self.hermann_mauguin)
        print()
        print("Laue  group ", self.laue_group)
        print("Point group ", self.point_group)

        print(self.crystal_system)
        if self.isCentrosymmetric:
            print("Centrosymmetric")
        if self.isChiral:
            print("Chiral")
        if self.isEnantiomorphic:
            print("Enantiomorphic")

        if self.unique_axis:
            print("Unique axis", self.unique_axis)
        print()
        print("Order    ", self.order)
        print("Order P  ", self.order_p)
        print()

        # print "\nSymmetry operations"
        for symop in self._symmetry_operations(verbose=True):
            print(symop)

        # print "\nReflection conditions"
        # for rc in self.reflection_conditions:
        #     print rc

        # print "\nWyckoff positions (experimental, standard setting only!)"
        # for wyck in self.wyckoff_positions:
        #     print wyck

    def standardize(self, df, key=None):
        return standardize_indices(df, self, key=key)

    def completeness(self, df, dmin=None):
        return completeness(df, self, dmin=dmin)
    
    def standardize_indices(self, arr):
        """
        Standardizes reflection indices
        From Siena Computing School 2005, Reciprocal Space Tutorial (G. Sheldrick)
        http://www.iucr.org/resources/commissions/crystallographic-computing/schools
            /siena-2005-crystallographic-computing-school/speakers-notes

        Input:
            arr: (N, 3) np.array,
                reflection list as numpy array
        Returns:
            (N, 3) np.array: merged reflection list as numpy array

        """
        stacked_symops = np.stack([s.r for s in self.symmetry_operations_p])
        
        m = np.dot(arr, stacked_symops).astype(int)
        m = np.hstack([m, -m])
        i = np.lexsort(m.transpose((2,0,1)))
        merged =  m[np.arange(len(m)), i[:,-1]] # there must be a better way to index this, but this works and is quite fast
    
        return merged

    def _apply_along_index(self, arr, func):
        """Expects tuple/list of 3 elements or iterable

        return 1/0"""
        if isinstance(arr, pd.Index):
            return arr.map(func)
        elif isinstance(arr, pd.DataFrame):
            return arr.index.map(func)
        elif len(arr[0]) == 3:
            # assume nested list
            return np.array(list(map(func, arr)))
        else:
            return func(arr)


def test_print_all():
    """Parse and print all space groups"""
    f = open(os.path.join(os.path.dirname(__file__), 'spacegroups.txt'), 'r')
    spacegrouptxt = f.readlines()
    f.close()
    for i, row in enumerate(spacegrouptxt):
        print("====="*16)
        number = row[:12].strip()
        spgr = SpaceGroup(number)
        spgr.info()


def generate_hkl_listing(cell, dmin=1.0, dmax=np.inf, as_type=None, expand=False):
    """Generate hkllisting up to the specified dspacing.

    cell: instance of unitcell.UnitCell
    expand: bool
        expand to P1 symmetry
    dmin: float
        minimum d-spacing cut-off value
    as_type: str
        pd.Index, pd.DataFrame, np.array (default)

    Based on the routine described by:
    Le Page and Gabe, J. Appl. Cryst. 1979, 12, 464-466
    """

    ax = np.argmax([cell.a, cell.b, cell.c])
    idx = [0,0,0]
    hmax = 1
    idx[ax] = hmax
    while cell._calc_dspacing(idx) > dmin:
        idx[ax] = hmax
        # print(idx, cell._calc_dspacing(idx))
        hmax += 1

    indices = np.array(cell.generate_hkl(hmax, expand=expand))
    d = np.array(cell.calc_dspacing(indices))
    
    sel = (d < dmax) & (d > dmin)
    indices = indices[sel]
    d = d[sel]

    i = d.argsort()[::-1]
    d = d[i].reshape(-1, 1)
    indices = indices[i]

    if as_type == "pd.Index":
        return pd.Index([tuple(idx) for idx in indices])
    if as_type == "pd.DataFrame":
        return pd.DataFrame(index=[tuple(idx) for idx in indices])
    if as_type == "np.array":
        return np.array(indices)
    else:
        return indices


def get_laue_symops(key):
    """take laue group and return list of symmetry operators (instances of SymOp)"""
    from .laue_symops import symops
    return (SymOp.from_str(op) for op in symops[key])


def get_merge_dct(cell, dmin=1.0):
    raise RuntimeError("New merging algorithm no longer uses get_merge_dct, will be removed soon. Use cell.merge(df) instead.")


def expand_to_p1(arr, cell):
    """Expand hkl indices in (M, 3) array to P1"""
    stacked_symops = np.stack([s.r for s in cell.laue_symmetry_operations]) # laue symops

    expanded = np.vstack(np.dot(arr, stacked_symops).astype(int))

    hkl_p1 = np.vstack({tuple(row) for row in expanded})
    # this also works (about 15x faster), but is a bit wtf (http://stackoverflow.com/a/16973510)
    # hkl_p1 = np.unique(expanded.view(np.dtype((np.void, expanded.dtype.itemsize*expanded.shape[1])))).view(expanded.dtype).reshape(-1, expanded.shape[1])
    
    return hkl_p1


def standardize_indices(df, cell, key=None):
    """
    Standardizes reflection indices
    From Siena Computing School 2005, Reciprocal Space Tutorial (G. Sheldrick)
    http://www.iucr.org/resources/commissions/crystallographic-computing/schools
        /siena-2005-crystallographic-computing-school/speakers-notes
    """
    stacked_symops = np.stack([s.r for s in cell.laue_symmetry_operations])
    
    if not key:
        m = np.dot(np.array(df.index.tolist()), stacked_symops).astype(int)
    else:
        m = np.dot(df[key], stacked_symops).astype(int)
    m = np.hstack([m, -m])
    i = np.lexsort(m.transpose((2,0,1)))
    merged =  m[np.arange(len(m)), i[:,-1]] # there must be a better way to index this, but this works and is quite fast

    df["h"] = merged[:,0]
    df["k"] = merged[:,1]
    df["l"] = merged[:,2]

    return df


def merge(df, cell, remove_sysabs=True, key="F"):
    """
    Merges equivalent reflections
    From Siena Computing School 2005, Reciprocal Space Tutorial (G. Sheldrick)
    http://www.iucr.org/resources/commissions/crystallographic-computing/schools
        /siena-2005-crystallographic-computing-school/speakers-notes

    Merges 400k reflections in about 0.8 seconds
    """

    df = standardize_indices(df, cell)

    if not "esd" in df:
        df["esd"] = 1.0

    # print len(set(map(tuple, pd.Index(df.index))))

    hkl = ["h","k","l"]

    gb = df.groupby(hkl)
    df["sqrt"] = old_div(1,df.esd**2)
    merged = gb.agg({key:(np.mean)})

    merged["esd"] = old_div(1,np.sqrt(gb.agg({"sqrt":np.sum})["sqrt"]))

    if remove_sysabs:
        # merged["flag"] = merged.index.map(lambda x: is_centric_or_absent(x, stacked_symops, transops))
        # ncentric = np.sum(merged["flag"] == CENTRIC)
        # nabsent  = np.sum(merged["flag"] == ABSENT)
        # nmerged  = np.sum(merged["flag"] != ABSENT)
        # merged = merged[merged.flag != ABSENT]
        # print " >> Merged {} to {} reflections (centric: {}, absent: {})".format(len(df), nmerged, ncentric, nabsent)

        merged["flag"] = cell.is_absent(merged.index)
        nabsent = merged["flag"].sum()
        merged = merged[merged["flag"] == False]
        print(" >> Merged {} to {} reflections (absent: {})".format(len(df), len(merged), nabsent))
    else:
        print(" >> Merged {} to {} reflections".format(len(df), len(merged)))

    return merged


def completeness(df, cell, dmin=None):
    if 'd' not in df:
        df['d'] = df.index.map(cell.calc_dspacing)
    if not dmin:
        dmin = df['d'].min()

    unique_set = generate_hkl_listing(cell, dmin=dmin, as_type='pd.Index')
    unique_set = cell.filter_systematic_absences(unique_set)

    len_a = (df['d'] > dmin).sum()
    len_b = len(unique_set)

    return old_div(len_a, float(len_b))


if __name__ == '__main__':
    import sys
    for arg in sys.argv[1:]:
        print()
