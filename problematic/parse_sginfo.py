#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from past.utils import old_div

import subprocess as sp
import re

exe = "sginfo"

def get_transvec(string):
    transvec = []
    for i, val in enumerate(string.strip('()').split()):
        if '/' in val:
            frac = val.split('/')
            num = float(frac[0])
            denom = float(frac[1])
            transvec.append(old_div(num,denom))
        else:
            transvec.append(float(val))
    return transvec

get_centering = re.compile("\((.*)\)")

def parse(spgr, verbose=False):
    cmd = [exe, spgr, '-allxyz']
    p = sp.Popen(cmd, stdout=sp.PIPE)
    out, err = p.communicate()

    out = out.decode()

    symops = []
    centering_vecs = []
    centrosymm = False
    save_symops = False
    uniq_axis = None
    chiral_spgr = False
    enantiomorphic = False
    off_origin = False
    obverse = False

    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Space Group"):
            inp = line.split("  ")
            number = inp[1]
            schoenfliess = inp[2]
            conventional = hm = inp[3]
            hall = inp[4]
            qualif = setting = ""
            if ":" in number:
                number, postfix = number.split(":")
                if postfix in "RH":
                    qualif = postfix
                else:
                    setting = postfix
            if "=" in hm:
                conventional = hm.split(" = ")[-1]
        elif line.startswith('Point Group'):
            pgr = line.split()[-1]
        elif line.startswith('Laue  Group'):
            lauegr = line.split()[-1]
        elif line.startswith('Order   '):
            nsym = int(line.split()[-1])
        elif line.startswith('Order P '):
            nsymp = int(line.split()[-1])
        elif line.startswith('Unique'):
            uniq_axis = line.split()[-1]
        elif line.startswith("Chiral space group"):
            chiral_spgr = True
        elif line.startswith("Enantiomorphic"):
            enantiomorphic = True
        elif line.startswith("Obverse"):
            obverse = True
        elif line.startswith("Note: Inversion operation off origin"):
            off_origin = True

        m = re.search(get_centering, line)
        if m:
            centering = get_transvec(m.group())
            if centering not in centering_vecs:
                centering_vecs.append(centering)
        if "Inversion-Flag = 1" in line:
            centrosymm = True

        if m or line.startswith('x, y, z'):
            save_symops = True
        # if save_symops and line.startswith('#'):
            # save_symops = False
        if save_symops and not line:
            save_symops = False

        if save_symops and not line.startswith('#'):
            symops.append(line)

    cmd = [exe, spgr, '-Conditions']
    p = sp.Popen(cmd, stdout=sp.PIPE)
    out, err = p.communicate()

    out = out.decode()

    reflection_conditions = []
    phase_restrictions = []
    enhanced_reflections = []
    save_refl_cond = False
    save_refl_phase = False
    save_refl_enh = False
    for line in out.splitlines():
        if line.startswith("Reflection conditions"):
            save_refl_cond = True
            continue
        if line.startswith("Reflections with phase restriction"):
            save_refl_phase = True
            continue
        if line.startswith("Systematically enhanced reflections"):
            save_refl_enh = True
            continue
        
        if not line.split():
            save_refl_cond = False
            save_refl_phase = False
            save_refl_enh = False
        
        if save_refl_phase:
            phase_restrictions.append(repr(line.strip()))
        if save_refl_cond:
            reflection_conditions.append(repr(line.strip()))
        if save_refl_enh:
            if len(line.split("=")[-1]) <=4:
                enhanced_reflections.append(repr(line.strip()))
            elif verbose:
                print("skip", line)

    if len(phase_restrictions) == 0:
        phase_restrictions = ["'hkl: No Condition'"]

    if not centering_vecs:
        centering_vecs = [[0.0, 0.0, 0.0]]
    centering_vecs = [repr(vec) for vec in centering_vecs]

    symops = [repr(symop) for symop in symops]

    d = {
    "number": int(number),
    "setting": setting,
    "qualif": qualif,
    "schoenfliess": schoenfliess,
    "hall": hall,
    "hm": hm,
    "spgr_name": conventional,
    "pgr": pgr,
    "lauegr": lauegr,
    "nsym": nsym,
    "nsymp": nsymp,
    "uniq_axis": uniq_axis,
    "centrosymmetric": centrosymm,
    "enantiomorphic": enantiomorphic,
    "chiral": chiral_spgr,
    "obverse": obverse,
    "reflection_conditions": reflection_conditions,
    "enhanced_reflections": enhanced_reflections,
    "phase_restrictions": phase_restrictions,
    "centering_vectors": centering_vecs,
    "symops": symops}

    return d


if __name__ == '__main__':
    import sys
    for sg in sys.argv[1:]:
        for k,v in parse(sg).items():
            print(k, v)

