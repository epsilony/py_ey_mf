'''

@author: <epsilonyuan@gmail.com>
'''

import sympy as sp
import numpy as np
from nose.tools import ok_
from itertools import repeat

import sys
if sys.version_info <= (3, 0):
    raise ValueError("only supports python3")
    
def gen_raw_samples():
    result = []
    for xi in np.linspace(0, 2.2, 5):
        for yj in np.linspace(0, 2.1, 7):
            for zk in np.linspace(0, 2.05, 3):
                result.append((xi, yj, zk))
    return result

def gen_sample_inf_rad():
    return 2

def raw_sample_to_distance_by_diff_samples(raw_sample):
    result = []
    for rs in raw_sample:
        dis = np.dot(rs, rs) ** 0.5
        if 0 == dis:
            result.append((0, 0, 0, 0))
        else:
            result.append((dis, rs[0] / dis, rs[1] / dis, rs[2] / dis))
    return result

def gen_wendland_3_2_exps(raw_samples, inf_rad):
    r = sp.symbols("r")
    x, y, z = sp.symbols("x y z")
    wend = (1 - r) ** 6 * (35 * r ** 2 + 18 * r + 3)
    d = sp.sqrt(x ** 2 + y ** 2 + z ** 2)
    R = sp.symbols("R")
    wend_exp = wend.subs(r, d / R)
    wend_exp_x = wend_exp.diff(x)
    wend_exp_y = wend_exp.diff(y)
    wend_exp_z = wend_exp.diff(z)
    
    wends = (wend_exp, wend_exp_x, wend_exp_y, wend_exp_z)
    
    exps = []
    
    for s in raw_samples:
        
        exp = []
        dis = np.dot(s, s) ** 0.5
        if dis > inf_rad:
            exps.append([0, 0, 0, 0])
            continue
        
        for w in wends:
            val = w.subs([(x, s[0]), (y, s[1]), (z, s[2]), (R, inf_rad)]).evalf()
            val = float(val)
            exp.append(val)
        if np.dot(s, s) == 0:  # magic
            exp[1:] = repeat(0, len(exp) - 1)
        exps.append(exp)
    
    return exps
    
def test_RadialBasis_by_wendland_3_2(err_limit=1e-6):
    raw_samples = gen_raw_samples()
    inf_rad = gen_sample_inf_rad()
    dist_samples = raw_sample_to_distance_by_diff_samples(raw_samples)
    exps = gen_wendland_3_2_exps(raw_samples, inf_rad)
    from ..radial_basis import RadialBasis
    rb = RadialBasis()
    rb.dim = 3;
    rb.diff_order = 1
    for raw, sample, exp in zip(raw_samples, dist_samples, exps):
        act = rb(sample, inf_rad)
        exp = np.asarray(exp)
        act = np.asarray(act)
        err = max(abs(exp - act))
        try:
            ok_(err < err_limit)
        except:
            ok_(err < err_limit, "raw sample = " + str(raw) + ", sample =" + str(sample) + ", act = " + str(act) + ", exp = " + str(exp))

