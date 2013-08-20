'''

@author: <epsilonyuan@gmail.com>
'''

import sympy
import numpy as np
from ..monomial_bases import MonomialBases
from nose.tools import ok_,nottest
import json
from io import StringIO

from ...util.json_util import NumpyEncoder

def gen_symbols_set():
    x, y, z = sympy.symbols('x,y,z')
    one = x + 1
    one -= x
    ori = [
             [one, x, x ** 2, x ** 3],
             [one, x, y, x ** 2, x * y, y ** 2, x ** 3, x ** 2 * y, x * y ** 2, y ** 3],
             [one, x, y, z, x ** 2, x * y, y ** 2, x * z, y * z, z ** 2, x ** 3, x ** 2 * y, x * y ** 2, y ** 3, x ** 2 * z, x * y * z, y ** 2 * z, x * z ** 2, y * z ** 2, z ** 3]
             ]

    x_diff = [ori[0]]
    t = []
    for s in ori[0]:
        t.append(s.diff(x))
    x_diff.append(t)
    
    xy_diff = [ori[1]]
    
    for d in (x, y):
        t = []
        for s in ori[1]:
            t.append(s.diff(d))
        xy_diff.append(t)
        
    xyz_diff = [ori[2]]
    for d in (x, y, z):
        t = []
        for s in ori[2]:
            t.append(s.diff(d))
        xyz_diff.append(t)
        
    return [
              {'funcs':x_diff, 'diff_symbols':[x]},
              {'funcs':xy_diff, 'diff_symbols':[x, y]},
              {'funcs':xyz_diff, 'diff_symbols':[x, y, z]}]

def gen_samples_set():
    x_sample = []
    for x in np.linspace(-1, 12.5, 3):
        x_sample.append((x,))
        
    xy_sample = []
    for x in np.linspace(0.5, 6.2, 5):
        for y in np.linspace(-0.7, 1.3, 3):
            xy_sample.append((x, y))
            
    xyz_sample = []
    
    for x in np.linspace(-1.1, 0.79, 3):
        for y in np.linspace(0.2, 7.9, 4):
            for z in np.linspace(-0.2, 7.3, 3):
                xyz_sample.append((x, y, z))
    
    return (x_sample, xy_sample, xyz_sample)

def gen_exps(symbols, samples):
    exps = []

    funcs = symbols['funcs']
    diff_symbols = symbols['diff_symbols']
    num_rows = len(funcs)
    num_cols = len(funcs[0])
    for sp in samples:
        subs_list = list(zip(diff_symbols, sp))
        exp = np.ndarray((num_rows, num_cols), dtype=np.double)
        for i in range(num_rows):
            for j in range(num_cols):
                exp[i][j] = funcs[i][j].subs(subs_list).evalf()
        exps.append(exp)
    return exps

@nottest
def gen_test_datas():
    samples_set = gen_samples_set()
    symbols_set = gen_symbols_set()
    test_data = []
    for symbols, samples in zip(symbols_set, samples_set):
        exps = gen_exps(symbols, samples)
        test_data.append({'samples':samples,
                          'exps':exps,
                          'dimension':len(symbols['diff_symbols'])
                          }
                         )
    return test_data
            

@nottest
def gen_test_datas_json_string():
    test_datas=gen_test_datas()
    sio=StringIO()
    json.dump(test_datas,sio,cls=NumpyEncoder)
    return sio.getvalue()

@nottest
def gen_test_datas_json_file(file_name):
    json_str=gen_test_datas_json_string()
    with open(file_name,'w') as f:
        f.write(json_str)

def exam_monomial(test_data):
    err_limit = 1e-6
    mb = MonomialBases()
    mb.diff_order = 1
    mb._dim = test_data['dimension']
    mb.monomial_degree = 3
    for smp, exp in zip(test_data['samples'], test_data['exps']):   
        try:
            act = mb(smp)
            ok_(abs(act - exp).max() < err_limit)
        except:
            act = mb(smp)
            ok_(abs(act - exp).max() < err_limit)
    
def test_monomial():
    test_datas = gen_test_datas()
    for test_data in test_datas:
        exam_monomial(test_data)
