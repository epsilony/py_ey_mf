'''

@author: epsilonyuan@gmail.com
'''
# -*- coding: utf-8 -*-
'''

@author: <epsilonyuan@gmail.com>
'''

import sympy
from numpy.polynomial import Polynomial
import numpy as np

def _gen_wendlands_as_symbols():
    r = sympy.symbols("r")
    wendlands = {
    "wendland_3_1" : (1 - r) ** 4 * (4 * r + 1),
    "wendland_3_2" : (1 - r) ** 6 * (35 * r ** 2 + 18 * r + 3),
    "wendland_3_3" : (1 - r) ** 8 * (32 * r ** 3 + 25 * r ** 2 + 8 * r + 1)
    }
    return wendlands

def _symbol_to_poly(s):
    coefs=s.as_poly().all_coeffs()
    coefs.reverse()
    return Polynomial(coefs)
        
_continuity_name_map={2:'wendland_3_1',4:'wendland_3_2',6:'wendland_3_3'}

class Wendland(object):
    def __init__(self,continuity=None):
        if continuity is None:
            continuity = 4
        if continuity not in _continuity_name_map:
            raise ValueError("continuity must be in "+str(_continuity_name_map))
        key=_continuity_name_map[continuity]
        wendlands=_gen_wendlands_as_symbols()
        w=wendlands[key]
        w_1=w.diff('r')
        w_2=w_1.diff('r')
        ws=(w,w_1,w_2)
        self._polys=[]
        for s in ws:
            self._polys.append(_symbol_to_poly(s))
        self.set_diff_order(0)
    
    def __call__(self,x,result=None):
        if x<0:
            raise ValueError('input value should be nonnegative, not '+str(x))
        if result is None:
            result=np.zeros((self._diff_order+1,),dtype=np.double)
        if x>=1:
            result.fill(0)
            return result
        for i in range(self._diff_order+1):
            result[i]=self._polys[i](x)
        return result
    
    def values(self,x,result):
        return self.__call__(x, result)
    
    def set_diff_order(self, diff_order):
        if diff_order < 0 or diff_order > 1:
            raise ValueError("only support _diff_order 0/1, not " + str(diff_order))
        self._diff_order = diff_order
    
    def get_diff_order(self):
        return self._diff_order
    
    diff_order = property(get_diff_order, set_diff_order)
