'''
Created on 2013年8月19日

@author: epsilon
'''
import numpy as np
from numpy.polynomial import Polynomial

class TripleSpline(object):
    _coeffs=[
        [2 / 3.0, 0, -4, 4],
        [4 / 3.0, -4, 4, -4 / 3.0]]


    def __init__(self):
        self.func1=Polynomial(self._coeffs[0])
        self.func2=Polynomial(self._coeffs[1])
        self.func1_diff=self.func1.deriv()
        self.func2_diff=self.func2.deriv()
        self.diff_order=0
        
    def __call__(self,x,result=None):
        if x<0:
            raise ValueError('input value should be nonnegative, not '+str(x))
        if result is None:
            result=np.zeros((self._diff_order+1,),dtype=np.double)
        if x>=1:
            result.fill(0)
            return result
        if x<=0.5:
            result[0]=self.func1(x)
            if self.diff_order>=1:
                result[1]=self.func1_diff(x)
        else:
            result[0]=self.func2(x)
            if self.diff_order>=1:
                result[1]=self.func2_diff(x)
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