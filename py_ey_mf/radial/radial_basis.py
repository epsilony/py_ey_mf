'''

@author: <epsilonyuan@gmail.com>
'''

from ..util.with_diff_util import output_length
import numpy as np

class RadialBasis(object):
    '''
    classdocs
    '''


    def __init__(self, core=None):
        '''
        Constructor
        '''
        if core is None:
            from .core.wendland import Wendland
            core = Wendland()
        self.core = core
        self.set_diff_order(0)
        self.set_dim(1)
        
    def __call__(self, distance_by_diff, influence_rad, result=None):
        output_len = output_length(self.dim, self.diff_order)
        if result is None:
            result = np.ndarray((output_len,), dtype=np.double)
        delta = distance_by_diff[0] / influence_rad
        self.core(delta, result)
        if self.diff_order >= 1 :
            core_d = result[1]
            rad_reci = 1 / influence_rad
            for i in range(1, output_len):
                result[i] = core_d * rad_reci * distance_by_diff[i]
        return result
    
    def set_diff_order(self, diff_order):
        if diff_order < 0 or diff_order > 1:
            raise ValueError("only support _diff_order 0/1, not " + str(diff_order))
        self.core.set_diff_order(diff_order)
    
    def get_diff_order(self):
        return self.core.diff_order
    
    diff_order = property(get_diff_order, set_diff_order)
    
    def set_dim(self, dim):
        if dim < 1 or dim > 3:
            raise ValueError("only support dim 1/2/3, not " + str(dim))
        self._dim = dim
    
    def get_dim(self):
        return self._dim
    
    dim = property(get_dim, set_dim)
    
