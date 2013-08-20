'''

@author: <epsilonyuan@gmail.com>
'''
import numpy as np
from ..radial.radial_basis import RadialBasis
from ..monomial.monomial_bases import MonomialBases
from ..util.with_diff_util import output_length

class _MLSCache(object):
    _indcial_caches_size = 6  # must be strictly greater than the diff_size, for 3d, if only supporting first order, the diff_size is 4 
    
    def __init__(self):
        self._bases_size = -1
        self._nodes_size = -1
        self._diff_order=-1
        self._dim = -1
        self._B_caches_map = {}
        self._weight_caches = [np.ndarray((i + 1,), dtype=np.double) for i in range(0, self._indcial_caches_size)]
    
    def setup(self, diff_order, dim, bases_size, nodes_size):
        if diff_order <0 or diff_order >1:
            raise ValueError('diff_order must be 0-1, not'+str(diff_order))
        if dim < 1 or dim > 3:
            raise ValueError('dim must be 1-3, not ' + str(dim))
        if bases_size < 1:
            raise ValueError('bases_size must be positive, not' + str(bases_size))
        if nodes_size < 1:
            raise ValueError('nodes_size must be positive, not' + str(nodes_size))
        
        if self._diff_order!=diff_order or self._dim!=dim:
            self._diff_order=diff_order
            self._diff_size=output_length(dim,diff_order)
        if self._dim != dim:
            self._dim = dim
            self._dim_size_cache = np.ndarray((dim,), dtype=np.double)
            self._distance_cache = np.ndarray((dim + 1,), dtype=np.double)
        if self._bases_size != bases_size:
            self._bases_size = bases_size
            self._A_caches = self._new_A_caches()
            
            self._gamma_caches = self._new_gamma_caches()
            self._bases_caches = self._new_bases_caches()
            
            self._B_caches = None
            self._B_caches_map.clear()
            self._nodes_size = -1
        if self._nodes_size != nodes_size:
            self._nodes_size = nodes_size
            if nodes_size in self._B_caches_map:
                self._B_caches = self._B_caches_map[nodes_size]
            else:
                self._B_caches = self._new_B_caches()
                self._B_caches_map[nodes_size] = self._B_caches

    
    def get_dim_size_cache(self):
        return self._dim_size_cache
    
    def get_dist_cache(self):
        return self._distance_cache
    
    def get_weight_cache(self):
        return self._weight_caches[self._diff_size - 1]
        
    def get_gamma_cache(self, index):
        return self._gamma_caches[index]
    
    def get_cache_size(self):
        return self._indcial_caches_size
    
    def get_mat_As_cache(self):
        return self._A_caches
    
    def get_mat_Bs_cache(self):
        return self._B_caches
    
    def get_bases_cache(self):
        return self._bases_caches[self._diff_size - 1]
            
    def _new_B_caches(self):
        b_caches = [np.ndarray((self._bases_size, self._nodes_size), dtype=np.double) for _i in range(self._indcial_caches_size)]
        return b_caches
    
    def _new_A_caches(self):
        a_caches = [np.ndarray((self._bases_size, self._bases_size), dtype=np.double) for _i in range(self._indcial_caches_size)]
        return a_caches
    
    def _new_gamma_caches(self):
        return [np.ndarray((self._bases_size,), dtype=np.double) for _i in range(self._indcial_caches_size)]
    
    def _new_bases_caches(self):
        return [np.ndarray((i + 1, self._bases_size), dtype=np.double) for i in range(self._indcial_caches_size)]
    
class MLS(object):
    def __init__(self, dim=2, weight_func=None, bases=None):
        if weight_func is None:
            self.weight_func = RadialBasis()
        else:
            self.weight_func = weight_func
        
        if bases is None:
            self.bases = MonomialBases()
        else:
            self.bases = bases
        
        self._cache = _MLSCache()        
        
        self._dim = -1
        self.dim=2
        self._diff_order = -1
        self.diff_order=0
        
    def set_diff_order(self, diff_order):
        if diff_order < 0 or diff_order > 1:
            raise ValueError("only support _diff_order 0/1, not " + str(diff_order))
        if self._diff_order == diff_order:
            return
        self._diff_order = diff_order
        self.weight_func.diff_order=diff_order
    
    def get_diff_order(self):
        return self._diff_order
    
    diff_order = property(get_diff_order, set_diff_order)
    
    def set_dim(self, dim):
        if dim < 1 or dim > 3:
            raise ValueError("only support _dim 1/2/3, not " + str(dim))
        if self._dim == dim:
            return
        self._dim=dim
        self.bases.dim=dim
        self.weight_func.dim=dim
        self.ZERO = np.zeros((dim,), dtype=np.double)
    
    def get_dim(self):
        return self._dim
    
    dim = property(get_dim, set_dim)
                    
    def get_diff_size(self):
        return output_length(self._dim, self.diff_order)
    
    def get_bases_size(self):
        return self.bases.results_size()[1]
    
    def __call__(self, pos, nodes, dists=None, result=None):
        nodes_size = len(nodes)
        self._cache.setup(self._diff_order, self._dim,self.get_bases_size(),nodes_size)
        
        self._dists = dists
        self._nodes = nodes
        self._pos = pos
        
        self._calcMatAB()
        
        self.bases.diff_order=self._diff_order
        diff_size = self.get_diff_size()

        if result is None:
            result = np.ndarray((diff_size, nodes_size), dtype=np.double)
        
        gamma = self._cache.get_gamma_cache(0)
        bases_by_diff = self._cache.get_bases_cache()
        self.bases(self.ZERO, bases_by_diff)
        self._solve(self._mat_As[0], bases_by_diff[0], gamma)
        np.dot(gamma, self._mat_Bs[0], result[0])
        
        if self.diff_order >= 1:
            tv = self._cache.get_gamma_cache(diff_size)
            tv2 = self._mat_Bs[diff_size][0]
            gamma_i = self._cache.get_gamma_cache(1)
            for i in range(1, diff_size):
                np.dot(self._mat_As[i], gamma, tv)
                tv *= -1
                tv += bases_by_diff[i]
                self._solve(self._mat_As[0], tv, gamma_i)
                      
                np.dot(gamma_i, self._mat_Bs[0], tv2)
                np.dot(gamma, self._mat_Bs[i], result[i])
                result[i] += tv2
        return result

    def _get_dist_by_diff(self, node, node_index):
        if self._dists is not None:
            return self._dists[:, node_index]
        else:
            dst = self._cache.get_dist_cache()
            tv = self._cache.get_dim_size_cache()
            np.copyto(tv, self._pos)
            tv -= node.coord
            l = np.dot(tv, tv) ** 0.5
            dst[0] = l
            if self.diff_order >= 1:
                dst[1:] = tv
                dst[1:] /= l
        return dst

    def _calcMatAB(self):
        node_index = 0
        self._init_mat_AB()
        weight_by_diff = self._cache.get_weight_cache()
        self.bases.set_diff_order(0)
        bases_by_diff = self._cache.get_bases_cache()
        for nd in self._nodes:
            dst = self._get_dist_by_diff(nd, node_index)
            inf_rad = nd.influence_radius
            self.weight_func(dst, inf_rad, weight_by_diff)
            self.bases(nd.coord - self._pos, bases_by_diff)
            self._pushToMatA(weight_by_diff, bases_by_diff)
            self._pushToMatB(weight_by_diff, bases_by_diff, node_index)
            node_index += 1
    
    def _init_mat_AB(self):
        self._mat_Bs=self._cache.get_mat_Bs_cache()
        self._mat_As=self._cache.get_mat_As_cache()
        for i in range(self.get_diff_size()):
            self._mat_Bs[i].fill(0)
            self._mat_As[i].fill(0)
            
    def _pushToMatA(self, weight_by_diff, bases_by_diff):
        diff_size = self.get_diff_size()
        bases_size = self.bases.results_size()[1]
        t_mat = self._mat_As[diff_size]
        for i in range(diff_size):
            np.multiply(bases_by_diff[0].reshape((bases_size, 1)), bases_by_diff[0], t_mat)
            t_mat *= weight_by_diff[i]
            mat_A_i = self._mat_As[i]
            mat_A_i += t_mat
    
    def _pushToMatB(self, weight_by_diff, bases_by_diff, node_index):
        for i in range(self.get_diff_size()):
            mat_B_i = self._mat_Bs[i]
            mat_B_i[:, node_index] = bases_by_diff[0]
            mat_B_i[:, node_index] *= weight_by_diff[i]
    
    def _solve(self, mat, b, result):
        result[:] = np.linalg.solve(mat, b)
