'''

@author: <epsilonyuan@gmail.com>
'''
import numpy as np
from ..radial.radial_basis import RadialBasis
from ..monomial.monomial_bases import MonomialBases
from ..util.with_diff_util import output_length

class _MLSCache(object):
    _indial_caches_size = 6
    
    def __init__(self):
        self._bases_size = -1
        self._nodes_size = -1
        self._B_caches_map = {}
        self._weight_caches = [np.ndarray((i+1,), dtype=np.double) for i in range(0, self._indial_caches_size)]
    
    def set_dim(self,dim):
        self.dim=dim
        self._dim_cache=np.ndarray((dim,),dtype=np.double)
        self._distance_cache=np.ndarray((dim+1,),dtype=np.double)
    
    def get_dim_size_cache(self):
        return self._dim_cache
    
    def get_dist_cache(self):
        return self._distance_cache
    
    def get_weight_cache(self, diff_size):
        return self._weight_caches[diff_size - 1]
        
    def get_gamma_cache(self,index):
        return self._gamma_caches[index]
    
    def get_cache_size(self):
        return self._indial_caches_size
    
    def get_mat_A_cache(self, index):
        return self._A_caches[index]
    
    def get_mat_B_cache(self, index):
        return self._B_caches[index]
    
    def get_bases_cache(self,diff_size):
        return self._bases_caches[diff_size-1]

    def set_bases_size(self, bases_size):
        if bases_size < 1:
            raise ValueError("bases_size should be positive, not" + str(bases_size))
        if self._bases_size == bases_size:
            return
        else:
            self._bases_size = bases_size
            self._A_caches = self._new_A_caches()
            self._B_caches = None
            self._B_caches_map.clear()
            if self._nodes_size >= 1:
                self._set_nodes_size(self._nodes_size)
            self._gamma_caches=self._new_gamma_caches()
            self._bases_caches=self._new_bases_caches()

    def set_nodes_size(self, nodes_size):
        if nodes_size < 1:
            raise ValueError("nodes_size should be positive, not" + str(nodes_size))
        if self._nodes_size == nodes_size and self._B_caches is not None:
            return
        self._nodes_size = nodes_size
        if nodes_size in self._B_caches_map:
            self._B_caches = self._B_caches_map[nodes_size]
        else:
            self._B_caches = self._new_B_caches()
            self._B_caches_map[nodes_size] = self._B_caches
            
    def _new_B_caches(self):
        b_caches = [np.ndarray((self._bases_size, self._nodes_size), dtype=np.double) for _i in range(self._indial_caches_size)]
        return b_caches
    
    def _new_A_caches(self):
        a_caches = [np.ndarray((self._bases_size, self._bases_size), dtype=np.double) for _i in range(self._indial_caches_size)]
        return a_caches
    
    def _new_gamma_caches(self):
        return [np.ndarray((self._bases_size,),dtype=np.double) for _i in range(self._indial_caches_size)]
    
    def _new_bases_caches(self):
        return [np.ndarray((i+1,self._bases_size),dtype=np.double) for i in range(self._indial_caches_size)]
    
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
        
        self.dim = dim
        
        self.ZERO=np.zeros((dim,),dtype=np.double)
        
        self._cache = _MLSCache()
        self._cache.set_dim(dim)
        self._cache.set_bases_size(self.get_bases_size())
        
        self._diff_order=-1
        self.diff_order=0
        
    def set_diff_order(self, diff_order):
        if diff_order < 0 or diff_order > 1:
            raise ValueError("only support _diff_order 0/1, not " + str(diff_order))
        if self._diff_order==diff_order:
            return
        self._diff_order=diff_order
        self.bases.diff_order=diff_order
        self.weight_func.diff_order=diff_order
        diff_size = self.get_diff_size()
        self.mat_As = [self._cache.get_mat_A_cache(i) for i in range(diff_size)]
        self.mat_Bs = [None for _i in range(diff_size)]
    
    def get_diff_order(self):
        return self._diff_order
    
    diff_order = property(get_diff_order, set_diff_order)
    
    def set_dim(self, dim):
        if dim < 1 or dim > 3:
            raise ValueError("only support dim 1/2/3, not " + str(dim))
        self._dim = dim
        self.weight_func.set_dim(dim)
        self.bases.set_dim(dim)
    
    def get_dim(self):
        return self._dim
    
    dim = property(get_dim, set_dim)
    
    def _set_nodes_size(self, nodes_size):
        self._cache.set_nodes_size(nodes_size)
        self._nodes_size=nodes_size
    
    def _get_dist_by_diff(self, node, node_index):
        if self._dists is not None:
            return self._dists[:,node_index]
        else:
            dst=self._cache.get_dist_cache()
            tv=self._cache.get_dim_size_cache()
            np.copyto(tv,self._pos)
            tv-=node.coord
            l=np.dot(tv,tv)**0.5
            dst[0]=l
            if self.diff_order>=1:
                dst[1:]=tv
                dst[1:]/=l
        return dst
            
            
    def get_diff_size(self):
        return output_length(self.dim, self.diff_order)
    
    def get_bases_size(self):
        return self.bases.results_size()[1]
    
    def __call__(self, pos, nodes, dists=None, result=None):
        self._dists = dists
        self._nodes = nodes
        self._pos = pos
        
        self.calcMatAB()
        
        diff_size = self.get_diff_size()
        nodes_size = len(nodes)
        if result is None:
            result = np.ndarray((diff_size, nodes_size), dtype=np.double)
        gamma = self._cache.get_gamma_cache(0)
        bases_by_diff = self._cache.get_bases_cache(diff_size)
        self.bases(self.ZERO, bases_by_diff)
        self.solve(self.mat_As[0], bases_by_diff[0], gamma)
        np.dot(gamma, self.mat_Bs[0], result[0])
        
        if self.diff_order >= 1:
            tv = self._cache.get_gamma_cache(diff_size)
            tv2 = self._cache.get_mat_B_cache(diff_size)[0]
            gamma_i = self._cache.get_gamma_cache(1)
            for i in range(1, diff_size):
                np.dot(self.mat_As[i], gamma, tv)
                tv *= -1
                tv += bases_by_diff[i]
                self.solve(self.mat_As[0], tv, gamma_i)
                      
                np.dot(gamma_i, self.mat_Bs[0], tv2)
                np.dot(gamma, self.mat_Bs[i], result[i])
                result[i] += tv2
        return result

    def calcMatAB(self):
        node_index = 0
        self.initMatAB()
        diff_size=self.get_diff_size()
        weight_by_diff = self._cache.get_weight_cache(diff_size)
        bases_by_diff = self._cache.get_bases_cache(diff_size)
        for nd in self._nodes:
            dst = self._get_dist_by_diff(nd,node_index)
            inf_rad = nd.influence_radius
            self.weight_func(dst, inf_rad, weight_by_diff)
            self.bases(nd.coord-self._pos, bases_by_diff)
            self.pushToMatA(weight_by_diff, bases_by_diff)
            self.pushToMatB(weight_by_diff, bases_by_diff, node_index)
            node_index += 1
    
    def initMatAB(self):
        self._cache.set_nodes_size(len(self._nodes))
        for i in range(len(self.mat_Bs)):
            self.mat_Bs[i] = self._cache.get_mat_B_cache(i)
            self.mat_Bs[i].fill(0)
            self.mat_As[i].fill(0)
            
    def pushToMatA(self, weight_by_diff, bases_by_diff):
        diff_size = self.get_diff_size()
        bases_size = self.bases.results_size()[1]
        t_mat = self._cache.get_mat_A_cache(diff_size)
        for i in range(diff_size):
            np.multiply(bases_by_diff[0].reshape((bases_size, 1)), bases_by_diff[0], t_mat)
            t_mat *= weight_by_diff[i]
            mat_A_i = self.mat_As[i]
            mat_A_i += t_mat
    
    def pushToMatB(self, weight_by_diff, bases_by_diff, node_index):
        for i in range(self.get_diff_size()):
            mat_B_i = self.mat_Bs[i]
            mat_B_i[:, node_index] = bases_by_diff[0]
            mat_B_i[:, node_index] *= weight_by_diff[i]
    
    def solve(self,mat,b,result):
        result[:]=np.linalg.solve(mat,b)
