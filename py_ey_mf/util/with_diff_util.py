'''
Created on 2013-1-2

@author: Man YUAN <epsilonyuan@gmail.com>

'''
from scipy.misc import comb

def output_length(dim, diff_order):
    if dim == 1:
        return diff_order + 1
    elif dim == 2:
        return output_length_2d(diff_order)
    elif dim == 3:
        return output_length_3d(diff_order)
    else:
        raise ValueError("Only supports 1D, 2D and 3D")

def output_length_2d(diff_order):
    return (diff_order + 2) * (diff_order + 1) // 2

_3d_len_cache = [1, 4, 10, 20]
def output_length_3d(diff_order):
    if len(_3d_len_cache) > diff_order:
        return _3d_len_cache[diff_order]
    result = _3d_len_cache[-1]
    for i in range(len(_3d_len_cache), diff_order + 1):
        result += int(comb(2 + i, 2))
    return result
