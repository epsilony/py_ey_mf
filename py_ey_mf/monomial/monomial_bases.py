'''

@author: <epsilonyuan@gmail.com>
'''

from ..util.with_diff_util import output_length
import numpy as np

def _1d_ori(xs,degree,result=None):
    if degree<0 or degree >3:
        raise ValueError("only support degree 0/1/2/3, not "+str(degree))
    res_len=output_length(1,degree)
    if result is None:
        result=np.zeros((res_len,),dtype=np.double)
    else:
        result.fill(0)
    result[0]=1
    if degree>=1:
        (x,)=xs
        result[1]=x
    if degree >=2:
        x2=x*x
        result[2]=x2
    if degree >=3:
        x3=x2*x
        result[3]=x3
    return result

def _1d_x(xs,degree,result=None):
    if degree<0 or degree >3:
        raise ValueError("only support degree 0/1/2/3, not "+str(degree))
    res_len=output_length(1,degree)
    if result is None:
        result=np.zeros((res_len,),dtype=np.double)
    else:
        result.fill(0)
    result[0]=0
    if degree>=1:
        result[1]=1
    if degree >=2:
        (x,)=xs
        result[2]=2*x
    if degree >=3:
        x2=x*x
        result[3]=3*x2
    return result

def _2d_ori(xy,degree,result=None):
    if degree<0 or degree >3:
        raise ValueError("only support degree 0/1/2/3, not "+str(degree))
    res_len=output_length(2,degree)
    if result is None:
        result=np.zeros((res_len,),dtype=np.double)
    else:
        result.fill(0)
    result[0]=1
    if degree>=1:
        x,y=xy
        result[1]=x
        result[2]=y
    if degree >=2:
        x2=x*x
        y2=y*y
        result[3]=x2
        result[4]=x*y
        result[5]=y2
    if degree >=3:
        x3=x2*x
        y3=y2*y
        result[6]=x3
        result[7]=x2*y
        result[8]=x*y2
        result[9]=y3
    return result

def _2d_x(xy,degree,result=None):
    if degree<0 or degree >3:
        raise ValueError("only support degree 0/1/2/3, not "+str(degree))
    res_len=output_length(2,degree)
    if result is None:
        result=np.zeros((res_len,),dtype=np.double)
    else:
        result.fill(0)
    result[0]=0
    if degree>=1:
        result[1]=1
        result[2]=0
    if degree >=2:
        x,y=xy
        result[3]=2*x
        result[4]=y
        result[5]=0
    if degree >=3:
        x2=x*x
        y2=y*y
        result[6]=3*x2
        result[7]=2*x*y
        result[8]=y2
        result[9]=0
    return result

def _2d_y(xy,degree,result=None):
    if degree<0 or degree >3:
        raise ValueError("only support degree 0/1/2/3, not "+str(degree))
    res_len=output_length(2,degree)
    if result is None:
        result=np.zeros((res_len,),dtype=np.double)
    else:
        result.fill(0)
    result[0]=0
    if degree>=1:
        result[1]=0
        result[2]=1
    if degree >=2:
        x,y=xy
        result[3]=0
        result[4]=x
        result[5]=y*2
    if degree >=3:        
        x2=x*x
        y2=y*y
        result[6]=0
        result[7]=x2
        result[8]=2*x*y
        result[9]=3*y2
    return result

def _3d_ori(xyz,degree,result=None):
    if degree<0 or degree >3:
        raise ValueError("only support degree 0/1/2/3, not "+str(degree))
    res_len=output_length(3,degree)
    if result is None:
        result=np.zeros((res_len,),dtype=np.double)
    else:
        result.fill(0)
    result[0]=1
    if degree>=1:
        x,y,z=xyz
        result[1]=x
        result[2]=y
        result[3]=z
    if degree>=2:
        x2=x*x
        y2=y*y
        z2=z*z
        result[4]=x2
        result[5]=x*y
        result[6]=y2
        result[7]=x*z
        result[8]=y*z
        result[9]=z2
    if degree>=3:
        x3=x2*x
        y3=y2*y
        z3=z2*z
        result[10]=x3
        result[11]=x2*y
        result[12]=x*y2
        result[13]=y3
        result[14]=x2*z
        result[15]=x*y*z
        result[16]=y2*z
        result[17]=x*z2
        result[18]=y*z2
        result[19]=z3
    return result
 
def _3d_x(xyz,degree,result=None):
    if degree<0 or degree >3:
        raise ValueError("only support degree 0/1/2/3, not "+str(degree))
    res_len=output_length(3,degree)
    if result is None:
        result=np.zeros((res_len,),dtype=np.double)
    else:
        result.fill(0)
    result[0]=0
    if degree>=1:
        result[1]=1
        result[2]=0
        result[3]=0
    if degree>=2:
        x,y,z=xyz
        result[4]=2*x
        result[5]=y
        result[6]=0
        result[7]=z
        result[8]=0
        result[9]=0
    if degree>=3:
        x2=x*x
        y2=y*y
        z2=z*z
        result[10]=3*x2
        result[11]=2*x*y
        result[12]=y2
        result[13]=0
        result[14]=2*x*z
        result[15]=y*z
        result[16]=0
        result[17]=z2
        result[18]=0
        result[19]=0
    return result

def _3d_y(xyz,degree,result=None):
    if degree<0 or degree >3:
        raise ValueError("only support degree 0/1/2/3, not "+str(degree))
    res_len=output_length(3,degree)
    if result is None:
        result=np.zeros((res_len,),dtype=np.double)
    else:
        result.fill(0)
    result[0]=0
    if degree>=1:
        result[1]=0
        result[2]=1
        result[3]=0
    if degree>=2:
        x,y,z=xyz
        result[4]=0
        result[5]=x
        result[6]=y*2
        result[7]=0
        result[8]=z
        result[9]=0
    if degree>=3:
        x2=x*x
        y2=y*y
        z2=z*z
        result[10]=0
        result[11]=x2
        result[12]=2*x*y
        result[13]=3*y2
        result[14]=0
        result[15]=x*z
        result[16]=2*y*z
        result[17]=0
        result[18]=z2
        result[19]=0
    return result

def _3d_z(xyz,degree,result=None):
    if degree<0 or degree >3:
        raise ValueError("only support degree 0/1/2/3, not "+str(degree))
    res_len=output_length(3,degree)
    if result is None:
        result=np.zeros((res_len,),dtype=np.double)
    else:
        result.fill(0)
    result[0]=0
    if degree>=1:
        result[1]=0
        result[2]=0
        result[3]=1
    if degree>=2:
        x,y,z=xyz
        result[4]=0
        result[5]=0
        result[6]=0
        result[7]=x
        result[8]=y
        result[9]=2*z
    if degree>=3:
        x2=x*x
        y2=y*y
        z2=z*z
        result[10]=0
        result[11]=0
        result[12]=0
        result[13]=0
        result[14]=x2
        result[15]=x*y
        result[16]=y2
        result[17]=x*z*2
        result[18]=y*z*2
        result[19]=3*z2
    return result
    
_dim_ori_map={1:(_1d_ori,),2:(_2d_ori,),3:(_3d_ori,)}

_dim_diff_map={1:(_1d_ori,_1d_x),2:(_2d_ori,_2d_x,_2d_y),3:(_3d_ori,_3d_x,_3d_y,_3d_z)}

_diff_dim_funcs_map={0:_dim_ori_map,1:_dim_diff_map}

class MonomialBases(object):
    def __init__(self):
        self._dim=2
        self._monomial_degree=2
        self._diff_order=0
        self._choose_funcs()

    
    def _choose_funcs(self):
        self.funcs=_diff_dim_funcs_map[self._diff_order][self._dim]
    
    def set_diff_order(self,diff_order):
        if diff_order<0 or diff_order > 1:
            raise ValueError("only support _diff_order 0/1, not "+str(diff_order))
        if self._diff_order!=diff_order:
            self._diff_order=diff_order
            self._choose_funcs()
    
    def get_diff_order(self):
        return self._diff_order
    
    diff_order=property(get_diff_order,set_diff_order)
        
    def set_dim(self,dim):
        if dim < 1 or dim >3:
            raise ValueError("only support dim 1/2/3, not "+str(dim))
        if self._dim!=dim:
            self._dim=dim
            self._choose_funcs()
    
    def get_dim(self):
        return self._dim
    
    dim=property(get_dim,set_dim)
    
    def set_monomial_degree(self,degree):
        if degree < 1 or degree >3:
            raise ValueError("only support degree 1/2/3, not "+str(degree))
        if self._monomial_degree!=degree:
            self._monomial_degree=degree
            self._choose_funcs()
    
    def get_monomial_degree(self):
        return self._monomial_degree
    
    monomial_degree=property(get_monomial_degree,set_monomial_degree)
    
    def results_size(self):
        num_cols=output_length(self._dim,self._monomial_degree)
        num_rows=len(self.funcs)
        return (num_rows,num_cols)
    
    def __call__(self,pos,result=None):
        if result is None:
            num_rows,num_cols=self.results_size()
            result=np.ndarray((num_rows,num_cols),dtype=np.double)
        for func,row in zip(self.funcs,result):
            if row is None:
                raise ValueError("None row, wrong results container")
            func(pos,self._monomial_degree,row)
        return result
            
        