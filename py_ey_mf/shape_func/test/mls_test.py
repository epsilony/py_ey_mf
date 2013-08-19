'''

@author: <epsilonyuan@gmail.com>
'''

import numpy as np
from nose.tools import ok_,nottest
from py_ey_mf.shape_func.mls import MLS
from py_ey_mf.radial.core.triple_spline import TripleSpline

class tMFNode(object):
    pass

def gen_regular_nodes_samples_2d():
    coords=[]
    for x in np.linspace(-1,5,6):
        for y in np.linspace(2,6,6):
            coords.append(np.array((x,y),dtype=np.double))
    infs=np.linspace(3,3.5,len(coords))
    result=[]
    for c,r in zip(coords,infs):
        node=tMFNode()
        node.coord=c
        node.influence_radius=r
        result.append(node)
    return result

def gen_xs_samples_2d():
    xs=[]
    for x in np.linspace(2,3,2):
        for y in np.linspace(3,5,3):
            xs.append(np.array((x,y),dtype=np.double))
    return xs

@nottest
def gen_test_data_2d():
    nodes=gen_regular_nodes_samples_2d()
    xs=gen_xs_samples_2d()
    return {'nodes':nodes,'xs':xs}

def test_partition_unity():
    error_lim=1e-10
    test_data=gen_test_data_2d()
    mls=MLS()
    #mls.weight_func.core=TripleSpline()
    mls.diff_order=0
    xs=test_data['xs']
    nodes=test_data['nodes']
    for x in xs:
        act_shape_func=mls(x,nodes)
        act_sum=np.sum(act_shape_func[0])
        ok_(abs(act_sum)-1<error_lim)
    
    mls.diff_order=1
    for x in xs:
        act_shape_func=mls(x,nodes)
        act_sum=np.sum(act_shape_func[0])
        ok_(abs(act_sum)-1<error_lim)
        for i in range(1,len(act_shape_func)):
            act_sum=np.sum(act_shape_func[i])
            ok_(abs(act_sum)<error_lim)

def gen_polygnomial():
    return [
            lambda x,y: 1.3-2.7*x+3.3*y,#+0.2*x**2+0.3*x*y-0.4*y**2,
            lambda x,y: -2.7,#+0.4*x+0.3*y,
            lambda x,y: 3.3#+0.3*x+0.8*y
            ]

def test_polynomial_represent():
    err_limit=1e-9
    poly=gen_polygnomial()
    test_data=gen_test_data_2d()
    xs=test_data['xs']
    nodes=test_data['nodes']
    for nd in nodes:
        nd.poly_value=np.array([poly[i](*nd.coord) for i in range(len(poly))],dtype=np.double)
    mls=MLS()
    #mls.weight_func.core=TripleSpline()
    mls.diff_order=1
    for x in xs:
        result=mls(x,nodes)
        exp=np.array([poly[i](*x) for i in range(len(poly))])
        act=np.zeros((3,),dtype=np.double)
        
        for i in range(len(nodes)):
            nd=nodes[i]
            act+=result[:,i]*nd.poly_value[0]
        
        err=exp-act
        err=np.dot(err,err)**0.5
        ok_(err<err_limit)

def nodes_to_MFNode(nodes,gw):
    MFNode=gw.jvm.net.epsilony.mf.geomodel.MFNode
    result=gw.jvm.java.util.ArrayList()
    for nd in nodes:
        mfnd=MFNode()
        coords=gw.new_array(gw.jvm.double,len(nd.coord))
        for i in range(len(nd.coord)):
            coords[i]=nd.coord[i]
        mfnd.setCoord(coords)
        mfnd.setInfluenceRadius(nd.influence_radius)
        result.append(mfnd)
    return result
        
if __name__=="__main__":
    from py4j.java_gateway import JavaGateway
    gw=JavaGateway()
    test_data=gen_test_data_2d()
    nodes=test_data['nodes']
    mf_nodes=nodes_to_MFNode(nodes,gw)
    
    j_xy=gw.new_array(gw.jvm.double,2)
    j_xy[0]=2
    j_xy[1]=3
    JMLS=gw.jvm.net.epsilony.mf.shape_func.MLS
    jmls=JMLS()
    jmls.setDiffOrder(1)
    j_results=jmls.values(j_xy,mf_nodes,None)
    result=np.ndarray((len(j_results),len(mf_nodes)),dtype=np.double)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i][j]=j_results[i].get(j)
    mls=MLS()
    mls.weight_func.core=TripleSpline()