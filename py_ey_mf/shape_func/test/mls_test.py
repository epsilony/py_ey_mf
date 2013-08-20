'''

@author: <epsilonyuan@gmail.com>
'''

import numpy as np
from nose.tools import ok_, nottest
from py_ey_mf.shape_func.mls import MLS
from py_ey_mf.radial.core.triple_spline import TripleSpline

class tMFNode(object):
    def __init__(self, coord=None, influence_radius=None):
        self.coord = coord
        self.influence_radius = influence_radius

@nottest
def gen_test_datas():
    
    nd_crds_1d = []
    for x in np.linspace(-1, 10, 10):
        nd_crds_1d.append(np.array((x,), dtype=np.double))
    infs_1d = np.linspace(3, 3.5, len(nd_crds_1d))
    test_pts_1d = []
    for x in np.linspace(1, 9, 7):
        test_pts_1d.append(np.array((x,), dtype=np.double))
    polys_1d = [lambda x:-3.3 + 4 * x - 2 * x ** 2,
              lambda x:4 - 4 * x
              ]
    
    nd_crds_2d = []
    for x in np.linspace(-1, 5, 6):
        for y in np.linspace(2, 6, 6):
            nd_crds_2d.append(np.array((x, y), dtype=np.double))
    infs_2d = np.linspace(3, 3.5, len(nd_crds_2d))
    test_pts_2d = []
    for x in np.linspace(0, 4, 3):
        for y in np.linspace(1, 5, 3):
            test_pts_2d.append(np.array((x, y), dtype=np.double))
    polys_2d = [
            lambda x, y: 1.3 - 2.7 * x + 3.3 * y + 0.2 * x ** 2 + 0.3 * x * y - 0.4 * y ** 2,
            lambda x, y:-2.7 + 0.4 * x + 0.3 * y,
            lambda x, y: 3.3 + 0.3 * x - 0.8 * y
            ]
    
    nd_crds_3d = []
    for x in np.linspace(-1, 5, 6):
        for y in np.linspace(2, 6, 6):
            for z in np.linspace(-4, 0, 6):
                nd_crds_3d.append(np.array((x, y, z), dtype=np.double))
    infs_3d = np.linspace(3, 3.5, len(nd_crds_3d))
    test_pts_3d = []
    for x in np.linspace(0, 4, 3):
        for y in np.linspace(1, 5, 3):
            for z in np.linspace(-3, -1, 3):
                test_pts_3d.append(np.array((x, y, z), dtype=np.double))
    polys_3d = [
              lambda x, y, z:1.1 - 2.1 * x + 3 * y + 0.4 * z - x ** 2 + 0.8 * x * y + 0.3 * y ** 2 - x * z + 0.2 * y * z + 0.7 * z ** 2,
              lambda x, y, z:-2.1 - 2 * x + 0.8 * y - z,
              lambda x, y, z:3 + 0.8 * x + 0.6 * y + 0.2 * z,
              lambda x, y, z:0.4 - x + 0.2 * y + 1.4 * z
              ]
    
    nd_crds_xd = (nd_crds_1d, nd_crds_2d, nd_crds_3d)
    infs_xd = (infs_1d, infs_2d, infs_3d)
    test_pts_xd = (test_pts_1d, test_pts_2d, test_pts_3d)
    polys_xd = (polys_1d, polys_2d, polys_3d)
    
    result = []
    for dim, nd_crds, infs, test_pts, polys in zip(range(1, 4), nd_crds_xd, infs_xd, test_pts_xd, polys_xd):
        nodes = []
        for crd, inf in zip(nd_crds, infs):
            nd = tMFNode(crd, inf)
            nodes.append(nd)
        result.append({'dim':dim, 'nodes':nodes, 'test_pts':test_pts, 'polys':polys})
    return result

def gen_regular_nodes_samples_2d():
    coords = []
    for x in np.linspace(-1, 5, 6):
        for y in np.linspace(2, 6, 6):
            coords.append(np.array((x, y), dtype=np.double))
    infs = np.linspace(3, 3.5, len(coords))
    result = []
    for c, r in zip(coords, infs):
        node = tMFNode()
        node.coord = c
        node.influence_radius = r
        result.append(node)
    return result

def test_partition_unity():
    test_datas = gen_test_datas()
    mls = MLS()
    for test_data in test_datas:
        _test_partition_unity(test_data, mls)
        
@nottest
def _test_partition_unity(test_data, mls):
    error_lim = 1e-11

    mls.dim = test_data['dim']
    mls.weight_func.core = TripleSpline()
    mls.diff_order = 0
    test_pts = test_data['test_pts']
    nodes = test_data['nodes']
    for x in test_pts:
        nds = _filet_nodes(x, nodes)
        act_shape_func = mls(x, nds)
        act_sum = np.sum(act_shape_func[0])
        ok_(abs(act_sum) - 1 < error_lim)
    
    mls.diff_order = 1
    for x in test_pts:
        nds = _filet_nodes(x, nodes)
        act_shape_func = mls(x, nds)
        act_sum = np.sum(act_shape_func[0])
        ok_(abs(act_sum) - 1 < error_lim)
        for i in range(1, len(act_shape_func)):
            act_sum = np.sum(act_shape_func[i])
            ok_(abs(act_sum) < error_lim, str({"dim":mls.dim, "diff_order":mls.diff_order, "test_pt":x, "act_sum":act_sum}))

def _filet_nodes(x, nodes):
    result = []
    for nd in nodes:
        cr = nd.coord
        if np.dot(x - cr, x - cr) ** 0.5 < nd.influence_radius:
            result.append(nd)
    return result

def test_polynomial_represent():
    test_datas = gen_test_datas()
    mls = MLS()
    for test_data in test_datas:
        _test_polynomial_represent(test_data, mls)

@nottest
def _test_polynomial_represent(test_data, mls):
    err_limit = 1e-11
    xs = test_data['test_pts']
    nodes = test_data['nodes']
    polys = test_data['polys']
    for nd in nodes:
        nd.poly_value = np.array([polys[i](*nd.coord) for i in range(len(polys))], dtype=np.double)
    mls.dim = test_data['dim']
    mls.weight_func.core = TripleSpline()
    mls.diff_order = 1
    for x in xs:
        nds = _filet_nodes(x, nodes)
        result = mls(x, nds)
        exp = np.array([polys[i](*x) for i in range(len(polys))])
        act = np.zeros((test_data['dim'] + 1,), dtype=np.double)
        
        for i in range(len(nds)):
            nd = nds[i]
            act += result[:, i] * nd.poly_value[0]
        
        err = exp - act
        err = np.dot(err, err) ** 0.5
        try:
            ok_(err < err_limit, str({'dim':mls.dim, 'err':err}))
        except:
            ok_(err<err_limit)

def _nodes_to_MFNode(nodes, gw):
    MFNode = gw.jvm.net.epsilony.mf.geomodel.MFNode
    result = gw.jvm.java.util.ArrayList()
    for nd in nodes:
        mfnd = MFNode()
        coords = gw.new_array(gw.jvm.double, len(nd.coord))
        for i in range(len(nd.coord)):
            coords[i] = nd.coord[i]
        mfnd.setCoord(coords)
        mfnd.setInfluenceRadius(nd.influence_radius)
        result.append(mfnd)
    return result
