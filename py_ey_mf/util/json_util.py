'''

@author: Man YUAN <epsilonyuan@gmail.com>
'''
from json import JSONEncoder
import numpy as np

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
