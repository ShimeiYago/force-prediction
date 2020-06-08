import numpy as np


class Nearest2indeces:
    def __init__(self, struct):
        self.struct = struct

    def __call__(self, i):
        radiuslist = np.sqrt(np.sum(np.square(np.subtract(self.struct, self.struct[i])), axis=1))
        return np.argsort(radiuslist)[1:3]