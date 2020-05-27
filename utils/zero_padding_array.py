import numpy as np


def zero_padding_array(x: list, dtype:str='float64', maxlen=None):
    if not maxlen:
        maxlen = max([len(li) for li in x])

    x = np.array([np.pad(arr, [(0,maxlen-arr.shape[0]), (0,0)], 'constant') 
        if arr.shape[0] != 0 
        else [[0,0,0,0]]*maxlen
        for arr in x], dtype=dtype)

    return x
