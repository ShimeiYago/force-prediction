import numpy as np
import sys


class ReadXVGs:
    def __init__(self, init_time:int, dtype:str='float64'):
        self.init_time = init_time
        self.dtype = dtype


    def __call__(self, fplist:list):
        coords_list, forces_list = [], []
        for fp_coord, fp_force in fplist:
            coord = self._read_xvg(fp_coord)[self.init_time:]
            force = self._read_xvg(fp_force)[self.init_time:]

            # check shape
            self._check_shape(coord.shape, force.shape)
            
            coords_list.append(coord)
            forces_list.append(force)
        
        # concatenate
        coords = np.concatenate(coords_list, 0)
        forces = np.concatenate(forces_list, 0)

        # check shape
        self._check_shape(coords.shape, forces.shape)

        return coords, forces

    def _read_xvg(self, filepath: str) -> np.ndarray:
        trj = np.loadtxt(filepath, comments=['#', '@'], delimiter='\t', dtype=self.dtype)[:, 1:]

        trj = trj.reshape(trj.shape[0], -1, 3)

        return trj
    
    def _check_shape(self, shape1:tuple, shape2:tuple):
            if shape1 != shape2:
                print(f"shape of coord({fp1}) and force({fp2}) is not matched.")
                sys.exit()

