import dask.array as da
import dask.dataframe as ddf
import sys
import numpy as np


class ReadXVGs:
    def __init__(self, target_atom_indeces, arranged_indeces: list, skip=1):
        self.target_atom_indeces = target_atom_indeces
        self.arranged_indeces = arranged_indeces
        self.skip = skip

    def __call__(self, fplist: list):
        coords_list, forces_list = [], []
        for fp_coord, fp_force, init_time, maxlen in fplist:
            init_time = int(init_time)
            maxlen = int(maxlen)
            coord = self._read_xvg(fp_coord)[init_time:][[i*self.skip for i in range(maxlen)]]
            force = self._read_xvg(fp_force)[init_time:][[i*self.skip for i in range(maxlen)]]

            # check shape
            self._check_shape(coord.shape, force.shape, fp_coord, fp_force)

            coords_list.append(coord)
            forces_list.append(force)

        # concatenate
        n_atoms = len(self.arranged_indeces)
        coords = da.stack(coords_list).transpose(1, 0, 2, 3).reshape(-1, n_atoms, 3)
        forces = da.stack(forces_list).transpose(1, 0, 2, 3).reshape(-1, n_atoms, 3)

        # arrange order
        coords = coords[:, self.arranged_indeces, :]
        forces = forces[:, self.arranged_indeces, :]

        return coords, forces

    def _read_xvg(self, filepath: str):
        data = ddf.read_csv(filepath, comment='@', delimiter='\t', sample=1000000,
                            header=None, skiprows=14).to_dask_array(lengths=True)[:, 1:]

        return data.reshape(data.shape[0], -1, 3)[:, self.target_atom_indeces, :]

    def _check_shape(self, shape1: tuple, shape2: tuple, fp1: str, fp2: str):
        if shape1 != shape2:
            print(f"shape of coord({fp1}) and force({fp2}) is not matched.")
            sys.exit()
