import dask.array as da
import dask.dataframe as ddf
import sys

MAINCHAIN = ['N', 'CA', 'C', 'O']


class ReadXVGs:
    def __init__(self, init_time: int, maxlen: int, gropath: str):
        self.init_time = init_time
        self.maxlen = maxlen

        self.atomlist = []
        with open(gropath) as f:
            for line in f:
                atom = line[13:15].strip()
                if atom in MAINCHAIN:
                    self.atomlist.append(atom)

    def __call__(self, fplist: list):
        coords_list, forces_list = [], []
        for fp_coord, fp_force in fplist:
            coord = self._read_xvg(fp_coord)[self.init_time:][:self.maxlen]
            force = self._read_xvg(fp_force)[self.init_time:][:self.maxlen]

            # check shape
            self._check_shape(coord.shape, force.shape, fp_coord, fp_force)

            coords_list.append(coord)
            forces_list.append(force)

        # concatenate
        coords = da.concatenate(coords_list, 0)
        forces = da.concatenate(forces_list, 0)

        # devide into each atom
        trjdict = {}
        for atom_name in MAINCHAIN:
            indeces = [i for i in range(len(self.atomlist)) if self.atomlist[i]==atom_name]
            trjdict[atom_name] = [coords[:, indeces, :], forces[:, indeces, :]]

        return trjdict

    def _read_xvg(self, filepath: str):
        data = ddf.read_csv(filepath, comment='@', delimiter='\t',
                            header=None, skiprows=14).to_dask_array(lengths=True)[:, 1:]

        return data.reshape(data.shape[0], -1, 3)

    def _check_shape(self, shape1: tuple, shape2: tuple, fp1: str, fp2: str):
        if shape1 != shape2:
            print(f"shape of coord({fp1}) and force({fp2}) is not matched.")
            sys.exit()
