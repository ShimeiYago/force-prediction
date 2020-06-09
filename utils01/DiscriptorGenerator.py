import numpy as np
import dask.array as da
import h5py
import math
import time

TRAIN_NAME = "training"
VAL_NAME = "validation"
EXPLANATORY_NAME = "x"
RESPONSE_NAME = "y"


class DiscriptorGenerator:
    def __init__(self, inputdata, N_ATOMS, CUTOFF_RADIUS, outpath, batchsize, ab_indeces,
                 Index2ID, no_atom_index, no_relative_distance):
        self.train_coords, self.train_forces, self.val_coords, self.val_forces = inputdata
        self.N_ATOMS = N_ATOMS
        self.CUTOFF_RADIUS = CUTOFF_RADIUS
        self.OUTPATH = outpath
        self.BATCHSIZE = batchsize
        self.ab_indeces = ab_indeces

        self.Index2ID = Index2ID
        self.no_atom_index = no_atom_index
        self.no_relative_distance = no_relative_distance

        self.MAX_RECIPROCAL_DADIUS = 0


    def __call__(self):
        # ## preprocess ## #
        self.train_descriptors = self._preprocess(self.train_coords)
        self.val_descriptors = self._preprocess(self.val_coords)

        # ## decide batchsize ## #
        self._decide_batchsize(30)

        # ## caluclate MAX_ATOMS ## #
        self.MAX_ATOMS, self.MAX_RECIPROCAL_DADIUS = self._cal_max_params()
        print(f'MAX_ATOMS: {self.MAX_ATOMS}')

        # ## main process ## #
        with h5py.File(self.OUTPATH, mode='w') as f:  # create file
            pass

        self._decide_inputdim()
        self._mainprocess(self.train_descriptors, self.train_forces, TRAIN_NAME)
        self._mainprocess(self.val_descriptors, self.val_forces, VAL_NAME)

        # ## normalize ## #
        self._normalize()

        # ## output final shape ## #
        with h5py.File(self.OUTPATH, mode='r') as f:
            train_x = f[f'/{TRAIN_NAME}/{EXPLANATORY_NAME}']
            train_y = f[f'/{TRAIN_NAME}/{RESPONSE_NAME}']
            val_x = f[f'/{VAL_NAME}/{EXPLANATORY_NAME}']
            val_y = f[f'/{VAL_NAME}/{RESPONSE_NAME}']

            print('--- output shape ---')
            print('train_x:', train_x.shape)
            print('train_y:', train_y.shape)
            print('val_x:', val_x.shape)
            print('val_y:', val_y.shape)


    def _preprocess(self, descriptors):  # da.array function
        descriptors = da.tile(descriptors, (1, self.N_ATOMS, 1)).reshape(descriptors.shape[0], self.N_ATOMS, self.N_ATOMS, 3)
        descriptors = descriptors.rechunk(chunks=('auto', -1, -1, -1))

        descriptors = da.subtract(descriptors, descriptors.transpose(0, 2, 1, 3))

        descriptors = descriptors.reshape(-1, self.N_ATOMS, 3)

        return descriptors


    def _decide_inputdim(self):
        self.INPUTDIM = 0

        if self.Index2ID:
            self.N_RESIDUE = len(np.unique(list(self.Index2ID.values())))
            self.INPUTDIM += self.N_RESIDUE

        if not self.no_atom_index:
            self.INPUTDIM += self.N_ATOMS

        if not self.no_relative_distance:
            self.INPUTDIM += self.MAX_ATOMS * 5
        else:
            self.INPUTDIM += self.MAX_ATOMS * 4

    def _decide_batchsize(self, n_unit):
        N_train_datasets = self.train_descriptors.shape[0]
        if not self.BATCHSIZE:
            self.BATCHSIZE = N_train_datasets // n_unit

    def _cal_max_params(self):
        radiuses = np.linalg.norm(self.train_descriptors[:self.BATCHSIZE*3].compute(), axis=2, ord=2)
        max_atoms = np.max(np.count_nonzero(radiuses <= self.CUTOFF_RADIUS, axis=1)) - 1
        max_reciprocal_radius = 1 / np.min(radiuses[radiuses != 0])
        return max_atoms, max_reciprocal_radius


    def _mainprocess(self, descriptors, forces, groupname):
        print(f'--- Process of {groupname} data ---')
        N_datasets = descriptors.shape[0]

        with h5py.File(self.OUTPATH, mode='r+') as f:
            X = f.create_dataset(
                name=f'/{groupname}/{EXPLANATORY_NAME}', shape=(N_datasets, self.INPUTDIM),
                compression='gzip', dtype=np.float64)
            Y = f.create_dataset(
                name=f'/{groupname}/{RESPONSE_NAME}', shape=forces.shape,
                compression='gzip', dtype=np.float64)

        # the process
        part_index_list = list(range(0, N_datasets, self.BATCHSIZE)) + [N_datasets]
        N_process = math.ceil(N_datasets / self.BATCHSIZE)
        for i in range(N_process):
            start_time = time.time()
            l, u = part_index_list[i:i+2]
            part_descriptors = descriptors[l:u].compute()
            part_forces = forces[l:u].compute()

            indeces = np.array([self._get_index(D) for D in part_descriptors])

            rot_matrices = np.array([self._cal_rotation_matrix(Ri) for Ri in part_descriptors])

            # rotate
            part_descriptors = np.array([np.dot(part_descriptors[i], rot_matrices[i]) for i in range(part_descriptors.shape[0])])
            part_forces = np.array([np.dot(part_forces[i], rot_matrices[i]) for i in range(part_forces.shape[0])])

            # add radius
            part_radiuses = np.linalg.norm(part_descriptors, axis=2, ord=2)
            part_descriptors = np.concatenate(
                [part_descriptors,
                 part_radiuses.reshape(part_radiuses.shape[0], part_radiuses.shape[1], 1)],
                 axis=2)
            del part_radiuses

            # descriptor
            part_descriptors = np.array(
                [self._descriptor(D) for D in part_descriptors])

            # normalize x
            if not self.no_relative_distance:
                x_normalization_values = np.array([self.MAX_RECIPROCAL_DADIUS, 1, 1, 1, self.N_ATOMS])
            else:
                x_normalization_values = np.array([self.MAX_RECIPROCAL_DADIUS, 1, 1, 1])
            part_descriptors = np.divide(part_descriptors, x_normalization_values)

            # flatten
            part_descriptors = part_descriptors.reshape(part_descriptors.shape[0], -1)

            # add atom index (one-hot)
            if not self.no_atom_index:
                part_descriptors = np.concatenate(
                    [part_descriptors, np.identity(self.N_ATOMS)[indeces]], axis=1)

            # add residue species (one-hot)
            if self.Index2ID:
                residue_ids = np.array([self.Index2ID[idx] for idx in indeces])
                part_descriptors = np.concatenate([part_descriptors, np.identity(self.N_RESIDUE)[residue_ids]], axis=1)
                del residue_ids

            # save to hdf5
            with h5py.File(self.OUTPATH, mode='r+') as f:
                X = f[f'/{groupname}/{EXPLANATORY_NAME}']
                Y = f[f'/{groupname}/{RESPONSE_NAME}']
                X[l:u] = part_descriptors
                Y[l:u] = part_forces

            # print progress
            elapsed_time = time.time() - start_time
            print(i+1, '/', N_process, ':', f'{elapsed_time:.2f}', 's')


    def _get_index(self, Ri):  # np.ndarray function
        return np.argmin(np.sqrt(np.sum(np.square(Ri), axis=1)))

    def _e(self, R):  # np.ndarray function
        radius = np.sqrt(np.sum(np.square(R)))
        return np.divide(R, radius)

    def _cal_rotation_matrix(self, Ri):
        # Ri.shape = (309,3)

        idx = self._get_index(Ri)
        idx_a, idx_b = self.ab_indeces[idx]

        # rotation matrix
        e1 = self._e(Ri[idx_a])
        e2 = self._e(np.subtract(Ri[idx_b], np.dot(Ri[idx_b], e1)*e1))
        e3 = np.cross(e1, e2)
        rotation_matrix = np.array([e1, e2, e3]).T

        return rotation_matrix

    def _descriptor(self, D):
        # D.shape=(309,4)
        # [[x,y,z,r], ...]

        # the index of this atom
        idx = self._get_index(D)

        # add relative residue distance
        relative_residue_distances = np.array([np.abs(i-idx) for i in range(D.shape[0])]).reshape(-1, 1)
        D = np.concatenate([D, relative_residue_distances], axis=1)

        # delete itself
        D = np.delete(D, obj=idx, axis=0)

        # sort
        D = D[np.argsort(D[:, 3])]

        #  cut
        D = D[:self.MAX_ATOMS]

        if not self.no_relative_distance:
            # descriptor shape=(309,5)
            return np.array([
                [1/r, x/r, y/r, z/r, rrd] if r <= self.CUTOFF_RADIUS else [0, 0, 0, 0, rrd]
                for x, y, z, r, rrd in D])

        else:
            # descriptor shape=(309,4)
            return np.array([
                [1/r, x/r, y/r, z/r] if r <= self.CUTOFF_RADIUS else [0, 0, 0, 0]
                for x, y, z, r, rrd in D])


    def _normalize(self):
        # ## normalize y ## #
        with h5py.File(self.OUTPATH, mode='r+') as f:
            # load
            train_y = da.from_array(f[f'/{TRAIN_NAME}/{RESPONSE_NAME}'], chunks=("auto", 3))
            val_y = da.from_array(f[f'/{VAL_NAME}/{RESPONSE_NAME}'], chunks=("auto", 3))

            total_y = da.concatenate([train_y, val_y], axis=0)
            y_mean = da.mean(total_y.reshape(-1), axis=0).compute()
            y_std = da.std(total_y.reshape(-1), axis=0).compute()

            # normalize
            train_y = da.divide(da.subtract(train_y, y_mean), y_std)
            val_y = da.divide(da.subtract(val_y, y_mean), y_std)

            # save
            da.to_hdf5(self.OUTPATH, f'/{TRAIN_NAME}/{RESPONSE_NAME}', train_y)
            da.to_hdf5(self.OUTPATH, f'/{VAL_NAME}/{RESPONSE_NAME}', val_y)

        # ## save normalization values ## #
        with h5py.File(self.OUTPATH, mode='r+') as f:
            normalization = f.create_dataset(
                name='/normalization', shape=(3,), dtype=np.float64)
            normalization[...] = np.array(
                [self.MAX_RECIPROCAL_DADIUS, y_mean, y_std])

        print(
            '--- Normalization values ---\n' +
            f'MAX_RECIPROCAL_DADIUS: {self.MAX_RECIPROCAL_DADIUS:.3f}\n' +
            f'y_mean: {y_mean:.3f}\ny_std: {y_std:.3f}')
