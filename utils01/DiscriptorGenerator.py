import numpy as np
import dask.array as da
import h5py
import math
import time


class DiscriptorGenerator:
    def __init__(self, train_dict, val_dict,
                 CUTOFF_RADIUS, outpath, batchsize, ab_indeces,
                 TRAIN_NAME, VAL_NAME, EXPLANATORY_NAME, RESPONSE_NAME):
        self.train_coords = {atom: trj[0] for atom, trj in train_dict.items()}
        self.train_forces = {atom: trj[1] for atom, trj in train_dict.items()}
        self.val_coords = {atom: trj[0] for atom, trj in val_dict.items()}
        self.val_forces = {atom: trj[1] for atom, trj in val_dict.items()}

        self.CUTOFF_RADIUS = CUTOFF_RADIUS
        self.OUTPATH = outpath
        self.BATCHSIZE = batchsize
        self.ab_indeces = ab_indeces

        self.TRAIN_NAME = TRAIN_NAME
        self.VAL_NAME = VAL_NAME
        self.EXPLANATORY_NAME = EXPLANATORY_NAME
        self.RESPONSE_NAME = RESPONSE_NAME

        self.MAX_RECIPROCAL_DADIUS = 0

        # define N_ATOMS
        self.N_ATOMS = {atom: coords.shape[1] for atom, coords in self.train_coords.items()}

        # atom list
        self.ATOM_NAMES = [atom for atom in self.train_coords.keys()]

    def __call__(self, target_atom):
        self.target_atom = target_atom

        # ## preprocess ## #
        self.train_descriptors = {atom: self._preprocess(self.train_coords, atom) for atom in self.train_coords.keys()}
        self.val_descriptors = {atom: self._preprocess(self.val_coords, atom) for atom in self.val_coords.keys()}

        # ## decide batchsize ## #
        if not self.BATCHSIZE:
            self._decide_batchsize(100)

        # ## caluclate MAX_ATOMS ## #
        self._cal_max_params()
        print(f'MAX_N_ATOMS: {self.MAX_N_ATOMS}')

        # ## main process ## #

        # ## input dim ## #
        self.INPUTDIM = sum([n for n in self.MAX_N_ATOMS.values()]) * 4 \
            + self.N_ATOMS[self.target_atom]

        self._mainprocess(self.train_descriptors, self.train_forces[self.target_atom], self.TRAIN_NAME)
        self._mainprocess(self.val_descriptors, self.val_forces[self.target_atom], self.VAL_NAME)

        # ## normalize ## #
        self._normalize()

        # ## output final shape ## #
        with h5py.File(self.OUTPATH, mode='r') as f:
            train_x = f[f'/{self.target_atom}/{self.TRAIN_NAME}/{self.EXPLANATORY_NAME}']
            train_y = f[f'/{self.target_atom}/{self.TRAIN_NAME}/{self.RESPONSE_NAME}']
            val_x = f[f'/{self.target_atom}/{self.VAL_NAME}/{self.EXPLANATORY_NAME}']
            val_y = f[f'/{self.target_atom}/{self.VAL_NAME}/{self.RESPONSE_NAME}']

            print('train_x:', train_x.shape)
            print('train_y:', train_y.shape)
            print('val_x:', val_x.shape)
            print('val_y:', val_y.shape)


    def _preprocess(self, coords, nearby_atom):  # da.array function
        nearby_coords = coords[nearby_atom]
        nearby_coords = da.tile(nearby_coords, (1, self.N_ATOMS[self.target_atom], 1)).reshape(nearby_coords.shape[0], self.N_ATOMS[self.target_atom], self.N_ATOMS[nearby_atom], 3)
        nearby_coords = nearby_coords.rechunk(chunks=('auto', -1, -1, -1))

        target_coords = coords[self.target_atom]
        target_coords = da.tile(target_coords, (1, self.N_ATOMS[nearby_atom], 1)).reshape(target_coords.shape[0], self.N_ATOMS[nearby_atom], self.N_ATOMS[self.target_atom],  3)
        target_coords = target_coords.rechunk(chunks=('auto', -1, -1, -1))

        descriptors = da.subtract(nearby_coords, target_coords.transpose(0, 2, 1, 3))
        descriptors = descriptors.reshape(-1, self.N_ATOMS[nearby_atom], 3)

        return descriptors

    def _decide_batchsize(self, n_unit):
        N_train_datasets = self.train_descriptors[self.ATOM_NAMES[0]].shape[0]
        self.BATCHSIZE = N_train_datasets // n_unit

    def _cal_max_params(self, width=5):
        self.MAX_N_ATOMS = {}
        for atom, desc in self.train_descriptors.items():
            radiuses = np.linalg.norm(desc[:self.BATCHSIZE*width].compute(), axis=2, ord=2)
            max_atoms = np.max(np.count_nonzero((radiuses <= 1.0) & (radiuses > 0), axis=1))
            self.MAX_N_ATOMS[atom] = max_atoms

            max_reciprocal_radius = 1 / np.min(radiuses[radiuses!=0])
            self.MAX_RECIPROCAL_DADIUS = max(self.MAX_RECIPROCAL_DADIUS, max_reciprocal_radius)


    def _mainprocess(self, descriptors, forces, groupname):
        N_datasets = descriptors[self.ATOM_NAMES[0]].shape[0]

        with h5py.File(self.OUTPATH, mode='r+') as f:
            X = f.create_dataset(
                name=f'/{self.target_atom}/{groupname}/{self.EXPLANATORY_NAME}', shape=(N_datasets, self.INPUTDIM),
                compression='gzip', dtype=np.float64)
            Y = f.create_dataset(
                name=f'/{self.target_atom}/{groupname}/{self.RESPONSE_NAME}', shape=forces.shape,
                compression='gzip', dtype=np.float64)

        # the process
        part_index_list = list(range(0, N_datasets, self.BATCHSIZE)) + [N_datasets]
        N_process = math.ceil(N_datasets / self.BATCHSIZE)
        for i in range(N_process):
            start_time = time.time()
            l, u = part_index_list[i:i+2]
            part_length = u - l
            part_descriptors = {atom: desc[l:u].compute() for atom, desc in descriptors.items()}
            part_forces = forces[l:u].compute()

            part_indeces = np.array([self._get_index(D) for D in part_descriptors[self.target_atom]])

            rot_matrices = np.array([
                self._cal_rotation_matrix(
                    part_indeces[i],
                    {atom: desc[i] for atom, desc in part_descriptors.items()})
                for i in range(part_length)])

            # rotate
            for atom in self.ATOM_NAMES:
                part_descriptors[atom] = np.array([
                    np.dot(part_descriptors[atom][i], rot_matrices[i]) for i in range(part_length)])

            part_forces = np.array([np.dot(part_forces[i], rot_matrices[i]) for i in range(part_forces.shape[0])])

            # add radius
            for atom, desc in part_descriptors.items():
                part_radiuses = np.linalg.norm(desc, axis=2, ord=2)
                part_descriptors[atom] = np.concatenate(
                    [desc,
                     part_radiuses.reshape(part_radiuses.shape[0], part_radiuses.shape[1], 1)],
                    axis=2)
                del part_radiuses

            # descriptor
            part_descriptors = np.array(
                [self._descriptor({atom: desc[i] for atom, desc in part_descriptors.items()})
                 for i in range(part_length)])

            # normalize x
            part_descriptors = np.divide(
                part_descriptors,
                np.array([self.MAX_RECIPROCAL_DADIUS, 1, 1, 1])
                )

            # flatten
            part_descriptors = part_descriptors.reshape(part_descriptors.shape[0], -1)

            # add atom index (one-hot)
            part_descriptors = np.concatenate(
                [part_descriptors, np.identity(self.N_ATOMS[self.target_atom])[part_indeces]], axis=1)

            # save to hdf5
            with h5py.File(self.OUTPATH, mode='r+') as f:
                X = f[f'/{self.target_atom}/{groupname}/{self.EXPLANATORY_NAME}']
                Y = f[f'/{self.target_atom}/{groupname}/{self.RESPONSE_NAME}']
                X[l:u] = part_descriptors
                Y[l:u] = part_forces

            # print progress
            elapsed_time = time.time() - start_time
            print('\r', f'Processing {groupname} data:', i+1, '/', N_process, f' ({elapsed_time:.2f}s)', end="")
        print()


    def _get_index(self, Ri):  # np.ndarray function
        return np.argmin(np.sqrt(np.sum(np.square(Ri), axis=1)))

    def _e(self, R):  # np.ndarray function
        radius = np.sqrt(np.sum(np.square(R)))
        return np.divide(R, radius)

    def _cal_rotation_matrix(self, idx, Ri):
        # Ri[atom].shape = (309,3)

        a_atom_name = self.ab_indeces[self.target_atom][0]
        b_atom_name = self.ab_indeces[self.target_atom][1]
        a_atom_index, b_atom_index = self.ab_indeces[self.target_atom][2][idx]
        Ri_a = Ri[a_atom_name][a_atom_index]
        Ri_b = Ri[b_atom_name][b_atom_index]

        # rotation matrix
        e1 = self._e(Ri_a)
        e2 = self._e(np.subtract(Ri_b, np.dot(Ri_b, e1)*e1))
        e3 = np.cross(e1, e2)
        rotation_matrix = np.array([e1, e2, e3]).T

        return rotation_matrix

    def _descriptor(self, D):
        # D[atom].shape=(N_ATOM,4)
        # [x,y,z,r]

        # sort
        D = {atom: d[np.argsort(d[:, 3])] for atom, d in D.items()}

        # delete zero
        for atom, d in D.items():
            while True:
                if d[0, 3] != 0:
                    break
                d = np.delete(d, obj=0, axis=0)
            D[atom] = d
            
        #  cut
        D = {atom: d[:self.MAX_N_ATOMS[atom]] for atom, d in D.items()}

        # concat
        D = np.concatenate([D[atom] for atom in self.ATOM_NAMES], axis=0)

        # descriptor
        return np.array([
            [1/r, x/r, y/r, z/r] if r <= self.CUTOFF_RADIUS else [0, 0, 0, 0]
            for x, y, z, r in D])


    def _normalize(self):
        # ## normalize y ## #
        with h5py.File(self.OUTPATH, mode='r+') as f:
            # load
            train_y = da.from_array(f[f'/{self.target_atom}/{self.TRAIN_NAME}/{self.RESPONSE_NAME}'], chunks=("auto", 3))
            val_y = da.from_array(f[f'/{self.target_atom}/{self.VAL_NAME}/{self.RESPONSE_NAME}'], chunks=("auto", 3))

            total_y = da.concatenate([train_y, val_y], axis=0)
            y_mean = da.mean(total_y.reshape(-1), axis=0).compute()
            y_std = da.std(total_y.reshape(-1), axis=0).compute()

            # normalize
            train_y = da.divide(da.subtract(train_y, y_mean), y_std)
            val_y = da.divide(da.subtract(val_y, y_mean), y_std)

            # save
            da.to_hdf5(self.OUTPATH, f'/{self.target_atom}/{self.TRAIN_NAME}/{self.RESPONSE_NAME}', train_y)
            da.to_hdf5(self.OUTPATH, f'/{self.target_atom}/{self.VAL_NAME}/{self.RESPONSE_NAME}', val_y)

        # ## save normalization values ## #
        with h5py.File(self.OUTPATH, mode='r+') as f:
            normalization = f.create_dataset(
                name=f'/{self.target_atom}/normalization', shape=(3,), dtype=np.float64)
            normalization[...] = np.array(
                [self.MAX_RECIPROCAL_DADIUS, y_mean, y_std])

        print(
            f'MAX_RECIPROCAL_DADIUS: {self.MAX_RECIPROCAL_DADIUS:.3f}\n' +
            f'y_mean: {y_mean:.3f}\ny_std: {y_std:.3f}')
