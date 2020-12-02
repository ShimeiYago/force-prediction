import numpy as np
import dask.array as da
import h5py
import math
import time

R_DUMMY_CN = 0.134
THETA_DUMMY_C = 96
PHI_DUMMY = 42  # わざと符合逆にしてる
THETA_DUMMY_N = 118
PSI_DUMMY = -54


class DiscriptorGenerator:
    def __init__(self, outpath, batchsize,
                 mainchain, n_atoms, each_n_atoms, slice_indeces,
                 adjacent_indeces, ab_indeces, atom_align, resid_dict,
                 EXPLANATORY_NAME, RESPONSE_NAME):

        self.OUTPATH = outpath
        self.BATCHSIZE = batchsize

        self.MAINCHAIN = mainchain
        self.N_ATOMS = n_atoms
        self.EACH_N_ATOMS = each_n_atoms
        self.SLICE_INDECES = slice_indeces
        self.AB_INDECES = ab_indeces

        self.ADJACENT_INDECES, self.MAX_N_ADJACENT, self.INPUTDIMS_ONLY_DESCRIPTOR, self.INPUTDIMS \
            = self._rewrite_indeces(adjacent_indeces, atom_align)

        self.EXPLANATORY_NAME = EXPLANATORY_NAME
        self.RESPONSE_NAME = RESPONSE_NAME

        self.MAX_RECIPROCAL_DADIUS = 10

        # dummy C of N-terminal and N of C-terminal
        x = R_DUMMY_CN * np.cos(np.deg2rad(THETA_DUMMY_C))
        y = R_DUMMY_CN * np.cos(np.deg2rad(PHI_DUMMY))
        z = np.sqrt(R_DUMMY_CN**2 - x**2 - y**2)
        self.dummy_c = np.array([x, y, z])

        x = R_DUMMY_CN * np.cos(np.deg2rad(THETA_DUMMY_N))
        y = R_DUMMY_CN * np.cos(np.deg2rad(PSI_DUMMY))
        z = np.sqrt(R_DUMMY_CN**2 - x**2 - y**2)
        self.dummy_n = np.array([x, y, z])

        # amino acid
        self.RESID_DICT = resid_dict


    def _rewrite_indeces(self, adjacent_indeces, atom_align):
        # max_n_adjacent
        max_n_adjacent = {atom: [] for atom in self.MAINCHAIN}
        for atom in self.MAINCHAIN:
            l, u = self.SLICE_INDECES[atom]
            tmp_adjacent_indeces = [adjacent_indeces[i] for i in range(l, u)]
            for n in range(len(adjacent_indeces[0])):
                max_n_adjacent[atom].append(max([len(x[n]) for x in tmp_adjacent_indeces]))

        # inputdim_only_descriptor （これはresidueのonehotは含まれない。つまり実際ののinputdimは+309）
        inputdims_only_descriptor = {atom: sum(li) * 4 for atom, li in max_n_adjacent.items()}
        max_inputdim_only = max([dim for dim in inputdims_only_descriptor.values()])

        inputdims = {atom: dims+self.EACH_N_ATOMS[atom] for atom, dims in inputdims_only_descriptor.items()}

        # maxになるように自分自身のindexで埋める
        new_adjacent_indeces = adjacent_indeces
        for i in range(len(adjacent_indeces)):
            atom = atom_align[i]
            for j in range(len(adjacent_indeces[i])):
                new_adjacent_indeces[i][j] = adjacent_indeces[i][j] + [i] * (max_n_adjacent[atom][j] - len(adjacent_indeces[i][j]))

        # join
        adjacent_indeces = new_adjacent_indeces
        for i in range(len(adjacent_indeces)):
            adjacent_indeces_i = []
            for j in range(len(adjacent_indeces[i])):
                adjacent_indeces_i = adjacent_indeces_i + adjacent_indeces[i][j]
            adjacent_indeces[i] = adjacent_indeces_i + ([i] * (max_inputdim_only // 4 - len(adjacent_indeces_i)))

        adjacent_indeces = np.array(adjacent_indeces)
        return adjacent_indeces, max_n_adjacent, inputdims_only_descriptor, inputdims


    def __call__(self, coords, forces, groupname, only_terminal_rate=0.0):
        # ## preprocess ## #
        descriptors = self._preprocess(coords)

        # ## main process ## #
        self._mainprocess(descriptors, forces, groupname, only_terminal_rate)

        # ## output final shape ## #
        with h5py.File(self.OUTPATH, mode='r') as f:
            for atom in self.MAINCHAIN:
                X = f[f'/{atom}/{groupname}/{self.EXPLANATORY_NAME}']
                Y = f[f'/{atom}/{groupname}/{self.RESPONSE_NAME}']

                print(f'[{atom}]\tX: {X.shape}\tY: {Y.shape}')


    def _preprocess(self, coords):  # da.array function
        adjacent_coords = da.tile(coords, (1, self.N_ATOMS, 1)).reshape(coords.shape[0], self.N_ATOMS, self.N_ATOMS, 3)
        adjacent_coords = adjacent_coords.rechunk(chunks=('auto', -1, -1, -1))

        descriptors = da.subtract(
            adjacent_coords,
            adjacent_coords.transpose(0, 2, 1, 3))

        return descriptors


    def _mainprocess(self, total_descriptors, total_forces, groupname, only_terminal_rate):
        N_frames = total_descriptors.shape[0]
        part_index_list = list(range(0, N_frames, self.BATCHSIZE)) + [N_frames]
        N_process = math.ceil(N_frames / self.BATCHSIZE)
        N_process_only_terminal = math.ceil(N_process * only_terminal_rate)
        N_process_full = N_process - N_process_only_terminal
        N_frames_full = part_index_list[N_process_full]

        with h5py.File(self.OUTPATH, mode='r+') as f:
            for atom in self.MAINCHAIN:
                n_datasets = N_frames_full * self.EACH_N_ATOMS[atom]
                if atom == 'N' or atom == 'C':
                    n_datasets += N_frames - N_frames_full

                inputdim = self.INPUTDIMS[atom]

                f.create_dataset(
                    name=f'/{atom}/{groupname}/{self.EXPLANATORY_NAME}', shape=(n_datasets, inputdim),
                    compression='gzip', dtype=np.float64)
                f.create_dataset(
                    name=f'/{atom}/{groupname}/{self.RESPONSE_NAME}', shape=(n_datasets, 3),
                    compression='gzip', dtype=np.float64)

        # the process
        for i in range(N_process):
            # mode
            if i < N_process_full:
                only_terminal = False
            else:
                only_terminal = True

            start_time = time.time()
            l, u = part_index_list[i:i+2]
            part_length = u - l
            descriptors = total_descriptors[l:u].compute()
            forces = total_forces[l:u].compute()

            results = [self._descriptor(d) for d in descriptors]

            discriptors = np.array([r[0] for r in results])
            rot_matrices = np.array([r[1] for r in results])

            # rotate force
            forces = forces.reshape(-1, 3)
            forces = np.array([np.dot(force, rot_matrix) for force, rot_matrix in zip(forces, rot_matrices.reshape(-1, 3, 3))])
            forces = forces.reshape(part_length, self.N_ATOMS, 3)

            # save to hdf5
            with h5py.File(self.OUTPATH, mode='r+') as f:
                for atom in self.MAINCHAIN:
                    if only_terminal and not(atom == 'C' or atom == 'N'):
                        continue

                    atom_indeces_l, atom_indeces_u = self.SLICE_INDECES[atom]

                    if not only_terminal:
                        ll = l * self.EACH_N_ATOMS[atom]
                        uu = u * self.EACH_N_ATOMS[atom]
                    else:
                        ll = N_frames_full * self.EACH_N_ATOMS[atom] + (l - N_frames_full)
                        uu = ll + part_length

                    inputdim_only = self.INPUTDIMS_ONLY_DESCRIPTOR[atom]

                    residue_onehot = np.tile(np.eye(self.EACH_N_ATOMS[atom]), part_length).transpose(1, 0)

                    x = discriptors[:, atom_indeces_l:atom_indeces_u, :inputdim_only].reshape(-1, inputdim_only)
                    x = np.concatenate([x, residue_onehot], axis=1)
                    if only_terminal:
                        if atom == 'N':
                            x = x.reshape(part_length, self.EACH_N_ATOMS[atom], -1)[:, 0, :].reshape(-1, self.INPUTDIMS[atom])
                            y = forces[:, atom_indeces_l:atom_indeces_u, :][:, 0, :].reshape(-1, 3)
                        elif atom == 'C':
                            x = x.reshape(part_length, self.EACH_N_ATOMS[atom], -1)[:, -1, :].reshape(-1, self.INPUTDIMS[atom])
                            y = forces[:, atom_indeces_l:atom_indeces_u, :][:, -1, :].reshape(-1, 3)
                    else:
                        y = forces[:, atom_indeces_l:atom_indeces_u, :].reshape(-1, 3)

                    X = f[f'/{atom}/{groupname}/{self.EXPLANATORY_NAME}']
                    Y = f[f'/{atom}/{groupname}/{self.RESPONSE_NAME}']
                    X[ll:uu] = x
                    Y[ll:uu] = y

            # print progress
            elapsed_time = time.time() - start_time
            print('\r', i+1, '/', N_process, f' ({elapsed_time:.2f}s)', end="")
        print()


    def _e(self, R):  # np.ndarray function
        radius = np.linalg.norm(R, axis=0, ord=2)
        return np.divide(R, radius)

    def _cal_rotation_matrix(self, Ri_a, Ri_b):
        # Ri.shape = (,3)

        # rotation matrix
        e1 = self._e(Ri_a)
        e2 = self._e(np.subtract(Ri_b, np.dot(Ri_b, e1)*e1))
        e3 = np.cross(e1, e2)
        rotation_matrix = np.array([e1, e2, e3]).T

        return rotation_matrix

    def _descriptor(self, descriptor, dummy_flag=False):
        # ## rotation matrix ## #
        rot_matrices = np.array([
            self._cal_rotation_matrix(struct[self.AB_INDECES[i][0]], struct[self.AB_INDECES[i][1]])
            for i, struct in enumerate(descriptor)
            ])

        # ## rotate ## #
        descriptor = np.array([np.dot(d, rot_matrix) for d, rot_matrix in zip(descriptor, rot_matrices)])

        # ## radius process ## #
        radiuses = np.linalg.norm(descriptor, axis=2, ord=2).reshape(self.N_ATOMS, self.N_ATOMS, 1)
        radiuses = np.where(radiuses==0, np.inf, radiuses)

        reciprocal_radiuses = np.divide(1, radiuses)
        reciprocal_radiuses = np.divide(reciprocal_radiuses, self.MAX_RECIPROCAL_DADIUS)  # normalize

        descriptor = np.divide(descriptor, radiuses)
        descriptor = np.concatenate([reciprocal_radiuses, descriptor], axis=2)

        descriptor = np.array([D[self.ADJACENT_INDECES[i]] for i, D in enumerate(descriptor)])

        # add dummy C into N-terminal and N into C-terminal
        if dummy_flag:
            descriptor = self._add_dummy(descriptor)

        # reshape
        descriptor = descriptor.reshape(descriptor.shape[0], -1)

        return descriptor, rot_matrices

    def _add_dummy(self, descriptor):
        x, y, z = self.dummy_c / R_DUMMY_CN
        descriptor[self.SLICE_INDECES['N'][0]] = np.array([R_DUMMY_CN, x, y, z])

        x, y, z = self.dummy_n / R_DUMMY_CN
        descriptor[self.SLICE_INDECES['C'][1]] = np.array([R_DUMMY_CN, x, y, z])

        return descriptor


    def normalize(self, gropuname1, groupname2):
        # ## normalize y ## #
        with h5py.File(self.OUTPATH, mode='r+') as f:
            for atom in self.MAINCHAIN:
                # load
                train_y = da.from_array(f[f'/{atom}/{gropuname1}/{self.RESPONSE_NAME}'], chunks=("auto", 3))
                val_y = da.from_array(f[f'/{atom}/{groupname2}/{self.RESPONSE_NAME}'], chunks=("auto", 3))

                total_y = da.concatenate([train_y, val_y], axis=0)
                y_mean = da.mean(total_y.reshape(-1), axis=0).compute()
                y_std = da.std(total_y.reshape(-1), axis=0).compute()

                # normalize
                train_y = da.divide(da.subtract(train_y, y_mean), y_std)
                val_y = da.divide(da.subtract(val_y, y_mean), y_std)

                # save
                da.to_hdf5(self.OUTPATH, f'/{atom}/{gropuname1}/{self.RESPONSE_NAME}', train_y)
                da.to_hdf5(self.OUTPATH, f'/{atom}/{groupname2}/{self.RESPONSE_NAME}', val_y)

                f.create_dataset(name=f'/{atom}/normalization', data=np.array([y_mean, y_std]))

                print(f'[{atom}]\tmean: {y_mean:.3f}\tstd: {y_std:.3f}')


    def shuffle(self, groupname):
        with h5py.File(self.OUTPATH, mode='r+') as f:
            for atom in self.MAINCHAIN:
                X = da.from_array(f[f'/{atom}/{groupname}/{self.EXPLANATORY_NAME}'])
                Y = da.from_array(f[f'/{atom}/{groupname}/{self.RESPONSE_NAME}'])

                random_order = np.random.permutation(X.shape[0])

                X = da.slicing.shuffle_slice(X, random_order)
                Y = da.slicing.shuffle_slice(Y, random_order)

                da.to_hdf5(self.OUTPATH, f'/{atom}/{groupname}/{self.EXPLANATORY_NAME}', X)
                da.to_hdf5(self.OUTPATH, f'/{atom}/{groupname}/{self.RESPONSE_NAME}', Y)

                print(f'{atom} shuffled.')
