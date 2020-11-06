import numpy as np

MASS = {'CA': 12.01100, 'CB': 12.01100, 'C': 12.01100, 'O': 15.99900, 'N': 14.00700}
DT = 0.002


class LeapFrog:
    def __init__(self, discriptor_generator, model, normalization,
                 k_length, k_angle,
                 N_ATOMS, MAINCHAIN, SLICE_INDECES, ATOM_ALIGN,
                 group_indeces,
                 CONNECT_INDECES, INIT_RADIUSES, 
                 INPUTDIMS_ONLY_DESCRIPTOR, EACH_N_ATOMS,
                 init_struct,
                 dummy_flag,
                 INIT_ANGLES):
        self.discriptor_generator = discriptor_generator
        self.model = model
        self.normalization = normalization
        self.k_length = k_length  # spring constant
        self.k_angle = k_angle  # angle constant
        self.N_ATOMS = N_ATOMS
        self.MAINCHAIN = MAINCHAIN
        self.SLICE_INDECES = SLICE_INDECES
        self.ATOM_ALIGN = ATOM_ALIGN
        self.GROUP_INDECES = group_indeces
        self.CONNECT_INDECES = CONNECT_INDECES
        self.INIT_RADIUSES = INIT_RADIUSES
        self.INPUTDIMS_ONLY_DESCRIPTOR = INPUTDIMS_ONLY_DESCRIPTOR
        self.EACH_N_ATOMS = EACH_N_ATOMS
        self.INIT_ANGLES = INIT_ANGLES

        self.weights = np.array([MASS[atom] for atom in ATOM_ALIGN])

        init_veloc = np.subtract(init_struct[1], init_struct[0]) / DT
        self.T2s = self._cal_KE2(init_veloc)

        self.dummy_flag = dummy_flag

    def __call__(self, pre_struct, current_struct):
        veloc = np.subtract(current_struct, pre_struct) / DT + np.divide(self._cal_force(current_struct), self.weights.reshape(-1, 1)) * DT
        alphas = self._cal_alpha(veloc)
        return np.add(current_struct, np.multiply(veloc, alphas) * DT)

    def _cal_force(self, discriptors):
        discriptors = np.tile(discriptors, (self.N_ATOMS, 1)).reshape(self.N_ATOMS, -1, 3)
        discriptors = discriptors - discriptors.transpose(1, 0, 2)

        discriptor, rot_matrices = self.discriptor_generator._descriptor(discriptors, dummy_flag=self.dummy_flag)

        # cal force by model
        forces = np.zeros((self.N_ATOMS, 3))
        for atom in self.MAINCHAIN:
            i, j = self.SLICE_INDECES[atom]
            inputdim_only_descriptor = self.INPUTDIMS_ONLY_DESCRIPTOR[atom]
            residue_onehot = np.eye(self.EACH_N_ATOMS[atom])
            X = np.hstack([discriptor[i:j, :inputdim_only_descriptor], residue_onehot])
            force = self.model[atom].predict(X)
            y_mean, y_std = self.normalization[atom]
            force = np.add(np.multiply(force, y_std), y_mean)
            forces[i:j] = force

        # rotate
        forces = np.array([np.dot(force, np.linalg.inv(rot_matrix)) for force, rot_matrix in zip(forces, rot_matrices)])

        # cal length straint force
        if self.k_length > 0:
            straint_forces = np.array([self._cal_spring_force(i, r_vecs) for i, r_vecs in enumerate(discriptors)])
            forces = np.add(forces, straint_forces)

        # cal angle straint force
        if self.k_angle > 0:
            straint_forces = self._cal_angle_straint_forces(discriptors)
            forces = np.add(forces, straint_forces)

        return forces


    def _cal_spring_force(self, i, r_vecs):
        r_vecs = r_vecs[self.CONNECT_INDECES[i]]
        Ls = self.INIT_RADIUSES[i][self.CONNECT_INDECES[i]].reshape(-1, 1)
        rs = np.linalg.norm(r_vecs, axis=1, ord=2).reshape(-1, 1)

        forces = np.multiply(r_vecs, self.k_length*(1-Ls/rs))
        return np.sum(forces, axis=0)

    def _cal_KE2(self, veloc):
        KE2s = []
        veloc_square = np.sum(np.square(veloc), axis=1)
        for indeces in self.GROUP_INDECES:
            KE2 = np.sum(np.multiply(veloc_square, self.weights)[indeces], axis=0)
            KE2s.append(KE2)

        return KE2s

    def _cal_alpha(self, veloc):
        alpha_list = [np.sqrt(T2 / KE2) for T2, KE2 in zip(self.T2s, self._cal_KE2(veloc))]

        alphas = np.ones((self.N_ATOMS, 3))
        for alpha_value, indeces in zip(alpha_list, self.GROUP_INDECES):
            alphas[indeces] = alpha_value

        return alphas


    def _cal_angle_straint_forces(self, discriptors):
        n_index_list = list(range(self.SLICE_INDECES['N'][0], self.SLICE_INDECES['N'][1]))
        c_index_list = list(range(self.SLICE_INDECES['C'][0], self.SLICE_INDECES['C'][1]))
        ca_index_list = list(range(self.SLICE_INDECES['CA'][0], self.SLICE_INDECES['CA'][1]))

        angle_straint_forces = np.zeros((self.N_ATOMS, 3))
        for i, d in enumerate(discriptors):
            init_angle = self.INIT_ANGLES[i]
            if np.isnan(init_angle):
                continue

            atom = self.ATOM_ALIGN[i]

            if atom == 'N' and i != n_index_list[0]:
                resid_i = n_index_list.index(i)
                index1 = c_index_list[resid_i-1]  # C
                index2 = ca_index_list[resid_i]  # CA

            elif atom == 'CA':
                resid_i = ca_index_list.index(i)
                index1 = c_index_list[resid_i]  # C
                index2 = n_index_list[resid_i]  # N

            elif atom == 'C' and i != c_index_list[-1]:
                resid_i = c_index_list.index(i)
                index1 = ca_index_list[resid_i]  # CA
                index2 = n_index_list[resid_i+1]  # N

            u = d[index1]
            v = d[index2]

            r_u = np.linalg.norm(u, ord=2)
            r_v = np.linalg.norm(v, ord=2)

            angle = self._cal_angle(u, v)
            diff_sin = np.sin(np.deg2rad(init_angle - angle))

            f_unit_vec1 = self._cal_straint_unit_vec(u, v)
            f_unit_vec2 = self._cal_straint_unit_vec(v, u)

            f_vec1 = self.k_angle * r_u * diff_sin * f_unit_vec1 / 2
            f_vec2 = self.k_angle * r_v * diff_sin * f_unit_vec2 / 2

            angle_straint_forces[index1] = np.add(angle_straint_forces[index1], f_vec1)
            angle_straint_forces[index2] = np.add(angle_straint_forces[index2], f_vec2)

        return angle_straint_forces


    def _cal_angle(self, u, v):
        inn = np.inner(u, v)
        norm = np.sqrt(np.sum(np.square(u))) * np.sqrt(np.sum(np.square(v)))

        cos_theta = inn / norm
        deg = np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        return deg

    def _cal_straint_unit_vec(self, u, v):
        f = v - (np.inner(u, v)/np.linalg.norm(u, ord=2)**2)*u
        return f / np.linalg.norm(f, ord=2)
