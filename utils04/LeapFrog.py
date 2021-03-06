import numpy as np

MASS = {'CA': 12.01100, 'C': 12.01100, 'O': 15.99900, 'N': 14.00700}
DT = 0.002


class LeapFrog:
    def __init__(self, discriptor_generator, model, normalization, k,
                 N_ATOMS, MAINCHAIN, SLICE_INDECES, ATOM_ALIGN,
                 group_indeces,
                 CONNECT_INDECES, INIT_RADIUSES, INPUTDIMS,
                 init_struct):
        self.discriptor_generator = discriptor_generator
        self.model = model
        self.normalization = normalization
        self.k = k  # spring constant
        self.N_ATOMS = N_ATOMS
        self.MAINCHAIN = MAINCHAIN
        self.SLICE_INDECES = SLICE_INDECES
        self.ATOM_ALIGN = ATOM_ALIGN
        self.GROUP_INDECES = group_indeces
        self.CONNECT_INDECES = CONNECT_INDECES
        self.INIT_RADIUSES = INIT_RADIUSES
        self.INPUTDIMS = INPUTDIMS

        self.weights = np.array([MASS[atom] for atom in ATOM_ALIGN])

        init_veloc = np.subtract(init_struct[1], init_struct[0]) / DT
        self.T2s = self._cal_KE2(init_veloc)

    def __call__(self, pre_struct, current_struct):
        veloc = np.subtract(current_struct, pre_struct) / DT + np.divide(self._cal_force(current_struct), self.weights.reshape(-1, 1)) * DT
        alphas = self._cal_alpha(veloc)
        return np.add(current_struct, np.multiply(veloc, alphas) * DT)

    def _cal_force(self, discriptors):
        discriptors = np.tile(discriptors, (self.N_ATOMS, 1)).reshape(self.N_ATOMS, -1, 3)
        discriptors = discriptors - discriptors.transpose(1, 0, 2)

        discriptor, rot_matrices = self.discriptor_generator._descriptor(discriptors)

        # cal force by model
        forces = np.zeros((self.N_ATOMS, 3))
        for atom in self.MAINCHAIN:
            i, j = self.SLICE_INDECES[atom]
            inputdim = self.INPUTDIMS[atom]
            force = self.model[atom].predict(discriptor[i:j, :inputdim])
            y_mean, y_std = self.normalization[atom]
            force = np.add(np.multiply(force, y_std), y_mean)
            forces[i:j] = force

        # rotate
        forces = np.array([np.dot(force, np.linalg.inv(rot_matrix)) for force, rot_matrix in zip(forces, rot_matrices)])

        # cal force of spring
        if self.k == 0:
            spring_forces = np.array([self._cal_spring_force(i, r_vecs) for i, r_vecs in enumerate(discriptors)])
            return np.add(forces, spring_forces)
        else:
            return forces

    def _cal_spring_force(self, i, r_vecs):
        r_vecs = r_vecs[self.CONNECT_INDECES[i]]
        Ls = self.INIT_RADIUSES[i][self.CONNECT_INDECES[i]].reshape(-1, 1)
        rs = np.linalg.norm(r_vecs, axis=1, ord=2).reshape(-1, 1)

        forces = np.multiply(r_vecs, self.k*(1-Ls/rs))
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
