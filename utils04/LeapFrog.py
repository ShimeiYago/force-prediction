import numpy as np

MASS = {'CA': 12.01100, 'C': 12.01100, 'O': 15.99900, 'N': 14.00700}
K = 1  # spring constant
DT = 0.002


class LeapFrog:
    def __init__(self, discriptor_generator, model, normalization,
                 N_ATOMS, MAINCHAIN, SLICE_INDECES, ATOM_ALIGN,
                 CONNECT_INDECES, INIT_RADIUSES):
        self.discriptor_generator = discriptor_generator
        self.model = model
        self.normalization = normalization
        self.N_ATOMS = N_ATOMS
        self.MAINCHAIN = MAINCHAIN
        self.SLICE_INDECES = SLICE_INDECES
        self.ATOM_ALIGN = ATOM_ALIGN
        self.CONNECT_INDECES = CONNECT_INDECES
        self.INIT_RADIUSES = INIT_RADIUSES

        self.weights = np.array([MASS[atom] for atom in ATOM_ALIGN]).reshape(-1, 1)

    def __call__(self, pre_struct, current_struct):
        return np.subtract(2*current_struct, pre_struct) + np.divide(self._cal_force(current_struct), self.weights) * (DT**2)

    def _cal_force(self, discriptors):
        discriptors = np.tile(discriptors, (self.N_ATOMS, 1)).reshape(self.N_ATOMS, -1, 3)
        discriptors = discriptors - discriptors.transpose(1, 0, 2)

        discriptor, rot_matrices = self.discriptor_generator._descriptor(discriptors)

        # cal force by model
        forces = np.zeros((self.N_ATOMS, 3))
        for atom in self.MAINCHAIN:
            i, j = self.SLICE_INDECES[atom]
            force = self.model[atom].predict(discriptor[i:j])
            y_mean, y_std = self.normalization[atom]
            force = np.add(np.multiply(force, y_std), y_mean)
            forces[i:j] = force

        # cal force of spring
        spring_forces = np.array([self._cal_spring_force(i, r_vecs) for i, r_vecs in enumerate(discriptors)])

        # rotate
        forces = np.array([np.dot(force, np.linalg.inv(rot_matrix)) for force, rot_matrix in zip(forces, rot_matrices)])

        return np.add(forces, spring_forces)

    def _cal_spring_force(self, i, r_vecs):
        r_vecs = r_vecs[self.CONNECT_INDECES[i]]
        Ls = self.INIT_RADIUSES[i][self.CONNECT_INDECES[i]].reshape(-1, 1)
        rs = np.linalg.norm(r_vecs, axis=1, ord=2).reshape(-1, 1)

        forces = np.multiply(r_vecs, K*(1-Ls/rs))
        return np.sum(forces, axis=0)