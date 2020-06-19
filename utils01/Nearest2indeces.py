import numpy as np

ABDICT = {
    'CA': ['N', 'C'],
    'C': ['CA', 'N'],
    'N': ['CA', 'C'],
    'O': ['C', 'CA']}


class Nearest2indeces:
    def __init__(self, datadict):
        self.init_struct_dict = {}
        for atom_name, trj in datadict.items():
            self.init_struct_dict[atom_name] = trj[0][0].compute()

    def __call__(self, atom):
        self.target_struct = self.init_struct_dict[atom]

        atom_a, atom_b = ABDICT[atom]
        self.ref_a_struct = self.init_struct_dict[atom_a]
        self.ref_b_struct = self.init_struct_dict[atom_b]

        return [atom_a, atom_b, np.array([self._ab_near_i(i) for i in range(self.target_struct.shape[0])])]

    def _ab_near_i(self, i):
        radiuslist = np.sqrt(np.sum(np.square(np.subtract(self.ref_a_struct, self.target_struct[i])), axis=1))
        a = np.argsort(radiuslist)[0]

        radiuslist = np.sqrt(np.sum(np.square(np.subtract(self.ref_b_struct, self.target_struct[i])), axis=1))
        b = np.argsort(radiuslist)[0]

        return a, b
