import numpy as np


MAINCHAIN = ['N', 'CA', 'C', 'O']


class GROParser:
    def __init__(self, grofile_path):
        self.fp = grofile_path
        self.mainchains = MAINCHAIN

        self.atom_align = []
        self.struct = []
        with open(self.fp) as f:
            f.readline()
            f.readline()
            for line in f:
                atom = line[13:15].strip()
                if atom in self.mainchains:
                    self.atom_align.append(atom)

                    xyz = [float(v) for v in line[23:].split()]
                    self.struct.append(xyz)

        self.struct = np.array(self.struct)

        # n atoms
        self.n_atoms = len(self.atom_align)

        # each atom indeces
        self.eachatom_indeces = {}
        for atom in MAINCHAIN:
            self.eachatom_indeces[atom] = [i for i in range(self.n_atoms) if self.atom_align[i] == atom]

    def cal_adjacent(self, cutoff_radius):
        # define indeces
        adjacent_indeces = []  # adjacent_indeces[0] = [back-chains, front-chains, floatN, floatCA, floatC, floatO]
        ab_indeces = []  # [index_a, index_b]
        max_n_adjacent = [0, 0, 0, 0, 0, 0]
        for i in range(self.n_atoms):
            radiuses = np.linalg.norm(np.subtract(self.struct, self.struct[i]), axis=1, ord=2)

            backchain_indeces = list(range(0, i))[::-1]
            frontchain_indeces = list(range(i+1, self.n_atoms))

            # define a, b
            atom = self.atom_align[i]
            if atom == 'N':
                index_a, index_b = frontchain_indeces[0:2]
            elif atom == 'CA':
                index_a, index_b = backchain_indeces[0], frontchain_indeces[0]
            elif atom == 'C':
                index_a, index_b = backchain_indeces[0:2]
            elif atom == 'O':
                index_a, index_b = backchain_indeces[0:2]
            ab_indeces.append([index_a, index_b])

            # cut back indeces
            for j in range(len(backchain_indeces)):
                grobal_idx = backchain_indeces[j]
                if radiuses[grobal_idx] <= cutoff_radius:
                    continue

                backchain_indeces = backchain_indeces[:j]

                if self.atom_align[grobal_idx] == "O" and radiuses[grobal_idx-1] <= cutoff_radius:
                    backchain_indeces.append(grobal_idx)  # "O"
                    backchain_indeces.append(grobal_idx-1)  # "C"
                break

            # cut front indeces
            for j in range(len(frontchain_indeces)):
                grobal_idx = frontchain_indeces[j]
                if radiuses[grobal_idx] > cutoff_radius:
                    frontchain_indeces = frontchain_indeces[:j]
                    break

            # floats indeces
            float_indeces = [
                j for j in range(self.n_atoms)
                if (j not in backchain_indeces+frontchain_indeces+[i]) and radiuses[j] <= cutoff_radius]

            float_indeces = [float_indeces[j] for j in np.argsort(radiuses[float_indeces])]  # sort

            float_indeces_eachatom = {atom: [] for atom in self.mainchains}
            for j in float_indeces:
                float_indeces_eachatom[self.atom_align[j]].append(j)

            float_indeces = []
            for atom in self.mainchains:
                float_indeces.append(float_indeces_eachatom[atom])

            # update max_n_adjacent
            max_n_adjacent[0] = max(max_n_adjacent[0], len(backchain_indeces))
            max_n_adjacent[1] = max(max_n_adjacent[1], len(frontchain_indeces))
            for j, float_indeces_one_atom in enumerate(float_indeces):
                max_n_adjacent[j+2] = max(max_n_adjacent[j+2], len(float_indeces_one_atom))

            # append
            adjacent_indeces.append([backchain_indeces, frontchain_indeces] + float_indeces)

        return adjacent_indeces, ab_indeces, max_n_adjacent
