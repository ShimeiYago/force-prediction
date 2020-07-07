import numpy as np


MAINCHAIN = ['N', 'CA', 'C', 'O']


class GROParser:
    def __init__(self, grofile_path, cutoff_radius):
        self.mainchains = MAINCHAIN

        # ## load gro file ## #
        self.atom_align = []
        self.struct = []
        with open(grofile_path) as f:
            f.readline()
            f.readline()
            for line in f:
                atom = line[13:15].strip()
                if atom in self.mainchains:
                    self.atom_align.append(atom)

                    xyz = [float(v) for v in line[23:].split()]
                    self.struct.append(xyz)

        self.struct = np.array(self.struct)

        #  ## n atoms ## #
        self.n_atoms = len(self.atom_align)

        # ## each atom indeces ## #
        self.eachatom_indeces = {}
        for atom in MAINCHAIN:
            self.eachatom_indeces[atom] = [i for i in range(self.n_atoms) if self.atom_align[i] == atom]
        self.each_n_atoms = {atom: len(indeces) for atom, indeces in self.eachatom_indeces.items()}

        self._cal_adjacent(cutoff_radius)

        self._arrange_order()

    def _cal_adjacent(self, cutoff_radius):
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

        self.adjacent_indeces = adjacent_indeces
        self.ab_indeces = ab_indeces
        self.max_n_adjacent = max_n_adjacent

    def _arrange_order(self):
        arranged_indeces = []
        for atom in self.mainchains:
            arranged_indeces.extend(self.eachatom_indeces[atom])

        self.struct = self.struct[arranged_indeces, :]
        self.arranged_indeces = arranged_indeces

        self.rearranged_indeces = np.argsort(arranged_indeces).tolist()

        i = 0
        ATOM_ALIGN = []
        slice_indeces = {}
        for atom in self.mainchains:
            j = i+len(self.eachatom_indeces[atom])
            slice_indeces[atom] = [i, j]
            ATOM_ALIGN = ATOM_ALIGN + [atom] * (j-i)
            i = j

        self.atom_align = ATOM_ALIGN
        self.slice_indeces = slice_indeces

        self.eachatom_indeces = {atom: list(range(i, j)) for atom, [i, j] in slice_indeces.items()}

        index_convert_dict = {orig_index: new_index for new_index, orig_index in enumerate(arranged_indeces)}
        self.index_convert_dict = index_convert_dict

        ADJACENT_INDECES = self.adjacent_indeces
        for i in range(len(ADJACENT_INDECES)):
            for j in range(len(ADJACENT_INDECES[i])):
                for k in range(len(ADJACENT_INDECES[i][j])):
                    ADJACENT_INDECES[i][j][k] = index_convert_dict[ADJACENT_INDECES[i][j][k]]

        AB_INDECES = self.ab_indeces
        for i in range(len(AB_INDECES)):
            for j in range(len(AB_INDECES[i])):
                AB_INDECES[i][j] = index_convert_dict[AB_INDECES[i][j]]

        self.adjacent_indeces = []
        self.ab_indeces = []
        for i in self.arranged_indeces:
            self.adjacent_indeces.append(ADJACENT_INDECES[i])
            self.ab_indeces.append(AB_INDECES[i])
