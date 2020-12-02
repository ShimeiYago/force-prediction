import numpy as np
import sys


MAINCHAIN_CB = ['N', 'CA', 'CB', 'C', 'O']
MAINCHAIN = ['N', 'CA', 'C', 'O']
MAINCHAIN_CONVERT = {'OT1': 'O'}


class GROParser:
    def __init__(self, grofile_path, cutoff_radius, cb_mode, init_struct=None):
        if cb_mode:
            self.mainchains = MAINCHAIN_CB
        else:
            self.mainchains = MAINCHAIN

        self.resid_group_indeces = {}
        # ## load gro file ## #
        self.atom_align = []
        self.struct = []
        self.target_atom_indeces_for_xvg = []  # xvgファイルからmainchainを取り出す用のindex
        self.resid_dict = {}  # {resid_number:[name, atom_count]}
        with open(grofile_path) as f:
            f.readline()
            f.readline()
            i = 0
            for line in f:
                # about resid
                try:
                    resid = int(line[0:5])
                    residname = line[5:8]
                except ValueError:
                    continue

                if resid in self.resid_dict:
                    self.resid_dict[resid][1] = self.resid_dict[resid][1] + 1
                else:
                    self.resid_dict[resid] = [residname, 1]

                # about atom
                atom = line[12:15].strip()
                if atom in self.mainchains or atom in MAINCHAIN_CONVERT:
                    if atom in MAINCHAIN_CONVERT:
                        atom = MAINCHAIN_CONVERT[atom]

                    self.atom_align.append(atom)

                    xyz = [float(v) for v in line[23:].split()]
                    self.struct.append(xyz)

                    self.target_atom_indeces_for_xvg.append(int(line[16:20].strip())-1)

                else:
                    continue

                # about resid
                if resid not in self.resid_group_indeces:
                    self.resid_group_indeces[resid] = []

                self.resid_group_indeces[resid].append(i)
                i += 1


        if init_struct is None:
            self.struct = np.array(self.struct)
        else:
            self.struct = init_struct

        #  ## n atoms ## #
        self.n_atoms = len(self.atom_align)

        ## each atom indeces ## #
        self.eachatom_indeces = {}
        for atom in self.mainchains:
            self.eachatom_indeces[atom] = [i for i in range(self.n_atoms) if self.atom_align[i] == atom]
        self.each_n_atoms = {atom: len(indeces) for atom, indeces in self.eachatom_indeces.items()}

        self._cal_adjacent(cutoff_radius)

        self._arrange_order()

        self.init_angles = self._cal_init_angles(self.struct)


    def _cal_adjacent(self, cutoff_radius):
        # define indeces
        adjacent_indeces = []  # adjacent_indeces[0] = [back-chains, front-chains, floatN, floatCA, ...]
        ab_indeces = []  # [index_a, index_b]
        max_n_adjacent = [0] * (len(self.mainchains) + 2)
        connects_indeces = []
        self.init_radiuses = []
        for i in range(self.n_atoms):
            radiuses = np.linalg.norm(np.subtract(self.struct, self.struct[i]), axis=1, ord=2)
            self.init_radiuses.append(radiuses)

            backchain_indeces = list(range(0, i))[::-1]
            frontchain_indeces = list(range(i+1, self.n_atoms))

            # define a, b, and connects
            atom = self.atom_align[i]
            if atom == 'N':
                index_a = frontchain_indeces[0]
                if self.atom_align[i+2] == 'CB':
                    index_b = frontchain_indeces[2]
                elif self.atom_align[i+2] == 'C':
                    index_b = frontchain_indeces[1]
                else:
                    print('Error: define a, b, and connects')
                    sys.exit()

                if len(backchain_indeces) > 0:
                    connects_indeces.append([backchain_indeces[1], frontchain_indeces[0]])
                else:
                    connects_indeces.append([frontchain_indeces[0]])

            elif atom == 'CA':
                index_a = backchain_indeces[0]
                if self.atom_align[i+1] == 'CB':
                    index_b = frontchain_indeces[1]
                    connects_indeces.append([index_a, frontchain_indeces[0], index_b])
                elif self.atom_align[i+1] == 'C':
                    index_b = frontchain_indeces[0]
                    connects_indeces.append([index_a, index_b])
                else:
                    print('Error: define a, b, and connects')
                    sys.exit()

            elif atom == 'CB':
                index_a, index_b = backchain_indeces[0:2]
                connects_indeces.append([backchain_indeces[0]])

            elif atom == 'C':
                if self.atom_align[i-1] == 'CB':
                    index_a, index_b = backchain_indeces[1:3]
                elif self.atom_align[i-1] == 'CA':
                    index_a, index_b = backchain_indeces[:2]
                else:
                    print('Error: define a, b, and connects')
                    sys.exit()

                if len(frontchain_indeces) > 1:
                    connects_indeces.append([index_a, index_b, frontchain_indeces[1]])
                else:
                    connects_indeces.append([index_a, index_b])

            elif atom == 'O':
                index_a = backchain_indeces[0]
                if self.atom_align[i-2] == 'CB':
                    index_b = backchain_indeces[2]
                elif self.atom_align[i-2] == 'CA':
                    index_b = backchain_indeces[1]
                else:
                    print('Error: define a, b, and connects')
                    sys.exit()
                connects_indeces.append([index_a])

            ab_indeces.append([index_a, index_b])

            # cut back indeces
            for j in range(len(backchain_indeces)):
                grobal_idx = backchain_indeces[j]
                if radiuses[grobal_idx] <= cutoff_radius:
                    continue

                backchain_indeces = backchain_indeces[:j]

                # cut CB
                backchain_indeces = [x for x in backchain_indeces if self.atom_align[grobal_idx] != 'CB']

                # if self.atom_align[grobal_idx] == "O" and radiuses[grobal_idx-1] <= cutoff_radius:
                #     backchain_indeces.append(grobal_idx)  # "O"
                #     backchain_indeces.append(grobal_idx-1)  # "C"

                break

            # cut front indeces
            for j in range(len(frontchain_indeces)):
                grobal_idx = frontchain_indeces[j]
                if radiuses[grobal_idx] <= cutoff_radius:
                    continue

                frontchain_indeces = frontchain_indeces[:j]

                # cut CB
                frontchain_indeces = [x for x in frontchain_indeces if self.atom_align[grobal_idx] != 'CB']

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
        self.connects_indeces = connects_indeces
        self.init_radiuses = np.array(self.init_radiuses)

    def _arrange_order(self):
        arranged_indeces = []
        for atom in self.mainchains:
            arranged_indeces.extend(self.eachatom_indeces[atom])

        self.struct = self.struct[arranged_indeces, :]
        self.init_radiuses = self.init_radiuses[:, arranged_indeces][arranged_indeces, :]
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

        CONNECT_INDECES = self.connects_indeces
        for i in range(len(CONNECT_INDECES)):
            for j in range(len(CONNECT_INDECES[i])):
                CONNECT_INDECES[i][j] = index_convert_dict[CONNECT_INDECES[i][j]]

        RESID_GROUP_INDECES = self.resid_group_indeces
        for k, v in self.resid_group_indeces.items():
            RESID_GROUP_INDECES[k] = [index_convert_dict[x] for x in v]
        self.resid_group_indeces = RESID_GROUP_INDECES

        self.adjacent_indeces = []
        self.ab_indeces = []
        self.connects_indeces = []
        for i in self.arranged_indeces:
            self.adjacent_indeces.append(ADJACENT_INDECES[i])
            self.ab_indeces.append(AB_INDECES[i])
            self.connects_indeces.append(CONNECT_INDECES[i])



    def _cal_init_angles(self, init_struct):
        discriptors = np.tile(init_struct, (self.n_atoms, 1)).reshape(self.n_atoms, -1, 3)
        discriptors = discriptors - discriptors.transpose(1, 0, 2)

        n_index_list = list(range(self.slice_indeces['N'][0], self.slice_indeces['N'][1]))
        c_index_list = list(range(self.slice_indeces['C'][0], self.slice_indeces['C'][1]))
        ca_index_list = list(range(self.slice_indeces['CA'][0], self.slice_indeces['CA'][1]))

        init_angles = []
        for i, d in enumerate(discriptors):
            atom = self.atom_align[i]

            if atom == 'N' and i != n_index_list[0]:
                resid_i = n_index_list.index(i)
                c_index = c_index_list[resid_i-1]
                ca_index = ca_index_list[resid_i]
                angle = self._cal_angle(d[ca_index], d[c_index])

            elif atom == 'CA':
                resid_i = ca_index_list.index(i)
                c_index = c_index_list[resid_i]
                n_index = n_index_list[resid_i]
                angle = self._cal_angle(d[c_index], d[n_index])

            elif atom == 'C' and i != c_index_list[-1]:
                resid_i = c_index_list.index(i)
                ca_index = ca_index_list[resid_i]
                n_index = n_index_list[resid_i+1]
                angle = self._cal_angle(d[ca_index], d[n_index])

            else:
                angle = np.nan

            init_angles.append(angle)

        return init_angles


    def _cal_angle(self, u, v):
        inn = np.inner(u, v)
        norm = np.sqrt(np.sum(np.square(u))) * np.sqrt(np.sum(np.square(v)))

        cos_theta = inn / norm
        deg = np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        return deg
