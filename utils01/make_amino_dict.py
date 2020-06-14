# make dict which comvert atom index to amino symbol


def make_amino_dict(gro_path):
    index2symbol = {}
    symbol2id = {}
    id = 0
    with open(gro_path) as f:
        f.readline()
        f.readline()  # pass first 2 rows

        for line in f:
            if len(line) < 45:
                continue

            index = int(line[:5]) - 1
            symbol = line[5:8]
            if index not in index2symbol:
                index2symbol[index] = symbol
                
            if symbol not in symbol2id:
                symbol2id[symbol] = id
                id += 1
    
    index2id = {k: symbol2id[v] for k, v in index2symbol.items()}
    return index2id
