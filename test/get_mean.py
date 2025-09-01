import numpy as np

for fn in ["fda", "investigation", "chembl", "zinc15", "gdb17"]:
    with open("./score/rnn_pubchem_worlddrug/" + f"{fn}.csv") as f:
        lines = f.readlines()
        scores = [float(v.strip().split(",")[1]) for v in lines]
    print(fn, np.mean(scores))
