from argparse import ArgumentParser
import os
from sklearn.metrics import roc_auc_score,roc_curve
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
If you only use our default testset elements, the command below is sufficient.
default testset:
    positive: fda
    negative: chembl, zinc15, gdb17
python calculate_auroc.py -d <score_dir>

Our script supports broadcasting when you use argument SCORE_DIR.
If there is no file with the exact same name in the directory (due to extension, etc.),
it searches for the file through broadcasting and uses it.
ex) <score_dir>/ => fda.csv, chembl.csv, zinc15.csv, gdb17.csv, GDB17.csv
    - label 'gdb17.csv' uses 'gdb17.csv'    (Find the exact same name)
    - label 'FDA' uses 'fda.csv'            (Use similar name)
    - label 'chembl' uses 'chembl.csv'      (Use similar name)
#    - label 'gdb17' => ERROR.               (There are two files, gdb17.csv, GDB17.csv)
    - label 'gdb17' => 'gdb17.csv', 'GDB17.csv'
General command
python calculate_auroc.py -d <score_dir> -p fda -n chembl zinc15 gdb17
"""

def broadcast (score_dir, score_dir_list, name) :
    if name in score_dir_list :
        return [(name, os.path.join(score_dir, name))]
    else :
        filelist = [f for f in score_dir_list if name.lower() in f.lower()]
        assert len(filelist) > 0, \
            f"Fail to broadcast '{name}' in '{args.score_dir}': NO FILE"
        #assert len(filelist) == 1, \
        #    f"Fail to broadcast '{name}' in '{args.score_dir}': TOO MANY FILE - {str(filelist)[1:-1]}"
        filelist.sort()
        return [(f, os.path.join(score_dir, f)) for f in filelist]

def compute_auroc (pos, neg) :
    true_list = np.array([1 for _ in range(len(pos))] + [0 for _ in range(len(neg))])
    score_list = np.array(pos + neg)
    return roc_auc_score(true_list, score_list)

parser = ArgumentParser(description='Calculation AUROC score')
parser.add_argument('-d', '--score_dir', type=str, default=None, \
                                        help='score file directory path')
parser.add_argument('-p', '--positive', nargs='+', type=str, default=['fda'], \
                                        help='positive(drug) set list')
parser.add_argument('-n', '--negative', nargs='+', type=str, default=['chembl', 'zinc15', 'gdb17'], \
                                        help='negative(non-drug-like) set list')
parser.add_argument('-o', '--output', type=str, default = None, \
                                        help='ROC graph output path(matplotlib required)') # TODO
args = parser.parse_args()

# Load Data (Broadcast in args.SCORE_DIR)
pos_set = {}
neg_set = {}
score_dir_list = []
if args.score_dir :
    score_dir_list = os.listdir(args.score_dir)

pos_list = [] 
neg_list = []

for p in args.positive :
    if args.score_dir :
        file_paths = broadcast(args.score_dir, score_dir_list, p)
    else :
        file_paths = [(p, p)]
    for fn, file_path in file_paths :
        pos_list.append(fn)
        data = pd.read_csv (file_path, index_col = False, names = ['SMILES', 'score'])
        pos_set[fn] = data.score.tolist()

for n in args.negative :
    if args.score_dir :
        file_paths = broadcast(args.score_dir, score_dir_list, n)
    else :
        file_paths = [(n, n)]
    for fn, file_path in file_paths :
        neg_list.append(fn)
        data = pd.read_csv (file_path, index_col = False, names = ['SMILES', 'score'])
        neg_set[fn] = data.score.tolist()

# Print AUROC score
for p in pos_list :
    for n in neg_list :
        auroc = compute_auroc (pos_set[p], neg_set[n])
        print(f'{p}\t{n}\t{auroc:.4f}')
