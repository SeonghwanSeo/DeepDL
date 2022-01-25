# Drug-likeness scoring based on unsupervised learning

github for *Drug-likeness scoring based on unsupervised learning*
by Kyunghoon Lee, Jinho Jang, Seonghwan Seo, Jaechang Lim, and Woo Youn Kim.

After submitting the paper, I modified the code for readability and convenient use. The seed value may change during this process, which may change the result. All model weights used in paper are accessible at `test/result` (RNNLM: ~100MB, GCN: ~550KB)

If you have any problems or need help with the code, please add an issue or contact shwan0106@kaist.ac.kr.

## Table of Contents

- [Environment](#environment)
- [Data](#data)
  - [Download PubChem](#prepare-pubchem-data-for-pretrain)
  - [Data Directory Structure](#data-directory-structure)
- [Model Training](#model-training)
  - [RNNLM](#rnnlm)
  - [GCN](#gcn)
  - [PU-Learning](#pu-learning)
- [Test](#test)
  - [Scoring](#scoring)
  - [Research](#research)

## Environment

- Python=3.7
- [RDKit](https://www.rdkit.org/docs/Install.html)=2020.09.4
- [PyTorch](https://pytorch.org/)=1.7.1, CUDA=10.2
- [Hydra](https://hydra.cc/)=1.0.6
- [scikit-learn](https://scikit-learn.org/stable/)=0.24.1
- [Pandas](https://pandas.pydata.org/)=1.2.2 (Used in Research Section)

## Data

### Prepare PubChem Data for pretrain

Most of the data used in our paper is already uploaded on github(\<root>/data/).

However, in the case of the PubChem data we used for pre-training of the RNNLM model, I couldn't upload it because it contains 10 million molecules. Therefore, you should download it from [PubChem](https://pubchem.ncbi.nlm.nih.gov/).([Download](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz)) You will get the file `CID-SMILES`.

Since the total number of data is over 100 million, I recommend shuffle the raw file and extract about 12 million lines of it before preprocessing. (ex. command `shuf` of linux)

Our work only used the molecules whose SMILES are RDKit-readable, shorter than 100 characters, and represent a single molecule (SMILES without ‘.’).

```bash
cd <root_dir>/data
wget https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz
gzip -d CID-SMILES.gz

shuf -n 12000000 CID-SMILES > CID-SMILES-SMALL  # optional, but recommend
python preprocessing.py CID-SMILES-SMALL train/pubchem.smi 10000000 <cpus>

rm CID-SMILES           # optional
rm CID-SMILES-SMALL     # optional
```

### Data Directory Structure

After upper step, the `data/` directory structure will be as follows.

```bash
├── data/
    ├── preprocess.py
    ├── train/
    │   ├── chembl.smi
    │   ├── gdb17.smi
    │   ├── pubchem.smi
    │   ├── worlddrug.smi
    │   └── zinc15.smi
    └── test/
        ├── chembl.smi
        ├── fda.smi
        ├── gdb17.smi
        ├── investigation.smi
        └── zinc15.smi
```

## Model Training

Move to the `test/` directory. For RNNLM, training script is `train_rnn.py`, and for GCN, training script is `train_gcn.py`. The training result is saved in the directory `result/`. The following commands are the minimum command for implementing our paper, and you can handle more arguments and hyper parameters with [hydra override](https://hydra.cc/docs/intro#basic-example).

### RNNLM

In RNNLM, the model is trained with all data for specific times(100 epoch) without validation tests. Config file is `config/rnn.yaml`.

```bash
# pretrain with PubChem
python -u train_rnn.py \
name='rnn_pretrain' \
data.positive_file=../data/train/pubchem.smi \
train.batch_size=400 \
train.gpus=<gpus> \
train.num_workers=<num-workers> \
train.epoch=100

# fine tuning with Worlddrug
python -u train_rnn.py \
name='rnn_worlddrug' \
data.positive_file=../data/train/worlddrug.smi \
train.init_weight=result/rnn_pretrain/save.pt       # pretrain model weight file
train.batch_size=100 \
train.gpus=<gpus> \
train.num_workers=<num-workers> \
train.epoch=100
```

### GCN

In gcn learning, you can choose between two validation test mode, normal validation test (`val`) and 5 fold cross-validation test (`5cv`). Default is 5 fold cross-validation test mode, which is used in our paper. Config file is `config/gcn.yaml`

Because of using ParameterList Class of PyTorch, multi-gpu is not supported.

```bash
# Two-class classification model with Worlddrug as positive set and ZINC15 as negative set.
python -u train_gcn.py \
name='gcn_worlddrug_zinc15' \
data.positive_file=../data/train/worlddrug.smi \
data.negative_file=../data/train/zinc15.smi \
train.batch_size=100 \
train.val_mode='5cv' \
train.gpus=1 \
train.num_workers=<num-workers> \
train.epoch=200 # recommend.

# Two-class classification model with Worlddrug as positive set and ChEMBL as negative set.
python -u train_gcn.py \
name='rnn_worlddrug_chembl' \
data.positive_file=../data/train/worlddrug.smi \
data.negative_file=../data/train/chembl.smi \
train.batch_size=100 \
train.val_mode='5cv' \
train.gpus=1 \
train.num_workers=<num-workers> \
train.epoch=200

# Two-class classification model with Worlddrug as positive set and GDB17 as negative set.
python -u train_gcn.py \
name='rnn_worlddrug_gdb17' \
data.positive_file=../data/train/worlddrug.smi \
data.negative_file=../data/train/gdb17.smi \
train.batch_size=100 \
train.val_mode='5cv' \
train.gpus=1 \
train.num_workers=<num-workers> \
train.epoch=200
```

Both RNNLM and GCN, config, log, and model parameters will be saved in `<root>/test/result/<name>/`.

```bash
├── test/
    ├── calculate_score.py
    ├── result/
    │   ├── rnn_pretrain/
    │   │   ├── config.yaml # Model Hyperparameter
    │   │   ├── output.log  # Training Log
    │   │   └── save.pt     # Model Weights
    │   ├── rnn_worlddrug/
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   └── save.pt
    │   └── gcn_worlddrug_zinc15/
    │       ├── config.yaml
    │       ├── output.log
    │       └── save.pt
    ├── train_gcn.py
    ├── train_pu_gcn.py
    └── train_rnn.py
```

### PU Learning

As an additional work, we applied PU learning to minimize false negatives in the negative set. The results of this work are in the Supplementary Information. In this study, we only used the [Fusilier work](#https://www.sciencedirect.com/science/article/abs/pii/S0306457314001095), but the algorithm proposed by [Liu](#https://ieeexplore.ieee.org/abstract/document/1250918) was also implemented. GCN model was used as the classification model architecture for PU learning. During PU learning, the size of the negative set starts at 10000 and decreases until it becomes smaller than the size of the positive set (WorldDrug, 2833 molecules). Config file is `config/gcn_pu.yaml`

```bash
# PU learning with Worlddrug as positive set and ZINC15 as negative set.
python -u train_pu_gcn.py \
name='gcn_pu_worlddrug_zinc15_fusilier' \
data.positive_file=../data/train/worlddrug.smi \
data.negative_file=../data/train/zinc15_10K.smi \
train.batch_size=100 \
train.val_mode='5cv' \
train.gpus=1 \
train.num_workers=<num-workers> \
train.epoch=200 \ # recommend.
pu_learning.architecture='Fusilier' \ # 'Fusilier' or 'Liu'
pu_learning.threshold=0.2
```

After PU Learning, the refined negative set is stored for each iteration. In general, use the negative set file of the last iteration.

```bash
├── test/result/gcn_pu_worlddrug_zinc15_fusilier/
    ├── config.yaml     # Model Hyperparameter
    ├── output.log      # Training Log
    ├── iteration_0/    # After the first iteration
    │   ├── negative.smi
    │   └── save.pt
    ├── iteration_1/    # After the second iteration
    │   ├── negative.smi
    │   └── save.pt
    ...
```

After PU learning is finished, train again with the GCN model. In training, we used only 2,833 molecules (size of the positive set) among the refined negative set.

```bash
# Two-class classification model with Worlddrug as positive set and refined ZINC15 as negative set.
python -u train_gcn.py \
name='gcn_worlddrug_zinc15_pu' \
data.positive_file=../data/train/worlddrug.smi \
data.negative_file=result/gcn_pu_worlddrug_zinc15_fusilier/iteration_<last>/negative.smi \
train.batch_size=100 \
train.val_mode='5cv' \
train.gpus=1 \
train.num_workers=<num-workers> \
train.epoch=200
```

## Test

### Scoring

After model training is complete, you can calculate the score of a given molecule using the model method `load_model` and `test`.

```Python
class Model(~~) :
  @classmethod
  def load_model(cls, model_path: str, device: Union[str, torch.device]) -> Model:
    pass
  def test(self, smiles: str) -> float :
    pass
"""
>>> model_path = 'result/rnn_worlddrug/'
>>> model = RNNLM.load_model(model_path, 'cuda:0')
>>> model.test('c1ccccc1')
85.605
>>> model_path = 'result/gcn_worlddrug_zinc15/'
>>> model = GCNModel.load_model(model_path, 'cuda:0')
>>> model.test('c1ccccc1')
0.999
"""
```

`calculate_score.py` is a simple test script I used for our study. It does not support parallel computation. You can choose model with the argument MODEL     (`-m`, `--model`), and you also use QED architecture for scoring theorem with `-m QED`, instead of Deep learning model. 

```bash
python calculate_score.py -m 'QED' -t '../data/test/chembl.smi' -o <output>

python calculate_score.py -m 'result/rnn_worlddrug' -t '../data/test/fda.smi' -o <output>

python calculate_score.py -c -m 'result/rnn_worlddrug' -s 'c1ccccc1'
# output
c1ccccc1,85.605

python calculate_score.py -m 'result/gcn_worlddrug_zinc15' -t 'gdb17'
# output
CC12CCC34CC(N)CN3C=NC14CNC2=N,1.000
CC12C(O)CNC13C1CC(N1)C23,1.000
CC12CC3COC(=O)C3OC1CCNC2CN,1.000
...
```

Argument List.

```
usage: calculate_score.py [-h] [-g] [-c] -m MODEL [-t TEST_FILE] [-s SMILES]
                          [-o OUTPUT]

Calculate Drug-likeness With Model

optional arguments:
  -h, --help            show this help message and exit
  -g, --gpu             use device CUDA, default
  -c, --cpu             use device CPU
  -m MODEL, --model MODEL
                        model path or model architecture(QED)
  -t TEST_FILE, --test_file TEST_FILE
                        test file path
  -s SMILES, --smiles SMILES
                        test smiles
  -o OUTPUT, --output OUTPUT
                        output file. default is STDOUT
```

### Research

#### AUROC

We provided the simple script to measure the classification performance, AUROC. Run this script after scoring is complete.

```bash
python calculate_auroc.py -d 'score/rnn_worlddrug/' -p fda -n gdb17 zinc15 chembl
# output (Due to the difference in seeds, it is different from the results of paper.)
fda.csv	gdb17.csv	0.9681
fda.csv	zinc15.csv	0.9297
fda.csv	chembl.csv	0.8103
```

Argument List.(Ignore argument `OUTPUT`. Not implemented yet)

```
usage: calculate_auroc.py [-h] [-d SCORE_DIR] [-p POSITIVE [POSITIVE ...]]
                          [-n NEGATIVE [NEGATIVE ...]] [-o OUTPUT]

Calculation AUROC score

optional arguments:
  -h, --help            show this help message and exit
  -d SCORE_DIR, --score_dir SCORE_DIR
                        score file directory path
  -p POSITIVE [POSITIVE ...], --positive POSITIVE [POSITIVE ...]
                        positive(drug) set list
  -n NEGATIVE [NEGATIVE ...], --negative NEGATIVE [NEGATIVE ...]
                        negative(non-drug-like) set list
  -o OUTPUT, --output OUTPUT # TODO
                        ROC graph output path(matplotlib required) 
```

TODO
