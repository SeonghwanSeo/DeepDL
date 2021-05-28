# Drug-likeness scoring based on unsupervised learning

github for Drug-likeness scoring based on unsupervised learning

## Table of Contents

- [Environment](#environment)
- [Data](#data)
  - [Download PubChem](#prepare-pubchem-data-for-pretrain)
  - [Data Directory Structure](#data-directory-structure)
- [RNNLM](#rnnlm)
- [GCN](#gcn)
- [Test](#test)

## Environment

- Python>=3.7
- [RDKit](https://www.rdkit.org/docs/Install.html)
- [PyTorch](https://pytorch.org/)>=1.7.1
- [Hydra](https://hydra.cc/)>=1.0.6
- [scikit-learn](https://scikit-learn.org/stable/)>=0.24.1

## Data

### Prepare PubChem Data for pretrain

Most of the data we used for training is already uploaded on github(data/).

However, in the case of the PubChem data we used for pre-training of the RNNLM model, I couldn't upload it because it contains 10 million molecules. Therefore, you should download it from [PubChem](https://pubchem.ncbi.nlm.nih.gov/).([Download](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz)) You will get the file `CID-SMILES`.

Since the total number of data is over 100 million, I recommend shuffle the raw file and extract about 12 million lines of it before preprocessing. (ex. command `shuf` of linux)

Our work only used the molecules whose SMILES are RDKit-readable, shorter than 100 characters, and represent a single molecule (SMILES without ‘.’)

```bash
cd <root_dir>/data
wget https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz
gzip -d CID-SMILES.gz

shuf -n 12000000 CID-SMILES > CID-SMILES-SMALL	# optional, but recommend
python preprocessing.py CID-SMILES-SMALL train/pubchem.smi 10000000 <cpus>

rm CID-SMILES      		# optional
rm CID-SMILES-SMALL		# optional
```

### Data Directory Structure

After upper step, the `data/` directory structure will be as follows.

```bash
├── data
    ├── preprocess.py
    ├── train
    │   ├── chembl.smi
    │   ├── gdb17.smi
    │   ├── pubchem.smi
    │   ├── worlddrug.smi
    │   └── zinc.smi
    └── test
        ├── chembl.smi
        ├── fda.smi
        ├── gdb17.smi
        ├── investigation.smi
        └── zinc.smi
```

## RNNLM

Move to the `script/` directory. For RNNLM, training script is `train_rnn.py`, and for GCN, training script is `train_gcn.py`. Config file is `config/rnn.yaml`.

The following commands are the minimum command for implementing our paper, and you can handle more with [hydra override](https://hydra.cc/docs/intro#basic-example).

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
train.init_model=result/rnn_pretrain/
train.batch_size=100 \
train.gpus=<gpus> \
train.num_workers=<num-workers> \
train.epoch=100
```

Both RNNLM and GCN, config, log, and model parameters will be saved in `<root>/script/result/<name>/`.

```bash
├── script
    ├── result
        ├── rnn_pretrain
        │   ├── config.yaml
        │   ├── output.log
        │   └── save.pt
        └── rnn_worlddrug
            ├── config.yaml
            ├── output.log
            └── save.pt
```

## GCN

In gcn learning, we can choose between two validation test mode, normal validation test (`val`) and 5 fold cross-validation test (`5cv`). Default is 5 fold cross-validation test mode. Config file is `config/gcn.yaml`

```bash
# Two-class classification model with Worlddrug as positive set and ZINC as negative set.
python -u train_gcn.py \
name='gcn_worlddrug_zinc' \
data.negative_file=../data/train/zinc.smi \
train.batch_size=100 \
train.valid_mode='5cv' \
train.gpus=<gpus> \
train.num_workers=<num-workers> \
train.epoch=200 # recommend.

# Two-class classification model with Worlddrug as positive set and ChEMBL as negative set.
python -u train_gcn.py \
name='rnn_worlddrug_chembl' \
data.negative_file=../data/train/chembl.smi \
train.batch_size=100 \
train.valid_mode='5cv' \
train.gpus=<gpus> \
train.num_workers=<num-workers> \
train.epoch=100 # recommend

# Two-class classification model with Worlddrug as positive set and GDB17 as negative set.
python -u train_gcn.py \
name='rnn_worlddrug_chembl' \
data.negative_file=../data/train/gdb17.smi \
train.batch_size=100 \
train.valid_mode='5cv' \
train.gpus=<gpus> \
train.num_workers=<num-workers> \
train.epoch=100 # recommend
```



## Test
