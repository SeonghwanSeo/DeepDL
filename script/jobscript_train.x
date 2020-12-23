#!/bin/bash
#PBS -N ssh_train
#PBS -l nodes=gnode3:ppn=16
#PBS -l walltime=1000:00:00

cd $PBS_O_WORKDIR
echo `cat $PBS_NODEFILE`
cat $PBS_NODEFILE
NPROCS=`wc -l < $PBS_NODEFILE`

date
source ~/.bashrc
source activate seonghwan-python-3.6
export OMP_NUM_THREADS=1

python -u train1.py --config=config/gat_train.yaml --epoch=1000 --save_dir=result/gat > result/gat/log

date
