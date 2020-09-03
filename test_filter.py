import numpy as np
import random
import pickle
import argparse
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP, MolMR
from rdkit.Chem.rdMolDescriptors import CalcNumRings,CalcNumRotatableBonds,CalcTPSA
from rdkit.Chem.Descriptors import ExactMolWt,MolLogP,NumHDonors,NumHAcceptors
#STOCK

parser = argparse.ArgumentParser()
parser.add_argument("--positive_file", help="data file", type=str)
parser.add_argument("--negative_file", help="data file", type=str)

args = parser.parse_args()

print("positive set     :", args.positive_file)
print("negative set     :", args.negative_file)
print()

with open(args.positive_file, 'r') as f:
    lines = f.readlines()
    yes = [l.strip().split('\t')[1] for l in lines]

with open(args.negative_file, 'r') as f:
    lines = f.readlines()
    no = [l.strip().split('\t')[1] for l in lines]

nopass=[0,0,0,0,0,0,0,0]
yespass=[0,0,0,0,0,0,0,0]

for l in no:
    m=Chem.MolFromSmiles(l)
    if (-0.4<=MolLogP(m)<=5.6 and 40<=MolMR(m)<=130 and 160<=ExactMolWt(m)<=480 and 20<=m.GetNumAtoms()<=70 and CalcTPSA(m)<140):   nopass[0]+=1#ghose filter
    if (CalcTPSA(m)<=140 and CalcNumRotatableBonds(m)<=10):    nopass[1]+=1#veber rule
    if (CalcNumRings(m)>=3 or CalcNumRotatableBonds(m)>=6 or (m.GetNumBonds()-CalcNumRotatableBonds(m))>=18):  nopass[2]+=1#mddr-like rules
    if (ExactMolWt(m)<=500 and MolLogP(m)<=4.5 and NumHDonors(m)<=5 and NumHAcceptors(m)<=10):  nopass[3]+=1#ro5
    if (ExactMolWt(m)<300 and NumHDonors(m)<=3 and NumHAcceptors(m)<=3 and MolLogP(m)<=3): nopass[4]+=1
    if (MolLogP(m)<4 and ExactMolWt(m)<400): nopass[5]+=1
    if (MolLogP(m)<3 and CalcTPSA(m)>75): nopass[6]+=1
    if (0<=CalcTPSA(m)<=132 and -1<=MolLogP(m)<=6): nopass[7]+=1
for l in yes:
    m=Chem.MolFromSmiles(l)
    if (-0.4<=MolLogP(m)<=5.6 and 40<=MolMR(m)<=130 and 160<=ExactMolWt(m)<=480 and 20<=m.GetNumAtoms()<=70 and CalcTPSA(m)<140):   yespass[0]+=1#ghose filter
    if (CalcTPSA(m)<=140 and CalcNumRotatableBonds(m)<=10):    yespass[1]+=1#veber rule
    if (CalcNumRings(m)>=3 or CalcNumRotatableBonds(m)>=6 or (m.GetNumBonds()-CalcNumRotatableBonds(m))>=18):  yespass[2]+=1#mddr-like rules
    if (ExactMolWt(m)<=500 and MolLogP(m)<=4.5 and NumHDonors(m)<=5 and NumHAcceptors(m)<=10):  yespass[3]+=1#ro5
    if (ExactMolWt(m)<300 and NumHDonors(m)<=3 and NumHAcceptors(m)<=3 and MolLogP(m)<=3): yespass[4]+=1
    if (MolLogP(m)<4 and ExactMolWt(m)<400): yespass[5]+=1
    if (MolLogP(m)<3 and CalcTPSA(m)>75): yespass[6]+=1
    if (0<=CalcTPSA(m)<=132 and -1<=MolLogP(m)<=6): yespass[7]+=1

nopass=np.array(nopass)/len(no)
yespass=np.array(yespass)/len(yes)
with open('roc_result/filter/'+args.positive_file.strip().split('/')[-1]+'_'+args.negative_file.strip().split('/')[-1]+'.pkl','wb') as w:
    pickle.dump((nopass,yespass),w)
print(f'Ghose       : tpr= {yespass[0]:.4f}\tfpr={nopass[0]:.4f}')
print(f'Veber       : tpr= {yespass[1]:.4f}\tfpr={nopass[1]:.4f}')
print(f'MDDR-like   : tpr= {yespass[2]:.4f}\tfpr={nopass[2]:.4f}')
print(f'Ro5         : tpr= {yespass[3]:.4f}\tfpr={nopass[3]:.4f}')
print(f'Ro3         : tpr= {yespass[4]:.4f}\tfpr={nopass[4]:.4f}')
print(f'Hsk4/400    : tpr= {yespass[5]:.4f}\tfpr={nopass[5]:.4f}')
print(f'Pfizer3/75  : tpr= {yespass[6]:.4f}\tfpr={nopass[6]:.4f}')
print(f'Egan        : tpr= {yespass[7]:.4f}\tfpr={nopass[7]:.4f}')
