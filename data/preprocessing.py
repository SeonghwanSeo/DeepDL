from rdkit import Chem, RDLogger
from multiprocessing import Pool
import sys
import random
RDLogger.DisableLog('rdApp.*')

def main(datafile, outputfile, max_size, cpus) :
    with open(datafile) as f :
        lines = f.readlines()
    p = Pool(cpu)
    smiles_list = list(set(p.map(filtering, lines)))
    p.terminate()
    p.join()
    
    random.shuffle(smiles_list)
    print("size:", len(smiles_list))
    t = 0
    with open(outputfile, 'w') as w :
        for smiles in smiles_list :
            if smiles is not None :
                w.write(smiles + '\n')
                t += 1
            if t == max_size :
                break

def filtering(line) :
    smiles = line.strip().split('\t')[-1]
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None :
        try :
            s = Chem.MolToSmiles(mol)
            if len(s) > 100 or '.' in s :
                return None
            else :
                return s
        except:
            return None
    else :
        return None

if __name__ == '__main__' :
    datafile = sys.argv[1]
    outputfile = sys.argv[2]
    max_size = int(sys.argv[3])
    cpu = int(sys.argv[4])
    main(datafile, outputfile, max_size, cpu)
