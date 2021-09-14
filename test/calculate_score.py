from argparse import ArgumentParser
import sys
import os
import logging
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem.Descriptors import qed

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'src'))
from models import RNNLM, GCNModel
import utils as UTILS

"""
i) When the argument MODEL is path for directory, we use deep learning model for scoring.
ex)
--model=../result/rnn_worlddrug/
root/
  - result/
    - rnn_worlddrug/
      - config.yaml     required
      - output.log
      - save.pt         required

example: 
>> python calculate_score.py -g -m result/worlddrug_rnn/ -t FDA -o <output>

ii) When the argument MODEL is QED, we use QED scoring theorem.
--model=QED

example: 
>> python calculate_score.py -c -m QED -t ../data/test/fda.smi -o <output>

=================
The argument 'test_file' is path for testset files, or their alias.
For the testsets used in our study, I set simple aliases for them. (See TEST_FILE_LIST, below directly)

* The two commands below do the same thing. *
>> python calculate_score.py -g -m result/worlddrug_rnn/ -t fda -o <output>
>> python calculate_score.py -g -m result/worlddrug_rnn/ -t ../data/test/fda.smi -o <output>
"""

TEST_FILE_LIST = {
    'chembl': '../data/test/chembl.smi',
    'fda': '../data/test/fda.smi',
    'gdb17': '../data/test/gdb17.smi',
    'invest': '../data/test/investigation.smi',
    'investigation': '../data/test/investigation.smi',
    'zinc15': '../data/test/zinc15.smi',
}

class QED_model (object):
    @staticmethod
    def test (smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        return qed(mol)

parser = ArgumentParser(description='Calculate Drug-likeness With Model')
parser.add_argument('-g', '--gpu', dest='cuda', action='store_true', help='use device CUDA, default')
parser.add_argument('-c', '--cpu', dest='cuda', action='store_false', help='use device CPU')
parser.set_defaults(cuda=True)
parser.add_argument('-m', '--model', required=True, type=str, help='model path or model architecture(QED)')
parser.add_argument('-t', '--test_file', type=str, help='test file path', default = None)
parser.add_argument('-s', '--smiles', type=str, help='test smiles', default = None)
parser.add_argument('-o', '--output', type=str, help='output file. default is STDOUT', default='stdout')
args = parser.parse_args()

# Set up
assert (args.test_file or args.smiles), "Neither TEST_FILE nor SMILES exist."
if args.test_file != None :
    test_file = TEST_FILE_LIST.get(args.test_file, args.test_file)
    with open(test_file) as f :
        test_smiles = [l.strip() for l in f.readlines()]
else :
    test_smiles = [args.smiles]

device = "cuda:0" if args.cuda else 'cpu'
output_file = args.output

Logger = logging.getLogger()
Logger.setLevel(logging.INFO)
if output_file == 'stdout' :
    Logger.addHandler(logging.StreamHandler())
else :
    Logger.addHandler(logging.FileHandler(output_file, 'w'))

if args.model == 'QED' or args.model == 'qed' :
    model = QED_model()
else :
    model_path = args.model
    model_config_file = os.path.join(model_path, 'config.yaml')
    config = OmegaConf.load(model_config_file)
    model_architecture = config.model.model # RNNLM or GCNModel

    if model_architecture == 'RNNLM' :
        model = RNNLM.load_model(model_path, device)
    elif model_architecture == 'GCNModel' :
        model = GCNModel.load_model(model_path, device)
    else :
        logging.warning("ERR: Not Allowed Model Architecture")
        exit(1)

# Run
for smiles in test_smiles :
    score = model.test(smiles)
    logging.info(f'{smiles},{score:.3f}')
