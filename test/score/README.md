# Result for scoring

Results of the section [Scoring](https://github.com/SeonghwanSeo/DeepDL/tree/master#scoring).

The score result is calculated from the model trained after seed setting. Therefore, it is different from the scoring result used in our paper. Also in the case of QED, I think it is not pretty important, the float points of the score used in the paper and used in the github are different, so the FDA/ChEMBL score is slightly different. (0.551 in github, 0.552 in paper)

For RNNLM (`rnn_worlddrug/`) you will get different results from a given result file due to differences in PubChem data set. However, in the case of the GCN model, the same result will be obtained.

These results are used in the section [research](https://github.com/SeonghwanSeo/DeepDL/tree/master#research).

These files are csv files and follow the format below.

```
#SMILES,score (omitted in file)
CC(=O)Oc1ccccc1C(=O)O,87.830
NC[C@H](CC(=O)O)c1ccc(Cl)cc1,76.905
C[C@@H](C[N+](C)(C)C)OC(N)=O,73.678
CN(C)CC[C@@H](c1ccc(Br)cc1)c1ccccn1,78.359
CN(C)CCO[C@@H](c1ccc(Cl)cc1)c1ccccn1,70.233
CC(C)(C)NC[C@@H](O)COc1cccc2c1CCC(=O)N2,76.590
...
```
