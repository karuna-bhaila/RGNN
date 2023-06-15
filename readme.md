# RGNN

Repository for Local Differential Privacy in Graph Neural Networks: a Reconstruction Approach

## Run
The main code to train/test is in `main.py`.
The default experiment setting is 5 runs with different random seeds.
For the semi-synthetic datasets, the pickle files are available in the `datasets` folder. 
Code for pre-processing these datasets are in `data.py`. 
All other datasets are automatically downloaded.

### Dataset and Model
| Argument      | Description             |
|---------------|-------------------------|
| dataset		| Name of dataset (citeseer, cora, dblp, facebook, german, student)|
| cols_to_group	| No. of feature columns to group to reduce feature matrix sparsity|
| model			| GNN architecture (sage, gat, gcn) |


### LDP
| Argument      | Description             |
|---------------|-------------------------|
| x_eps			| Privacy budget for one feature |
| m 			| No. of features to sample for GRR_FS |
| y_eps			| Label privacy budget |

### Reconstruction and LLP
| Argument      | Description             |
|---------------|-------------------------|
| x_hops		| No. of hops for feature propagation during reconstruction |
| y_hops		| No. of hops for label propagation during reconstruction |
| num_clusters	| No. of clusters for graph partitioning for LLP loss |
| alpha			| Hyperparameter to control influence of LLP loss |