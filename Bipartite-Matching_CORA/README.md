# Bipartite Matching

Bipartite Matching experiment without diversity constraints, from:

Wilder, B., Dilkina, B., & Tambe, M. (2019, July). Melding the data-decisions pipeline: Decision-focused learning for combinatorial optimization.
In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 33, No. 01, pp. 1658-1665).


## Data generation

1. Run `get_data.sh` to download preprocessed CORA files.
   You can also directly download preprocessed data, in which case make sure to delete the `make_cora_dataset.py` from the zip.
2. Run `make_cora_dataset.py´ to build bipartite graphs and feature vectors associated with each node. 

Data is now available in the `data` directory as numpy arrays saved on disk.

## Description 

From the original README:
> The ground truth is a (27 x 2500) numpy matrix. Each row is an instance. This row contains the flattened adjacency matrix for the 50 nodes in the instance, which is a binary indicator for whether each of the 2500 possible edges is present. The features contain a (27 x 2500 x 2866) numpy matrix where the ij entry is a feature vector for edge j in instance i. This is the concatenation of the bag of words features for the nodes that the potential edge connects.

`true_cost.npy`: (27 x 2500) ground truth

`features.npy`: (27 x 2500 x 2866) concatenated bag of words features