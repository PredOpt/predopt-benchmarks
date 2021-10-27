"""
    Code to build data for the Bipartite Matching experiment from the paper: 
    Wilder, B., Dilkina, B., & Tambe, M. (2019, July). Melding the data-decisions pipeline: Decision-focused learning for combinatorial optimization.
    In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 33, No. 01, pp. 1658-1665).
    
    Adapted from https://github.com/bwilder0/aaai_melding_code 

    Download preprocessed CORA  data from https://bryanwilder.github.io/files/data_decisions_benchmarks.zip)
"""

import numpy as np
import networkx as nx
import pickle
import os
os.chdir('benchmarks_release')

g = nx.read_edgelist('cora.cites')
nodes_before = [int(v) for v in g.nodes()]

g = nx.convert_node_labels_to_integers(g, first_label=0)
a = np.loadtxt('cora_cites_metis.txt.part.27')
g_part = []
for i in range(27):
    g_part.append(nx.Graph(nx.subgraph(g, list(np.where(a == i))[0])))

nodes_available = []
for i in range(27):
    if len(g_part[i]) > 100:
        degrees = [g_part[i].degree(v) for v in g_part[i]]
        order = np.argsort(degrees)
        nodes = np.array(g_part[i].nodes())
        num_remove = len(g_part[i]) - 100
        to_remove = nodes[order[:num_remove]]
        g_part[i].remove_nodes_from(to_remove)
        nodes_available.extend(to_remove)

for i in range(27):
    if len(g_part[i]) < 100:
        num_needed =  100 - len(g_part[i])
        to_add = nodes_available[:num_needed]
        nodes_available = nodes_available[num_needed:]
        g_part[i].add_nodes_from(to_add)
    

for i in range(27):
    g_part.append(nx.subgraph(g, list(np.where(a == i))[0]))

features = np.loadtxt('cora.content')
features_idx = features[:, 0]
features = features[:, 1:]

# nodes per bigraph
n_nodes = 50
# store bigraphs as adj. matrices
Ps = np.zeros((27, n_nodes**2))
n_features = 1433
# features 
data = np.zeros((27, n_nodes**2, 2*n_features))
# preprocessed partitions from CORA
partition = pickle.load(open('cora_partition.pickle', 'rb'))


percent_removed = []
for i in range(27):
    lhs_nodes, rhs_nodes = partition[i]
    lhs_nodes_idx = []
    rhs_nodes_idx = []
    gnodes = list(g_part[i].nodes())
    
    to_add = set([i for i in range(len(gnodes))])
    for v in lhs_nodes:
        try:
            lhs_nodes_idx.append(gnodes.index(v))
            to_add.remove(gnodes.index(v))
        except:
            print(v, ' not in lhs list')
    for v in rhs_nodes:
        try:
            rhs_nodes_idx.append(gnodes.index(v))
            to_add.remove(gnodes.index(v))
        except:
            print(v, ' not in list')
    incomplete_list = lhs_nodes_idx if len(lhs_nodes_idx) < len(rhs_nodes_idx) else rhs_nodes_idx
        
    while len(incomplete_list) < 50:
        misidx = to_add.pop()
        print('node {} idx added successfully'.format(gnodes[misidx]))
        incomplete_list.append(misidx)
    assert len(lhs_nodes_idx) == len(rhs_nodes_idx)
    adj = nx.to_numpy_array(g_part[i])
    sum_before = adj.sum()
    adj = adj[lhs_nodes_idx]
    adj = adj[:, rhs_nodes_idx]
    edges_before = sum_before/2
    print(sum_before/2, adj.sum())
    percent_removed.append((edges_before - adj.sum())/edges_before)
    Ps[i] = adj.flatten()
     
    node_ids_lhs = [nodes_before[v] for v in lhs_nodes]
    node_ids_rhs = [nodes_before[v] for v in rhs_nodes]
    curr_data_idx = 0
    for j, nid in enumerate(node_ids_lhs):
        row_idx_j = int(np.where(features_idx == nid)[0][0])
        for k, nid_other in enumerate(node_ids_rhs):
            row_idx_k = int(np.where(features_idx == nid_other)[0][0])
            data[i, curr_data_idx, :n_features] = features[row_idx_j]
            data[i, curr_data_idx, n_features:] = features[row_idx_k]
            curr_data_idx += 1

datadir = os.path.join('..','data')
if not os.path.exists(datadir):
    os.makedirs(datadir)
np.save(os.path.join(datadir, 'true_cost.npy'), Ps)
np.save(os.path.join(datadir, 'features.npy'), data)