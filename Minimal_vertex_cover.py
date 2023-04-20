# CSE 282 project
# data is read from the binarized matrix

import numpy as np
import pandas as pd
import random
from numpy.random import choice
import matplotlib.pyplot as plt
import time

def hypergraph_index(hypergraph_dict):
    hg_index = {}
    for i_edge, edge in hypergraph_dict.items():
        edge_l = len(edge)
        for i in range(edge_l):
            node_1 = edge[i]
            if node_1 not in hg_index:
                hg_index[node_1] = [i_edge]
            else:
                hg_index[node_1].append(i_edge)
    return hg_index

def node_degree(hypergraph_dict):
    all_nodes = set()
    for x in hypergraph_dict.values():
        all_nodes.update(x)
    node_deg = {n:0 for n in all_nodes}
    for key, node_list in hypergraph_dict.items():
        for node in node_list:
            node_deg[node] += 1
    return node_deg

def get_largest_node(node_deg):
    node_deg_sorted = sorted([(n, c) for n, c in node_deg.items()], key = lambda x:x[1], reverse = True)
    node = node_deg_sorted[0][0]
    deg = node_deg_sorted[0][1]
    if deg > 0:
        return node, deg

def remove_node(hypergraph_dict, hg_index, node_deg, node):
    index_set = hg_index[node]
    for index in index_set:
        for n in hypergraph_dict[index]:
            node_deg[n] -= 1
        hypergraph_dict[index] = []

def get_edge_list(df):
    edge_list = {}
    count = 0
    for i in range(len(df)):
        for j in range(i, len(df)):
            if df[i][j] == 1:
                edge_list[count] = (i,j)
                count += 1
    return edge_list

def get_lethal_pairs_dist(hyperedge_dict):
    node_deg = node_degree(hyperedge_dict)
    dist = {}
    sum_degs = 0
    for node in node_deg:
        degs = node_deg[node]
        sum_degs += degs
    for node in node_deg:
        dist[node] = node_deg[node] / sum_degs
    return dist

def finished(hypergraph_dict):
    fin = True
    for key in hypergraph_dict:
        if len(hypergraph_dict[key]) > 0:
            fin = False
            break
    return fin

# --------------------------------------------------------------------------------------------------------- #
# read the data and convert it to a symmetric matrix:
df = pd.read_table('./binarized_genetic_interaction_data_threshold_0.05.txt', "\t", header = None)
df_array = df.values

# make the matrix symmetric:
df_symmetric = np.maximum(df_array, df_array.T)

# check if there is any gene with degree zero:
for i in range(df_symmetric.shape[0]):
    if df_symmetric.sum(axis = 1)[i] == 0:
        row_del = i

df_symmetric_updated = np.delete(df_symmetric, row_del, 0)
df_symmetric_updated = np.delete(df_symmetric_updated, row_del, 1)
df_symmetric = df_symmetric_updated

# --------------------------------------------------------------------------------------------------------- #
# find the minimal vertex cover of the lethal pairs:
edge_list = get_edge_list(df_symmetric)
print('number of lethal pairs:', len(edge_list))

start = time.time()
node_deg = node_degree(edge_list)
hg_index = hypergraph_index(edge_list)
vc_set_lethal_pairs = set()
while not finished(edge_list):
    node_max, deg_max = get_largest_node(node_deg)
    vc_set_lethal_pairs.update([node_max])
    remove_node(edge_list, hg_index, node_deg, node_max)
# print(vc_set_lethal_pairs)
end = time.time()
print('length of minimal vertex cover for lethal pairs:', len(vc_set_lethal_pairs))
print('time taken:', (end - start))
print('------------------------------')

# --------------------------------------------------------------------------------------------------------- #
# generate 100 random graphs with the same number of nodes, edges and degree distribution:
edge_list = get_edge_list(df_symmetric)
lethal_pairs_dist = get_lethal_pairs_dist(edge_list)
nodes_list = list(lethal_pairs_dist.keys())
probs_list = list(lethal_pairs_dist.values())

pairs_dist = {}
sum = 0
for i in range(len(nodes_list)):
    for j in range((i+1),len(nodes_list)):
        node1 = nodes_list[i]
        node2 = nodes_list[j]
        pairs_dist[(node1,node2)] = lethal_pairs_dist[node1]*lethal_pairs_dist[node2]
        sum += lethal_pairs_dist[node1]*lethal_pairs_dist[node2]

pairs_dist_normalized = {}
for i in pairs_dist:
    pairs_dist_normalized[i] = pairs_dist[i]/sum

pairs_list = list(pairs_dist_normalized.keys())
pairs_prob = list(pairs_dist_normalized.values())

pairs_length = 318757
num = 1
vc_pairs = []
while num <= 100:
    sampled_index = np.random.choice(range(len(pairs_list)), p = pairs_prob, size = pairs_length, replace = False)
    hg_dict = {}
    for i in range(len(sampled_index)):
        hg_dict[i] = pairs_list[sampled_index[i]]

    hg_index = hypergraph_index(hg_dict)
    node_deg_hypergraph = node_degree(hg_dict)

    vc_set = set()
    while not finished(hg_dict):
       node_max, deg_max = get_largest_node(node_deg_hypergraph)
       vc_set.update([node_max])
       remove_node(hg_dict, hg_index, node_deg_hypergraph, node_max)
    vc_pairs.append(len(vc_set))
    num += 1
print('length of minimal vertex cover for 100 trials:', vc_pairs)
print('------------------------------')

# --------------------------------------------------------------------------------------------------------- #
# sampling triples with three methods:
print('Method 1 for Sampling triples ...')
vc_1st_method = []
pairs_length = 318757
p = 0.01
while p <= 0.5:
    triples = []
    n = int(pairs_length*p)
    np.random.seed(0)
    for i in range(n):
        sampled_triples = choice(nodes_list, size = 3, replace = False)
        triples.append(sampled_triples)

    hg_dict = {}
    for i in range(len(triples)):
        hg_dict[i] = tuple(triples[i])
    len_triples = len(triples)

    for i in range(len(edge_list)):
        hg_dict[i + len_triples] = edge_list[i]

    hg_index = hypergraph_index(hg_dict)
    node_deg_hypergraph = node_degree(hg_dict)

    vc_set = set()
    while not finished(hg_dict):
       node_max, deg_max = get_largest_node(node_deg_hypergraph)
       vc_set.update([node_max])
       remove_node(hg_dict, hg_index, node_deg_hypergraph, node_max)
    vc_1st_method.append(len(vc_set))
    p += 0.01
print('Done!')


print('Method 2 for sampling triples ...')
vc_2nd_method = []
p = 0.01
while p <= 0.5:
    triples = []
    n = int(pairs_length*p)
    np.random.seed(0)
    for i in range(n):
        sampled_triples = choice(nodes_list, p = probs_list, size = 3, replace = False)
        triples.append(sampled_triples)

    hg_dict = {}
    for i in range(len(triples)):
        hg_dict[i] = tuple(triples[i])
    len_triples = len(triples)

    for i in range(len(edge_list)):
        hg_dict[i + len_triples] = edge_list[i]

    hg_index = hypergraph_index(hg_dict)
    node_deg_hypergraph = node_degree(hg_dict)

    vc_set = set()
    while not finished(hg_dict):
       node_max, deg_max = get_largest_node(node_deg_hypergraph)
       vc_set.update([node_max])
       remove_node(hg_dict, hg_index, node_deg_hypergraph, node_max)
    vc_2nd_method.append(len(vc_set))
    p += 0.01
print('Done!')


print('Method 3 for sampling triples ...')
nodes_list_updated = np.delete(nodes_list, list(vc_set_lethal_pairs))
# print(len(nodes_list_updated))
vc_3rd_method = []
p = 0.01
while p <= 0.5:
    triples = []
    n = int(pairs_length*p)
    np.random.seed(0)
    for i in range(n):
        sampled_triples = choice(nodes_list_updated, size = 3, replace = False)
        triples.append(sampled_triples)

    hg_dict = {}
    for i in range(len(triples)):
        hg_dict[i] = tuple(triples[i])
    len_triples = len(triples)

    hg_index = hypergraph_index(hg_dict)
    node_deg_hypergraph = node_degree(hg_dict)

    vc_set = set()
    while not finished(hg_dict):
       node_max, deg_max = get_largest_node(node_deg_hypergraph)
       vc_set.update([node_max])
       remove_node(hg_dict, hg_index, node_deg_hypergraph, node_max)
    vc_set.update(vc_set_lethal_pairs)
    vc_3rd_method.append(len(vc_set))
    p += 0.01
print('Done!')
# print('length of minimal vertex covers using the 1st method', vc_1st_method)
# print('length of minimal vertex covers using the 2nd method', vc_2nd_method)
# print('length of minimal vertex covers using the 3rd method', vc_3rd_method)

plt.plot(np.linspace(0.01,0.5,49), vc_1st_method, '-o', color = '#4472C4', label = '1st method')
plt.plot(np.linspace(0.01,0.5,49), vc_2nd_method, '-o', color = '#ED7D31', label = '2nd method')
plt.plot(np.linspace(0.01,0.5,49), vc_3rd_method, '-o', color = '#70AD47', label = '3rd method')
plt.ylim([0, 4500])
plt.legend(loc = "upper left")
plt.show()
