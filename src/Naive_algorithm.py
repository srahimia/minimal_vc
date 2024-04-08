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

def remove_node(hypergraph_dict, hg_index, node):
    index_set = hg_index[node]
    for index in index_set:
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
hg_index = hypergraph_index(edge_list)
vc_set_lethal_pairs = set()
while not finished(edge_list):
    node_deg = node_degree(edge_list)
    node_max, deg_max = get_largest_node(node_deg)
    vc_set_lethal_pairs.update([node_max])
    remove_node(edge_list, hg_index, node_max)
# print(vc_set_lethal_pairs)
end = time.time()
print('length of minimal vertex cover for lethal pairs:', len(vc_set_lethal_pairs))
print('time taken:', (end - start))
