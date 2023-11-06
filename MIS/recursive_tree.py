import torch
import sys
import random

from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from tqdm import tqdm

import model
import warnings
import os
import math
import scipy.special
import networkx as nx
import pickle
import buffer
import numpy as np
from heuristic_optimal_solvers import greedy_MIS

random.seed(33)


class datasample:
    def __init__(self, G1, G2, device):
        self.G1 = G1
        self.G2 = G2
        self.target = self.create_target(device)

    def create_target(self, device):
        if self.G1.nodes[0]['MIS'] > self.G2.nodes[0]['MIS']:
            return (torch.tensor([[1.0, 0.0]])).to(device)
        elif self.G1.nodes[0]['MIS'] < self.G2.nodes[0]['MIS']:
            return (torch.tensor([[0.0, 1.0]])).to(device)
        else:
            return (torch.tensor([[0.5, 0.5]])).to(device)
        


def build_tree(G, cmp, device):
    list_graphs = []
    while 1:
        non_indip_list = get_non_independent_set(G) 
        if not non_indip_list:  
            break
        node = random.choice(non_indip_list)  

        # define G1 as G without the node 'node'
        G1 = G.copy()
        G1.remove_node(node)
        G1 = nx.convert_node_labels_to_integers(G1)


        # define G2 as G without the neighborhoods of 'node'
        G2 = G.copy()
        G2.remove_nodes_from([n for n in G.neighbors(node)])
        G2 = nx.convert_node_labels_to_integers(G2)

        with torch.no_grad():
            G1_mat = torch.tensor(nx.adjacency_matrix(G1).todense(), dtype=torch.float).to(device)
            G2_mat = torch.tensor(nx.adjacency_matrix(G2).todense(), dtype=torch.float).to(device)
            index = torch.argmax(cmp.forward(G1_mat, G2_mat, device)).item()

        if index == 0:
            G = G1.copy()
        else:
            G = G2.copy()


        list_graphs.append(G1)
        list_graphs.append(G2)

        
    return list_graphs, G.number_of_nodes()

def find_MIS_value(cmp,adj,device):
    while(1):

        #get non-isaolated nodes
        sum_rows = np.sum(adj,axis=1)
        non_indep_nodes = np.nonzero(sum_rows)[0].tolist()
        if not non_indep_nodes:  #stop if all the nodes are in the independent set
            break

        node = random.choice(non_indep_nodes) #pick a random node in G that is not in the independent set

        #consider the graph without the node 'node'
        adj_1 = np.delete(adj,node,axis=0)
        adj_1 = np.delete(adj_1,node,axis=1)

        #consider the graph without the neighbors of 'node'
        list_neighbors = np.nonzero(adj[node])[0].tolist()
        adj_2 = np.delete(adj,list_neighbors,axis=0)
        adj_2 = np.delete(adj_2,list_neighbors,axis=1)

        with torch.no_grad():
            G1_mat = torch.tensor(adj_1.tolist(), dtype=torch.float).to(device)
            G2_mat = torch.tensor(adj_2.tolist(), dtype=torch.float).to(device)
            index = torch.argmax(cmp.forward(G1_mat, G2_mat, device)).item()

        if index == 0:
            adj = adj_1
        else:
            adj = adj_2

    return adj.shape[0]

def get_non_independent_set(G):
    list_indep = []
    for n in G.nodes:
        if nx.is_isolate(G, n) == False:
            list_indep.append(n)
    return list_indep

def run_several_times(list_graphs, times, cmp, device, mixed_rollout='False'):
    for G in list_graphs:
        
        if mixed_rollout == 'True':
            max_val = greedy_MIS(G)
        elif mixed_rollout == 'False':
            max_val = -1
        else:
            sys.exit('mixed_rollout has to be a string equal to either True or False')
        for n in range(times):
            val = find_MIS_value(cmp, nx.to_numpy_array(G), device)
            if val > max_val:
                max_val = val
        nx.set_node_attributes(G, max_val, 'MIS')

    return list_graphs


def create_list_datasamples(list_graphs, device, num_samples):
    list_datasamples = []
    for i in range(len(list_graphs)):
        G1 = list_graphs[i]
        for j in range(i + 1, len(list_graphs)):
            if len(list_datasamples) == num_samples:
                return list_datasamples
            G2 = list_graphs[j]
            list_datasamples.append(datasample(G1, G2, device))

    return list_datasamples

def create_dataset_buffer(cmp, device, times, dim_datasamples, list_G,mixed_rollout='False',show_graphs_stats=True):
    list_val = []
    list_dataset = []

    num = find_max_list_graph(dim_datasamples)

    for G in list_G:
        if show_graphs_stats:
            print(f'Density: {G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1)) * 2:<.3f}', G,sep=' | ')

        list_graphs, val = build_tree(G, cmp, device)  
        list_val.append(val)

        if len(list_graphs) > num:
            list_graphs = list_graphs[0:num]

        list_graphs = run_several_times(list_graphs, times, cmp, device,mixed_rollout=mixed_rollout)

        samples = create_list_datasamples(list_graphs, device, dim_datasamples)

        if len(samples) > dim_datasamples:
            list_dataset = list_dataset + samples[0:dim_datasamples]
        else:
            list_dataset = list_dataset + samples

    return list_dataset, list_val

def find_max_list_graph(max_num):
    i = 5
    val = -1
    while (val < max_num):
        val = scipy.special.binom(i, 2)
        i += 1
    return i - 1

