import torch
import sys
import random

import os
import scipy.special
import networkx as nx
import pickle
import numpy as np

random.seed(33)


class datasample:
    def __init__(self, G1, G2, device):
        self.G1 = G1
        self.G2 = G2
        self.target = self.create_target(device)

    def create_target(self, device):
        if self.G1.nodes[0]['MVC'] > self.G2.nodes[0]['MVC']:
            return (torch.tensor([[0.0, 1.0]])).to(device)
        elif self.G1.nodes[0]['MVC'] < self.G2.nodes[0]['MVC']:
            return (torch.tensor([[1.0, 0.0]])).to(device)
        else:
            return (torch.tensor([[0.5, 0.5]])).to(device)
        




def build_tree(cmp,adj,device,flag_density = 'False'):
    N = adj.shape[0] #original shape of the adjacency matrix
    list_G = []
    while(1):
        #STEP 1: check if the while loop should stop and eventually pick a random node 
        list_nodes = node_list_for_MVC(adj,N)
        if not list_nodes:
            break
        node = random.choice(list_nodes)

        #STEP 2: create the two graphs (with 'node' and without 'node')

        #create G1
        G1 = np.pad(adj, ((0,1),(0,1)), mode='constant') #add new node
        for i in range(N): #remove the edges 
            if G1[node][i] == 1:
                G1[node][i] = 0
                G1[i][node] = 0
        G1[node][-1] = 1
        G1[-1][node] = 1
        grafo = nx.from_numpy_array(G1[:N,:N])
        if flag_density == 'False':
            list_G.append(grafo)
        elif flag_density == 'True':
            if nx.density(grafo) > 0.02:
                list_G.append(grafo)
        else:
            sys.exit('flag_density has to be a string (not bool) and either True or False')
        

        #create G2
        list_neighbors = np.nonzero(adj[node])[0].tolist()
        G2 = np.copy(adj)
        for n in list_neighbors:
            G2 = np.pad(G2, ((0,1),(0,1)), mode='constant') #add new node
            for i in range(N):
                if G2[n][i] == 1:
                    G2[n][i] = 0
                    G2[i][n] = 0
            G2[n][-1] = 1
            G2[-1][n] = 1
        grafo = nx.from_numpy_array(G2[:N,:N])
        if flag_density == 'False':
            list_G.append(grafo)
        elif flag_density == 'True':
            if nx.density(grafo) > 0.02:
                list_G.append(grafo)
        else:
            sys.exit('flag_density has to be a string (not bool) and either True or False')
        


        #STEP 3: let the comparator choose
        with torch.no_grad():
            G1_torch = torch.tensor(G1.tolist(), dtype=torch.float).to(device)
            G2_torch = torch.tensor(G2.tolist(), dtype=torch.float).to(device)
            index = torch.argmax(cmp.forward(G1_torch, G2_torch, device)).item()


        if index == 0:
            adj = G1
        else:
            adj = G2

    return list_G, np.sum(adj)/2

def find_MVC_value(cmp,adj,device):
    N = adj.shape[0] #original shape of the adjacency matrix
    while(1):
        #STEP 1: check if the while loop should stop and eventually pick a random node 
        list_nodes = node_list_for_MVC(adj,N)
        if not list_nodes:
            break
        node = random.choice(list_nodes)

        #STEP 2: create the two graphs (with 'node' and without 'node')

        #create G1
        G1 = np.pad(adj, ((0,1),(0,1)), mode='constant') #add new node
        for i in range(N): #remove the edges 
            if G1[node][i] == 1:
                G1[node][i] = 0
                G1[i][node] = 0
        G1[node][-1] = 1
        G1[-1][node] = 1

        #create G2
        list_neighbors = np.nonzero(adj[node])[0].tolist()
        G2 = np.copy(adj)
        for n in list_neighbors:
            G2 = np.pad(G2, ((0,1),(0,1)), mode='constant') #add new node
            for i in range(N):
                if G2[n][i] == 1:
                    G2[n][i] = 0
                    G2[i][n] = 0
            G2[n][-1] = 1
            G2[-1][n] = 1

        #STEP 3: let the comparator choose
        with torch.no_grad():
            G1_torch = torch.tensor(G1.tolist(), dtype=torch.float).to(device)
            G2_torch = torch.tensor(G2.tolist(), dtype=torch.float).to(device)
            index = torch.argmax(cmp.forward(G1_torch, G2_torch, device)).item()

        if index == 0:
            adj = G1
        else:
            adj = G2

    return np.sum(adj)/2

def node_list_for_MVC(adj,N):
    list_for_MVC = []
    for i in range(N):
        val = np.sum(adj[i])
        if val >= 2:
            list_for_MVC.append(i)
        elif val == 1:
            idx_neigh = np.argmax(adj[i])
            val_neigh = np.sum(adj[idx_neigh])
            if val_neigh > 1:
                list_for_MVC.append(i)
            elif val_neigh == 0:
                sys.exit('Not possible for a node to have a neighbor without neighbors')
    return list_for_MVC

        
def run_several_times(list_graphs, times, cmp, device):
    cnt = 0
    for G in list_graphs:
        min_val = G.number_of_nodes()*2

        for n in range(times):
            val = find_MVC_value(cmp, nx.to_numpy_array(G), device)
            if val < min_val:
                min_val = val
        nx.set_node_attributes(G, min_val, 'MVC')
        cnt += 1

    return list_graphs

def create_dataset_buffer(cmp, device, times, dim_datasamples, list_G,show_graphs_stats = True,flag_density = 'False'):
    list_val = []
    list_dataset = []

    num = find_max_list_graph(dim_datasamples)
    if flag_density == 'True':
        num = int(num*2)

    for G in list_G:
        if show_graphs_stats == True:
            print(f'Density: {G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1)) * 2:<.3f}', G,
                    sep=' | ')


        list_graphs, val = build_tree(cmp,nx.to_numpy_array(G),device,flag_density)  # the list contains all the intermediate graphs
        list_val.append(val)

        if len(list_graphs) > num:
            list_graphs = list_graphs[0:num]

        list_graphs = run_several_times(list_graphs, times, cmp, device)

        samples = create_list_datasamples(list_graphs, device, dim_datasamples)

        if len(samples) > dim_datasamples:
            list_dataset = list_dataset + samples[0:dim_datasamples]
        else:
            list_dataset = list_dataset + samples


    return list_dataset, list_val


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


def find_max_list_graph(max_num):
    i = 5
    val = -1
    while (val < max_num):
        val = scipy.special.binom(i, 2)
        i += 1
    return i - 1


