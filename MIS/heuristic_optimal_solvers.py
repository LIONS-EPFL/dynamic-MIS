import torch
import sys
import random
import time

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
from ortools.linear_solver import pywraplp


def greedy_MIS(G):
    G = G.copy()
    while 1:
        node, min_val = get_minimum_degree(G)

        if node is None:  # stop if all the nodes are in the independent set
            break

        G.remove_nodes_from([n for n in G.neighbors(node)])

    return G.number_of_nodes()

# given a matrix, the function outputs the node ID with the minimum degree and its degree value
def get_minimum_degree(G):
    min_val = 2 * G.number_of_nodes()
    node_id = None
    for n in G.nodes():
        if G.degree(n) < min_val and G.degree(n) != 0:
            min_val = G.degree(n)
            node_id = n

    return node_id, min_val


def local_search(G, time_limit):
    start = time.time()

    cur_is = np.zeros(G.number_of_nodes()+1, dtype=bool)
    max_found_is = 0

    while (time.time() - start) < time_limit:
        rnd_node = np.random.choice(G.nodes())
        cur_is[rnd_node] = True
        cur_is[list(nx.neighbors(G, rnd_node))] = False

        cur_is_size = np.sum(cur_is)

        if cur_is_size > max_found_is:
            max_found_is = cur_is_size

    return max_found_is



def solve_mis_ilp(graph, time_limit_milliseconds, mode='GUROBI', solver=None):
    # Create OR-Tools integer programming solver
    if solver is None:
        solver = pywraplp.Solver.CreateSolver(mode)
    else:
        solver.Clear()

    # Define decision variables
    assignments = {}
    for node_id in graph.nodes():
        assignments[node_id] = solver.IntVar(0, 1, f'x[{node_id}]')

    # Define objective function
    objective = solver.Objective()
    for assignment_variable in assignments.values():
        objective.SetCoefficient(assignment_variable, 1)
    objective.SetMaximization()

    # Conflict constraint
    for edge in graph.edges():
        constraint = solver.RowConstraint(0, 1)
        constraint.SetCoefficient(assignments[edge[0]], 1)
        constraint.SetCoefficient(assignments[edge[1]], 1)


    solver.set_time_limit(time_limit_milliseconds)
    status = solver.Solve()

    optimal = False
    if status == pywraplp.Solver.OPTIMAL:
        optimal = True

    mip_value = solver.Objective().Value()

    return mip_value, optimal

