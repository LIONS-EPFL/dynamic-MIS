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


def max_degree_heuristic(adj_):
    adj = np.copy(adj_)
    num_nodes = 0
    while(1):
        # get the node with maximum degree
        max_val = -1
        node = None
        for i in range(adj.shape[0]):
            val = np.sum(adj[i])
            if val != 0 and val > max_val:
                max_val = val
                node = i

        #check if the graph has no more edges
        if node == None:
            return num_nodes
        num_nodes += 1

        #construct a new graph as the old one without the edges of the node 'node'
        for i in range(adj.shape[0]):
            if adj[node][i] == 1:
                adj[node][i] = 0
                adj[i][node] = 0


def solve_mvc_ilp(graph, time_limit_milliseconds, mode='GUROBI', solver=None):
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
    objective.SetMinimization()

    # Cover constraint
    for edge in graph.edges():
        constraint = solver.RowConstraint(1, solver.infinity())
        constraint.SetCoefficient(assignments[edge[0]], 1)
        constraint.SetCoefficient(assignments[edge[1]], 1)

    solver.set_time_limit(time_limit_milliseconds)
    status = solver.Solve()

    optimal = False
    if status == pywraplp.Solver.OPTIMAL:
        optimal = True

    mvc_value = solver.Objective().Value()

    return mvc_value, optimal



