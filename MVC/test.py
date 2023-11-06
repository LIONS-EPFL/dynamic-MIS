import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp
from tqdm import tqdm

import sys
import model
import networkx as nx
import datetime
import os
import numpy as np
import time

from recursion_tree import find_MVC_value
from heuristic_optimal_solvers import max_degree_heuristic
from heuristic_optimal_solvers import solve_mvc_ilp
from utils import load_benchmark_data


def test_dataset(cmp_buffer, device, D, dataset_name,optimum_mips=None, dataset_path=None):
    idx0, idx1 = get_indeces_dataset(dataset_name)
    list_G = load_benchmark_data(dataset_name, idxs=(idx0, idx1),dataset_path=dataset_path)

    gur_solver = pywraplp.Solver.CreateSolver('GUROBI')

    list_random = []
    list_model_buffer = []
    list_max_degree = []

    list_ortools_1 = []
    list_ortools_5 = []
    list_gurobi_05 = []
    list_gurobi_1 = []
    list_gurobi_5 = []

    list_time_random = []
    list_time_buffer = []
    list_time_max_degree = []

    pbar = tqdm(enumerate(list_G))
    for i, G in pbar:
        pbar.set_description(f'|V|={G.number_of_nodes()}, |E|={G.number_of_edges()}')
        start = time.time()
        # FIND MVC FOR RANDOM COMPARATOR
        cmp_random = model.Comparator(D, device)
        val_random = find_MVC_value(cmp_random, nx.to_numpy_array(G), device)
        end = time.time()
        list_time_random.append(end - start)
        
        start = time.time()
        # FIND MVC FOR MAX_DEGREE
        val_max_degree = max_degree_heuristic(nx.to_numpy_array(G))
        end = time.time()
        list_time_max_degree.append((end - start))
        
        start = time.time()
        # FIND MVC FOR THE MODEL_BUFFER
        val_model_buffer = find_MVC_value(cmp_buffer, nx.to_numpy_array(G), device)
        end = time.time()
        list_time_buffer.append((end - start))
        
        
        # FIND MVC FOR OR and GUROBI (FIX TIME)
        or_1, _ = solve_mvc_ilp(G, time_limit_milliseconds=1000, mode='SCIP')
        or_5, _ = solve_mvc_ilp(G, time_limit_milliseconds=5000, mode='SCIP')
        gur_05, _ = solve_mvc_ilp(G, time_limit_milliseconds=500, solver=gur_solver)
        gur_1, _ = solve_mvc_ilp(G, time_limit_milliseconds=1000, solver=gur_solver)
        gur_5, _ = solve_mvc_ilp(G, time_limit_milliseconds=5000, solver=gur_solver)

        if optimum_mips is not None:

            val_random /= optimum_mips[i]
            val_max_degree /= optimum_mips[i]
            val_model_buffer /= optimum_mips[i]
                        
            or_1 /= optimum_mips[i]
            or_5 /= optimum_mips[i]

            gur_05 /= optimum_mips[i]
            gur_1 /= optimum_mips[i]
            gur_5 /= optimum_mips[i]

        list_random.append(val_random)
        list_max_degree.append(val_max_degree)
        list_model_buffer.append(val_model_buffer)
                
        list_ortools_1.append(or_1)
        list_ortools_5.append(or_5)
        list_gurobi_05.append(gur_05)
        list_gurobi_1.append(gur_1)
        list_gurobi_5.append(gur_5)

    print(f"Mean ratio for random: {(np.array(list_random)).mean()} +/-  {(np.array(list_random)).std()}")
    print(f"Mean ratio for max_degree: {(np.array(list_max_degree)).mean()} +/-  {(np.array(list_max_degree)).std()}")
    print(f"Mean ratio for model_buffer: {(np.array(list_model_buffer)).mean()} +/-  {(np.array(list_model_buffer)).std()}")


    print(f"Mean ratio for or_1: {(np.array(list_ortools_1)).mean()} +/-  {(np.array(list_ortools_1)).std()}")
    print(f"Mean ratio for or_5: {(np.array(list_ortools_5)).mean()} +/-  {(np.array(list_ortools_5)).std()}")

    print(f"Mean ratio for gur_05: {(np.array(list_gurobi_05)).mean()} +/-  {(np.array(list_gurobi_05)).std()}")
    print(f"Mean ratio for gur_1: {(np.array(list_gurobi_1)).mean()} +/-  {(np.array(list_gurobi_1)).std()}")
    print(f"Mean ratio for gur_5: {(np.array(list_gurobi_5)).mean()} +/-  {(np.array(list_gurobi_5)).std()}")

    print('random: (' + str(np.array(list_time_random).mean()) + 's/g)')
    print('max_degree: (' + str(np.array(list_time_max_degree).mean()) + 's/g)')
    print('buffer: (' + str(np.array(list_time_buffer).mean()) + 's/g)')




def get_indeces_dataset(dataset_name):
    if dataset_name == 'RB200' or dataset_name == 'RB500':
        idx0 = 0
        idx1 = 100
    else:
        sys.exit('The provided dataset_name is not allowed')

    return idx0,idx1
