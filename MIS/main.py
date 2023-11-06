from argparse import ArgumentParser

import torch
import random
import matplotlib.pyplot as plt
import model
import torch.nn as nn
import warnings
import os
import buffer
import networkx as nx
import pickle
import sys

from utils import get_path_from_dataset_name
from utils import load_benchmark_data
from test import test_dataset
from train import train_dataset
from train import load

path = os.path.dirname(os.path.realpath(__file__))


def train_MIS(D, gnn_depth, dense_depth, dataset_name,batch_size, dim_datasamples, dim_dataset, root_graphs_per_iter,
                 idx0, idx1, idx0_validation,idx1_validation, mixed_rollout):

    buffer_size = 3*dim_dataset
    buf = buffer.buffer(dim_buffer=buffer_size)

    epochs_roll_out = int(300*root_graphs_per_iter/(idx1-idx0))

    warnings.filterwarnings("ignore")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    

    cmp = model.Comparator(D, device, gnn_depth, dense_depth)
    cmp.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cmp.parameters(), lr=0.001)
    times = 5

    '''
    epochs_roll_out = total number of epochs each dataset is run
    tot_graphs = total number of root graphs to be trained
    dim_datasamples = total number of datasamples each root graph can generate
    dim_dataset = at each iteration, that number of datasamples are trained
    root_graphs_per_iteration = total number of root graphs from which, at each iteration, datasamples are generated
    '''

    dataset_path = get_path_from_dataset_name(dataset_name)
    list_G = load_benchmark_data(dataset_name, dataset_path=dataset_path, idxs=(idx0, idx1))

    if idx0_validation != -1:
        list_G_validation = load_benchmark_data(dataset_name,dataset_path=dataset_path,idxs=(idx0_validation,idx1_validation))
    else:
        list_G_validation = None

    train_dataset(cmp, epochs_roll_out, optimizer, criterion, batch_size, buf, list_G, dim_datasamples,
                    dim_dataset, root_graphs_per_iter, times, device, D, gnn_depth, dense_depth,
                    dataset=dataset_name, mixed_rollout=mixed_rollout,
                    list_G_validation=list_G_validation)
    


def test_MIS(dataset_name, model_path,model_name,D, gnn_depth,dense_depth):
    warnings.filterwarnings("ignore")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    dataset_path = get_path_from_dataset_name(dataset_name)

    cmp_buffer = model.Comparator(D, device, num_dense_layers=dense_depth, num_gnn_layers=gnn_depth)
    load(cmp_buffer, os.path.join(model_path,model_name), device)
    cmp_buffer.eval()

    in_file_name = 'OPTIMA_' + str(dataset_name) + '.pickle'
    file_path = os.path.join(path, 'dataset_gurobi', in_file_name)
    optimum_mips = None
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as handle:
            optimum_mips = pickle.load(handle)

    test_dataset(cmp_buffer, device, D, dataset_name, optimum_mips=optimum_mips, dataset_path=dataset_path)

parser = ArgumentParser()
parser.add_argument('--mode', type=str,choices=[ 'train', 'test'],required=True)
parser.add_argument('--model_path', type=str, required=False)

# Some arguments for train_buffer
parser.add_argument('--D', default=32, type=int, required=False)
parser.add_argument('--gnn_depth', default=3, type=int, required=False)
parser.add_argument('--dense_depth', default=4, type=int, required=False)
parser.add_argument('--dataset_name', choices=["TWITTER_SNAP", "COLLAB", "SPECIAL","RB"], type=str,  required=False)
parser.add_argument('--batch_size', default=32, type=int, required=False)
parser.add_argument('--dim_datasamples', default=128, type=int, required=False)
parser.add_argument('--dim_dataset', default=5120, type=int, required=False)
parser.add_argument('--root_graphs_per_iter', default=40, type=int, required=False)
parser.add_argument('--idx0', default=0, type=int, required=False)
parser.add_argument('--idx1', default=100, type=int, required=False)
parser.add_argument('--idx0_validation', default=-1, type=int, required=False)
parser.add_argument('--idx1_validation', default=-1, type=int, required=False)
parser.add_argument('--model_name', default=None, type=str, required=False)
parser.add_argument('--mixed_rollout', default='True', type=str, required=False)

params = parser.parse_args()

if __name__ == '__main__':
    if params.model_path is None:
        params.model_path = os.path.join(path, 'model_parameters')

    if params.mode == 'train':
        train_MIS(params.D, params.gnn_depth, params.dense_depth,
                     params.dataset_name, params.batch_size,params.dim_datasamples, params.dim_dataset,
                     params.root_graphs_per_iter, params.idx0, params.idx1, params.idx0_validation,params.idx1_validation,params.mixed_rollout)
    elif params.mode == 'test':
        test_MIS(params.dataset_name, params.model_path, params.model_name,params.D,params.gnn_depth, params.dense_depth)
