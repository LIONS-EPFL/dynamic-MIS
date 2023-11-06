import os
import pickle
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
import torch
from tqdm import tqdm

def load_benchmark_data(dataset_name, dataset_path=None, idxs=(0, 100)):
    if dataset_name == "RB200Train" or dataset_name == 'RB500Train' or dataset_name == "RB200Test" or dataset_name == "RB500Train":
        if dataset_path is None:
            raise Exception("dataset_path cannot be None")
        stored_dataset = open(dataset_path, 'rb')
        dataset = pickle.load(stored_dataset)
        return dataset[idxs[0]:idxs[1]]
    else:
        raise Exception("The provided dataset_name is not allowed")





def get_path_from_dataset_name(dataset_name):
    path = os.path.dirname(os.path.realpath(__file__))

    rb200Train_path = os.path.join(path,'dataset_buffer','RB200Train_graphs.pickle')
    rb200Test_path = os.path.join(path,'dataset_buffer','RB200HardTest_graphs.pickle')
    rb500Train_path = os.path.join(path,'dataset_buffer','RB500Train_graphs.pickle')
    rb500Test_path = os.path.join(path,'dataset_buffer','RB500HardTest_graphs.pickle')

    if dataset_name == 'RB200Train':
        dataset_path = rb200Train_path
    elif dataset_name == 'RB200Test':
        dataset_path = rb200Test_path
    elif dataset_name == 'RB500Train':
        dataset_path = rb500Train_path
    elif dataset_name == 'RB500Train':
        dataset_path = rb500Test_path
    else:
        dataset_path = None

    return dataset_path
