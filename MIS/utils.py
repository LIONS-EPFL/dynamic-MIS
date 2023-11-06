import os
import pickle
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
import torch
from tqdm import tqdm

def load_benchmark_data(dataset_name, dataset_path=None, idxs=(0, 100)):
    if dataset_name == "TWITTER_SNAP":
        if dataset_path is None:
            raise Exception("dataset_path cannot be None when using the TWITTER_SNAP dataset!")
        stored_dataset = open(dataset_path, 'rb')
        dataset = pickle.load(stored_dataset)
        stored_dataset.close()
    elif dataset_name == "COLLAB" or dataset_name == 'SPECIAL' or dataset_name == 'RB':
        if dataset_path is None:
            raise Exception("dataset_path cannot be None when using the COLLAB dataset!")
        stored_dataset = open(dataset_path, 'rb')
        dataset = pickle.load(stored_dataset)
        return dataset[idxs[0]:idxs[1]]
    else:
        raise Exception("The provided dataset_name is not allowed")

    list_G_big = []
    for i in tqdm(range(idxs[0], idxs[1])):
        nx_G = to_networkx(Data(x=torch.ones(torch.max(dataset[i]['edge_index'][0]) + 1, dtype=float),
                                edge_index=dataset[i]['edge_index'])).to_undirected()
        list_G_big.append(nx_G)

    return list_G_big



def get_path_from_dataset_name(dataset_name):
    path = os.path.dirname(os.path.realpath(__file__))

    collab_path = os.path.join(path,'dataset_buffer','collab_graphs.pickle')
    twitter_path = os.path.join(path, 'dataset_buffer', 'TWITTER_SNAP_2.p')
    special_path = os.path.join(path, 'dataset_buffer', 'special_graphs.pickle')
    rb_path = os.path.join(path,'dataset_buffer', 'rb_graphs.pickle')

    if dataset_name == 'TWITTER_SNAP':
        dataset_path = twitter_path
    elif dataset_name == 'SPECIAL':
        dataset_path = special_path
    elif dataset_name == 'RB':
        dataset_path = rb_path
    elif dataset_name == 'COLLAB':
        dataset_path = collab_path
    else:
        dataset_path = None

    return dataset_path
