import torch
import matplotlib.pyplot as plt
import math
import random
import sys
import model
import networkx as nx
import time
import datetime
import os
import pickle
import numpy as np
from recursive_tree import create_dataset_buffer


def create_output_target(cmp, list_datasamples, batch_size, ind, device):
    G1 = list_datasamples[ind].G1
    G2 = list_datasamples[ind].G2
    G1_matrix = torch.tensor(nx.adjacency_matrix(G1).todense(), dtype=torch.float).to(device)
    G2_matrix = torch.tensor(nx.adjacency_matrix(G2).todense(), dtype=torch.float).to(device)
    output_vet = cmp.forward(G1_matrix, G2_matrix, device)
    target_vet = list_datasamples[ind].target
    ind += 1
    for i in range(1, batch_size):
        G1 = list_datasamples[ind].G1
        G2 = list_datasamples[ind].G2
        G1_matrix = torch.tensor(nx.adjacency_matrix(G1).todense(), dtype=torch.float).to(device)
        G2_matrix = torch.tensor(nx.adjacency_matrix(G2).todense(), dtype=torch.float).to(device)
        output = cmp.forward(G1_matrix, G2_matrix, device)
        target = list_datasamples[ind].target
        output_vet = torch.cat((output_vet, output), dim=0)
        target_vet = torch.cat((target_vet, target), dim=0)
        ind += 1
    return output_vet, target_vet, ind

def calculate_accuracy(target_vet, output_vet):
    accuracy = 0.0
    cnt = 0
    for target, output in zip(target_vet, output_vet):
        if target[0] != 0.5:
            output_class = torch.argmax(output)
            target_class = torch.argmax(target)
            if torch.sum(torch.abs(output_class - target_class)).item() == 0:
                accuracy += 1.0
            cnt += 1
    return accuracy, cnt

def train_dataset(cmp, epochs_roll_out, optimizer, criterion, batch_size, buf, root_graphs_list, dim_datasamples,
                 dim_dataset, root_graphs_per_iteration, times, device, D, gnn_depth, dense_depth, mixed_rollout='False', list_G_validation=None, dataset=''):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(path,'model_parameters') #path where I save the model parameters

    cnt = 0
    for i in range(0, len(root_graphs_list), root_graphs_per_iteration):

        list_G = root_graphs_list[i:i + root_graphs_per_iteration]
        list_datasamples, _ = create_dataset_buffer(cmp, device, times, dim_datasamples, list_G,mixed_rollout=mixed_rollout)

        print('ITERATION NUMBER ' + str(cnt) )

        buf.buffer_update(list_datasamples)

        list_dataset = buf.get_samples(dim_dataset)

        for epoch in range(epochs_roll_out):

            num_it = math.floor(len(list_dataset) / batch_size)

            ind = 0
            loss_epoch = 0.0
            accuracy = 0.0
            den = 0
            for j in range(num_it):
                optimizer.zero_grad()
                cmp.train()
                output_vet, target_vet, ind = create_output_target(cmp, list_dataset, batch_size, ind, device)

                loss = criterion(output_vet, target_vet)

                loss_epoch += loss.item()
                accuracy_,den_ = calculate_accuracy(target_vet, output_vet)
                accuracy += accuracy_
                den += den_ 

                loss.backward()
                optimizer.step()
                cmp.eval()

            if (epoch+1)%10 == 0 and list_G_validation != None:
                list_datasamples_validation, list_val_validation = create_dataset_buffer(cmp, device, times, np.floor(dim_datasamples/2), list_G_validation,mixed_rollout=mixed_rollout,show_graphs_stats=False)
                ind = 0
                output_vet_validation, target_vet_validation, ind = create_output_target(cmp, list_datasamples_validation, len(list_datasamples_validation), ind, device)
                loss_validation = criterion(output_vet_validation,target_vet_validation)
                loss_validation_ = loss_validation.item()
                loss_validation_ = loss_validation_/len(output_vet_validation)
                accuracy_validation,den_validation = calculate_accuracy(target_vet_validation,output_vet_validation)
                accuracy_validation = 100*accuracy_validation / den_validation
                indep_validation = np.array(list_val_validation).mean()

            if den !=0 :
                accuracy = 100 * accuracy / (den)
                loss_epoch = loss_epoch/(num_it * batch_size)
            
            if (epoch+1)%10 == 0 and list_G_validation != None:
                print(f'[EPOCH {epoch+1}]   train_loss: {loss_epoch:.4g} train_accuracy: {accuracy:.4g}    validation_loss: {loss_validation_:.4g} validation_accuracy: {accuracy_validation:.4g} validation_MIS: {indep_validation:.3g}')
            else:
                print(f'[EPOCH {epoch+1}]   train_loss: {loss_epoch:.4g} train_accuracy: {accuracy:.4g}')


        if model_path !=  None:
            stringa = ' dataset=' + str(dataset) + ' D=' + str(D) + ' gnn_depth=' + str(gnn_depth) + ' dense_depth=' + str(dense_depth) +' mixed_rollout='+str(mixed_rollout)+ 'dim_datasamples=' + str(dim_datasamples) + ' dim_dataset='+str(dim_dataset)+' root_graphs_per_iteration'+str(root_graphs_per_iteration)+ ' epochs='+str(epochs_roll_out)
            save(cmp, model_path, timestamp,dataset,stringa)
        cnt += 1


def save(cmp, out_file_path, timestamp,dataset,stringa=None):
    filename = dataset + '_' + timestamp +'_param.pth'
    path_ = os.path.join(out_file_path, filename)
    torch.save(cmp.state_dict(), path_)

    if stringa != None:
        filename = dataset + '_' + timestamp + '_features.txt'
        path_ = os.path.join(out_file_path, filename)
        f = open(path_, 'w')
        f.write(stringa)
        f.close()


def load(cmp, file_path, device):
    cmp.load_state_dict(torch.load(file_path, map_location=device))
