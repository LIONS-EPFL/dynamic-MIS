import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
import sys


class Comparator(nn.Module):

    def __init__(self, D, device, num_gnn_layers=3, num_dense_layers=3):
        super(Comparator, self).__init__()
        assert num_dense_layers > 2  # For skip connection

        self.d = D

        self.gnn_f_layers = nn.ModuleList([
            nn.Linear(3 * D, D) for _ in range(num_gnn_layers)
        ])

        self.gnn_s_layers = nn.ModuleList([
            nn.Linear(3 * D, D) for _ in range(num_gnn_layers)
        ])

        self.gnn_a_layers = nn.ModuleList([
            nn.Linear(3 * D, D) for _ in range(num_gnn_layers)
        ])

        self.gnn_layer_norms = nn.ModuleList([
            nn.LayerNorm(3 * D) for _ in range(num_gnn_layers)
        ])

        self.dense_layers = nn.ModuleList()
        for i in range(num_dense_layers):
            if i == 0:
                self.dense_layers.append(nn.Linear(3 * D, D))
            elif i + 1 == num_dense_layers:
                self.dense_layers.append(nn.Linear(2 * D, 1))

            else:
                self.dense_layers.append(nn.Linear(D, D))

        self.dense_layer_norms = nn.ModuleList([
            nn.LayerNorm(D) for _ in range(num_dense_layers - 1)
        ])

        self.to(device)

    # Takes the adjecancy matrx A of the graph and computes the D-dimensional embedding of the graph

    def Embed(self, A, device):
        GELU = nn.GELU()

        # Computes the complement (including self-loops, which we need anyway)
        B = 1 - A

        # Add self loops to original adjacency matrix!!!
        A = A.fill_diagonal_(1)

        n = A.shape[0]
        X = torch.zeros(n, self.d * 3).to(device)

        for i in range(len(self.gnn_a_layers)):
            Y = self.gnn_f_layers[i](X)
            Z = self.gnn_s_layers[i](X)
            W = self.gnn_a_layers[i](X)

            X = GELU(torch.cat((Z, A @ Y, B @ W), dim=-1))
            X = self.gnn_layer_norms[i](X)

        # Global pooling
        X = torch.mean(X, dim=0)

        return X

    def forward(self, A, B, device):
        warnings.filterwarnings("ignore")
        A = A.to(device)  # Add self loops, important!
        B = B.to(device)  # Add self loops, important!
        RELU = nn.ReLU()

        X = self.Embed(A, device)  # compute the embedding of A
        Y = self.Embed(B, device)  # compute the embedding of B

        for i in range(len(self.dense_layers)):
            if i + 1 == len(self.dense_layers):

                X = torch.cat((X, X_0), dim=-1)
                Y = torch.cat((Y, Y_0), dim=-1)

            X = self.dense_layers[i](X)
            Y = self.dense_layers[i](Y)
            if i + 1 < len(self.dense_layers):

                X = self.dense_layer_norms[i](RELU(X))
                Y = self.dense_layer_norms[i](RELU(Y))
            if i == 0:
                X_0, Y_0 = X, Y
        Z = torch.cat((X, Y), dim=0).reshape(1, 2)

        m = torch.nn.Softmax()
        return m(Z)  # Return the input to the FeedForward Net
