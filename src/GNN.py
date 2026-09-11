"""Inference-only checkpoint adapter; no training code or training data."""
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool
from generateGraphs import subnetworkParcels

class sGNN(torch.nn.Module):

    def __init__(self, nb_features, in_channels1, in_channels2, weight_dropout):
        super(sGNN, self).__init__()
        self.weight_dropout = weight_dropout
        self.gc1 = GCNConv(nb_features, in_channels1)
        self.fc = torch.nn.Linear(in_channels1, 1)

    def forward(self, x, edge_index, sigma, batch):
        x1 = x + sigma
        x1 = self.gc1(x1, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.weight_dropout, training=self.training)
        global_master_feat = global_mean_pool(x1, batch)
        ge = self.fc(global_master_feat)
        return ge

class GNN:

    def __init__(self, targetType, graphType, subnetwork, nb_features, in_channels1, in_channels2, weight_dropout):
        assert targetType in ['ge', 'modularity', 'algebraic_connectivity'], f"targetType must be one of ['ge', 'modularity'], got {targetType}"
        assert subnetwork in subnetworkParcels.keys() or subnetwork == 'overall', f"subnetwork must be one of {list(subnetworkParcels.keys())} or 'overall', got {subnetwork}"
        self.targetType = targetType
        self.subnetwork = subnetwork
        self.graphType = graphType
        self.nb_features = nb_features
        self.in_channels1 = in_channels1
        self.in_channels2 = in_channels2
        self.weight_dropout = weight_dropout
        self.sGNN = sGNN(nb_features, in_channels1, in_channels2, weight_dropout)
        self.criterion = None
        self.optimizer = None
        self.train_predictions = []
        self.train_actuals = []
        self.val_predictions = []
        self.val_actuals = []
        self.train_dataset = None
        self.test_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

    def load_model(self, model_path):
        self.sGNN.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))

    def relu(self, x):
        return np.maximum(0, x)

    def manual_forward(self, x, sigma, deg_inv_sqrt, A, theta1=None, bias1=None, beta=None, biasLin=None):
        if theta1 is None:
            theta1 = self.sGNN.gc1.lin.weight.detach().numpy()
        if bias1 is None:
            bias1 = self.sGNN.gc1.bias.detach().numpy()
        if beta is None:
            beta = self.sGNN.fc.weight.detach().numpy()
        if biasLin is None:
            biasLin = self.sGNN.fc.bias.detach().numpy()
        dAd = deg_inv_sqrt @ A @ deg_inv_sqrt
        x = self.relu(dAd @ (x + sigma) @ theta1.T + bias1) if self.targetType == 'modularity' else self.relu(dAd @ x @ theta1.T + bias1)
        xGlobal = np.mean(x, axis=0)
        ge = beta @ xGlobal.T + biasLin
        return ge.item()
