"""Experiment-specific checkpoint selection."""
from functools import lru_cache
import torch
import numpy as np
from generateGraphs import ROOT
from data import propagation, residual, indices, sigma_matrix

OPTIMIZATION = {
 ('ge', 'overall'): 'ge_overallGNN_25.05.25 00:36_overallTrained.pth',
 ('ge', 'Visual_Network'): 'ge_Visual_NetworkGNN_24.05.25 21:35_Visual_NetworkTrained.pth',
 ('ge', 'Multiple_Demand_Extended_Network'): 'ge_Multiple_Demand_Extended_NetworkGNN_25.05.25 17:13_Multiple_Demand_Extended_NetworkTrained.pth',
 ('modularity', 'overall'): 'modularity_overallGNN_25.05.25 20:41_overallTrained.pth',
}
STABILITY = dict(OPTIMIZATION, **{})
STABILITY[('modularity', 'overall')] = 'modularity_overallGNN_05.05.25 20:35.pth'
STABILITY[('ge', 'Multiple_Demand_Extended_Network')] = 'ge_Multiple_Demand_Extended_NetworkGNN_05.05.25 22:04.pth'

@lru_cache(None)
def weights(objective, subnetwork, profile='optimization'):
    filename = (STABILITY if profile == 'stability' else OPTIMIZATION)[objective, subnetwork]
    state = torch.load(ROOT / 'weights' / filename, map_location='cpu', weights_only=True)
    return tuple(state[k].numpy() for k in ('gc1.lin.weight', 'gc1.bias', 'fc.weight', 'fc.bias'))


def predict(instance, removed, objective, subnetwork, baseline='fully', profile='optimization'):
    fixed_removed = (range(len(instance.volumes)) if baseline == 'fully' else
                     removed if baseline == 'exact' else [])
    p = propagation(instance, subnetwork, fixed_removed)
    x = residual(instance, removed)[indices(instance, subnetwork)]
    if objective == 'modularity': x = x + sigma_matrix(instance)
    theta, bias, beta, out_bias = weights(objective, subnetwork, profile)
    return float((beta @ np.maximum(0, p @ x @ theta.T + bias).mean(0) + out_bias).item())


def load_gnn(objective, subnetwork):
    from GNN import GNN
    gnn = GNN(objective, 'brain', subnetwork, 360, 10, 5, 0.5)
    gnn.load_model(ROOT / 'weights' / OPTIMIZATION[objective, subnetwork])
    gnn.sGNN.eval()
    return gnn
