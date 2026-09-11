"""Portable loading and evaluation of the exact archived problem instances."""
from pathlib import Path
import pickle
import json
import datetime
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
import networkx as nx
from generateGraphs import ROOT, subnetworkParcels


def load_instance(instance_id):
    path = ROOT / 'instances' / f'mip_instance_{int(instance_id)}.pkl'
    with path.open('rb') as f:
        obj = pickle.load(f)
    # Old SciPy DOK pickles stored entries in the inherited dict, before _dict existed.
    for name in ('E', 'A'):
        matrix = getattr(obj, name)
        if isinstance(matrix, dict) and not hasattr(matrix, '_dict'):
            entries = dict(dict.items(matrix))
            restored = sp.dok_matrix(matrix.shape, dtype=matrix.dtype)
            for key, value in entries.items():
                restored[key] = value
            setattr(obj, name, restored)
    assert obj.id == int(instance_id)
    return obj


def indices(instance, subnetwork):
    return (np.arange(len(instance.new_order)) if subnetwork == 'overall' else
            np.flatnonzero(np.isin(instance.new_order, subnetworkParcels[subnetwork])))


def residual(instance, removed):
    graph = instance.E.toarray().copy()
    graph[removed, :] = 0
    graph[:, removed] = 0
    return graph


def efficiency(graph, selected, cutoff=5):
    """Historical metric: fixed node count, paths through the whole graph, L=5."""
    distances = shortest_path(sp.csr_matrix(graph), directed=False, unweighted=True)
    distances = distances[np.ix_(selected, selected)]
    good = (distances > 0) & (distances <= cutoff)
    return float(np.sum(1 / distances[good]) / (len(selected) * (len(selected) - 1)))


def partition(instance):
    return list(nx.community.greedy_modularity_communities(nx.Graph(instance.E.toarray())))


def metric(instance, removed, objective, subnetwork, communities=None):
    graph = residual(instance, removed)
    if objective == 'ge':
        return efficiency(graph, indices(instance, subnetwork))
    return float(nx.community.modularity(nx.Graph(graph), communities or partition(instance), weight=None))


def propagation(instance, subnetwork, removed=()):
    a = instance.A.toarray().copy()
    a[list(removed), :] = 0
    a[:, list(removed)] = 0
    ix = indices(instance, subnetwork)
    a = a[np.ix_(ix, ix)]
    np.fill_diagonal(a, 1)
    inverse = 1 / np.sqrt(a.sum(axis=1))
    return a * inverse[:, None] * inverse[None, :]


def sigma_matrix(instance):
    sigma = np.zeros(instance.E.shape)
    for community in partition(instance):
        ix = list(community)
        sigma[np.ix_(ix, ix)] = 1
    np.fill_diagonal(sigma, 0)
    return sigma


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, datetime.timedelta): return value.total_seconds()
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False))
    tmp.replace(path)
