"""Exact GE baseline extracted from the research implementation."""
import numpy as np
import scipy.sparse as sp
import networkx
import gurobipy as gp
import datetime

def get_manual_individual_subnetwork_GE(graph: np.array, L: int, indices: list):
    N = len(indices)
    sp = get_shortest_paths(graph)
    sp = sp[indices]
    sp = sp[:, indices]
    sp_ = sp[sp != 0]
    sp_ = sp_[sp_ <= L]
    return np.sum(1 / sp_) / (N * (N - 1))

def find_min_node_cut(s, t, tumorous_graph_beta, removed_nodes):
    T = len(tumorous_graph_beta)
    nodes_to_duplicate = [i for i in range(T) if i != s and i != t]
    R = len(nodes_to_duplicate)
    new_node_names = []
    for i in range(R):
        new_node_names.append(str(nodes_to_duplicate[i]) + "'")
    new_node_names = dict(list(zip([T + r for r in range(R)], new_node_names)))
    new_graph = networkx.DiGraph(np.zeros((T + R, T + R)))
    new_graph = networkx.relabel_nodes(new_graph, new_node_names)
    for i in nodes_to_duplicate:
        if i in removed_nodes:
            new_graph.add_edge(i, str(i) + "'", weight=1)
        else:
            new_graph.add_edge(i, str(i) + "'", weight=0 + 1e-06)
        for j in range(T):
            if tumorous_graph_beta[i][j] == 1:
                new_graph.add_edge(str(i) + "'", j, weight=1)
    for i in range(T):
        if i not in nodes_to_duplicate:
            for j in range(T):
                if tumorous_graph_beta[i][j] == 1:
                    new_graph.add_edge(i, j, weight=1)
    min_cut = networkx.minimum_cut(new_graph, s, t, capacity='weight')
    S_cut = list(min_cut[1][0])
    T_cut = list(min_cut[1][1])
    min_node_cut = []
    for i in range(len(S_cut)):
        if str(S_cut[i]) + "'" in T_cut:
            min_node_cut.append(S_cut[i])
    return (min_cut[0], min_node_cut)

def get_shortest_paths(graph: np.array, M: int=10 ** 3):
    G = networkx.Graph(graph)
    shortest_paths = dict(networkx.shortest_path_length(G))
    shortest_paths_list = []
    for i in range(graph.shape[0]):
        ar = M * np.ones(graph.shape[0])
        for node, distance in shortest_paths[i].items():
            ar[node] = distance
        shortest_paths_list.append(ar.tolist())
    return np.array(shortest_paths_list)

def f(l):
    return 1 / l

def MNRPaddVolConstr(m, x, volume, C):
    return m.addConstr(gp.LinExpr(gp.quicksum((x[i] * volume[i] for i in range(len(volume))))) >= C)

def MNRPaddGEConstr(m, GE, GE_min):
    return m.addConstr(GE >= GE_min)

def MNRPsolve(subnetwork, instance, mnrpConfig, mnrp, x, geDict, constrDict):

    def callback_constr(model, where):
        E_beta = instance.E_beta
        T = len(instance.tumorous_roi_ids)
        if where == gp.GRB.Callback.MIPSOL:
            vals = model.cbGetSolution(model._vars)
            removed_nodes = [i for i in range(T) if vals[i] > 0]
            removed_node_vals = [vals[i] for i in range(T) if vals[i] > 0]
            removed_nodes_graph = networkx.from_numpy_array(E_beta[removed_nodes][:, removed_nodes])
            cont_list = list(networkx.connected_components(removed_nodes_graph))
            if len(cont_list) != 1:
                mapped_cont_list = []
                for cont in cont_list:
                    mapped_cont_list.append([removed_nodes[i] for i in cont])
            for i in range(len(removed_nodes)):
                for j in range(1, len(removed_nodes)):
                    if E_beta[removed_nodes[i]][removed_nodes[j]] < 1 and i != j:
                        flow, node_cut = find_min_node_cut(removed_nodes[i], removed_nodes[j], E_beta[:T][:, :T], removed_nodes)
                        if flow - vals[removed_nodes[i]] - vals[removed_nodes[j]] + 1 < 0:
                            model.cbLazy(gp.quicksum((model._vars[s] for s in node_cut)), gp.GRB.GREATER_EQUAL, model._vars[removed_nodes[i]] + model._vars[removed_nodes[j]] - 1)
                            row = np.zeros(T)
                            for s in node_cut:
                                row[s] = 1
                            row[removed_nodes[i]] = -1
                            row[removed_nodes[j]] = -1
                        else:
                            pass
    for key in constrDict:
        mnrp.remove(constrDict[key])
    constrDict.clear()
    for cnfg in mnrpConfig:
        if cnfg['targetType'] == 'ge' and cnfg['minValue'] > 0:
            constrDict[cnfg['subnetwork']] = mnrp.addConstr(geDict[cnfg['subnetwork']] >= cnfg['minValue'])
    mnrp.setObjective(geDict[subnetwork], gp.GRB.MAXIMIZE)
    mnrp.Params.lazyConstraints = 1
    mnrp.update()
    mnrp._vars = x
    start_time = datetime.datetime.now()
    mnrp.optimize(callback_constr)
    end_time = datetime.datetime.now()
    if mnrp.status != gp.GRB.OPTIMAL:
        return (None, None, constrDict, end_time - start_time)
    resultingGEdict = {}
    for key in geDict:
        resultingGEdict[key] = geDict[key].getValue()
    removed_nodes = [i for i in range(len(x)) if x[i].X > 0]
    return (resultingGEdict, removed_nodes, constrDict, end_time - start_time)

def MNRP_MultiObjective(mnrpConfig, subnetworkParcels, new_order, E_alpha, volume, T, L, simplification, is_LP, time_limit_mins):

    def I(i, j):
        return int(E_alpha[i][j])
    V = len(E_alpha)
    H = V - T
    removed_graph = E_alpha.copy()
    removed_graph[np.arange(T)] = 0
    removed_graph[:, np.arange(T)] = 0
    m = gp.Model()
    m.setParam('NodefileStart', 0.5)
    m.setParam('OutputFlag', 1)
    m.setParam('TimeLimit', time_limit_mins * 60)
    const_start = datetime.datetime.now()
    print('Creating variables...')
    x = m.addVars(V, vtype=gp.GRB.CONTINUOUS if is_LP else gp.GRB.BINARY)
    u = m.addVars(L + 1, V, V, vtype=gp.GRB.CONTINUOUS, ub=1, lb=0)
    print('Variables created.\n')
    if simplification:
        initial_shortest_paths = get_shortest_paths(E_alpha)
        all_removed_paths = get_shortest_paths(removed_graph)
        indices_to_remove = []
        my_vector = np.ones(E_alpha.size * (L + 1))
        N = T + H
        for i in range(E_alpha.shape[0]):
            for j in range(E_alpha.shape[0]):
                if i != j and (i >= T or j >= T):
                    d_ij_0 = initial_shortest_paths[i, j]
                    d_ij_1 = all_removed_paths[i, j]
                    low_ind = get_index(i, j, min(d_ij_0, L), T + H)
                    indices_to_0 = np.arange(start=low_ind - N ** 2, stop=0, step=-N ** 2).astype(int)
                    up_ind = get_index(i, j, d_ij_1, T + H)
                    indices_to_1 = np.arange(start=up_ind, stop=N ** 2 * (L + 1), step=N ** 2).astype(int)
                    my_vector[indices_to_0] = 0
                    indices_to_remove.extend(list(indices_to_0) + list(indices_to_1))
        values = np.array(u.values().copy())
        values[indices_to_remove] = 1
        values[my_vector == 0] = 0
        keys = u.keys().copy()
        u = dict(zip(keys, values.tolist()))
        for i in range(T, V):
            x[i] = 0
        resected_volume = gp.LinExpr(gp.quicksum((x[i] * volume[i] for i in range(T))))
    else:
        resected_volume = gp.LinExpr(gp.quicksum((x[i] * volume[i] for i in range(T))))
        for i in range(T, V):
            x[i].ub = 0
    constrDict = {}
    geDict = {}
    for cnf in mnrpConfig:
        if cnf['targetType'] == 'ge':
            subnetwork, minValue = (cnf['subnetwork'], cnf['minValue'])
            subnetworkIndices = np.where(np.isin(new_order, subnetworkParcels[subnetwork]) == 1)[0] if subnetwork != 'overall' else np.arange(len(E_alpha))
            _tempGE = gp.quicksum((1 / l * (u[l, i, j] - u[l - 1, i, j]) for l in range(1, L + 1) for i in subnetworkIndices for j in subnetworkIndices if i != j))
            geDict[subnetwork] = _tempGE / (len(subnetworkIndices) * (len(subnetworkIndices) - 1))
            if minValue > 0:
                constrDict[subnetwork] = m.addConstr(geDict[subnetwork] >= minValue)
    print('Global efficiency constraints are added.')
    m.addConstrs((u[l, i, j] >= u[l - 1, i, j] for l in range(1, L + 1) for i in range(V) for j in range(V)))
    m.addConstrs((u[0, i, i] == 1 for i in range(V)))
    m.addConstrs((u[0, i, j] == 0 for i in range(V) for j in range(V) if i != j))
    print('3 other constrs are added')
    m.addConstrs((u[1, i, j] >= I(i, j) - x[i] - x[j] for i in range(V) for j in range(V) if i != j))
    m.addConstrs((u[1, i, j] <= I(i, j) for i in range(V) for j in range(V) if i != j))
    m.addConstrs((u[l, i, j] <= 1 - x[i] for l in range(1, L + 1) for i in range(V) for j in range(V) if i != j))
    print('3 other constrs are added')
    m.addConstrs((u[l, i, j] <= 1 - x[j] for l in range(1, L + 1) for i in range(V) for j in range(V) if i != j))
    m.addConstrs((u[l, i, j] == u[1, i, j] for l in range(2, L + 1) for i in range(V) for j in range(V) if I(i, j) == 1))
    m.addConstrs((u[l, i, j] <= gp.quicksum((u[l - 1, t, j] for t in range(V) if I(i, t) == 1)) for l in range(2, L + 1) for i in range(V) for j in range(V) if I(i, j) == 0 and i != j))
    print('3 other constrs are added')
    m.addConstrs((u[l, i, j] >= u[l - 1, t, j] - x[i] for l in range(2, L + 1) for i in range(V) for j in range(V) if I(i, j) == 0 for t in range(V) if I(i, t) == 1))
    m.addConstrs((u[l, i, j] == u[l, j, i] for l in range(1, L + 1) for i in range(V) for j in range(V) if i != j))
    print('Symmetry constrs are added')
    print('Constraints are done.')
    const_time = datetime.datetime.now() - const_start
    return (m, x, u, geDict, constrDict, const_time)
