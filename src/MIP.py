"""GNN MIP formulation extracted from the research implementation."""
import gurobipy as gp
from GNN import GNN
from generateGraphs import TrainingInstance, subnetworkParcels
from MNRPutils import get_manual_individual_subnetwork_GE
import numpy as np
from tqdm import tqdm
import networkx as nx
from functools import wraps
import time
import copy

def timeit_to_dict(method):

    @wraps(method)
    def timed(self, *args, **kwargs):
        start = time.time()
        result = method(self, *args, **kwargs)
        end = time.time()
        if not hasattr(self, 'times'):
            self.times = {}
        key = method.__name__
        if args and hasattr(args[0], 'targetType') and hasattr(args[0], 'subnetwork'):
            key += f'_{args[0].targetType}_{args[0].subnetwork}'
        if key in self.times:
            self.times[key].append(end - start)
        else:
            self.times[key] = [end - start]
        return result
    return timed

class MIP:

    def __init__(self, instance: TrainingInstance, subnetworkParcels: dict, mipGap: float=0.001, is_propagate_bigM: bool=True):
        self.instance = instance
        self.subnetworkParcels = subnetworkParcels
        self.is_propagate_bigM = is_propagate_bigM
        self.model = gp.Model(f'MIP_{self.instance.id}')
        self.model.setParam('MIPGap', mipGap)
        self.mipGap = mipGap
        self.model.setParam('NodefileStart', 0.5)
        self.expected_volume_constr = None
        self.gnnDict = {}
        self.times = {}
        self.L = 5
        self.instance.aggregatedSubnetworkParcels_Instance['overall'] = np.arange(360)
        self.instance_info = {}
        self.solution_record = {}
        self.lastResPct = 0

    def positive_mask(self, arr):
        return np.where(arr > 0, arr, 0)

    def negative_mask(self, arr):
        return np.where(arr < 0, arr, 0)

    @timeit_to_dict
    def propagate_bigM(self, N, dAd, theta1, bias1, isModularity: bool=False):
        L0 = np.zeros((N, 360))
        U0 = np.ones((N, 360))
        if isModularity:
            U0 = U0 * 2
        L1 = dAd @ self.positive_mask(L0) @ self.positive_mask(theta1).T + dAd @ self.positive_mask(U0) @ self.negative_mask(theta1).T + bias1
        U1 = dAd @ self.positive_mask(L0) @ self.negative_mask(theta1).T + dAd @ self.positive_mask(U0) @ self.positive_mask(theta1).T + bias1
        return (L1, U1)

    @timeit_to_dict
    def calculate_degree_inverse_sqrt(self, A):
        deg = A.toarray().sum(axis=1) if not isinstance(A, np.ndarray) else A.sum(axis=1)
        deg_inv_sqrt = np.diag(1 / np.sqrt(deg))
        return deg_inv_sqrt

    def getSubnetwork(self, E, subMask):
        E_sub = E.copy()
        E_sub = E[subMask]
        E_sub = E_sub[:, subMask]
        return E_sub

    @timeit_to_dict
    def createModelBase(self):
        E = self.instance.E
        tumorous = np.arange(len(self.instance.tumorous_roi_ids))
        healthy = np.arange(len(tumorous), len(self.instance.graph))
        nb_nodes, nb_features = E.shape
        self.x_0 = self.model.addMVar((nb_nodes, nb_features), vtype=gp.GRB.BINARY, name='x_0')
        self.y = self.model.addVars(nb_nodes, vtype=gp.GRB.BINARY, name='y')
        for i in tqdm(range(nb_nodes)):
            self.model.addConstr(self.x_0[i, i] == 0)
            self.model.addConstr(self.x_0[i] <= 1 - self.y[i])
            if i in healthy:
                self.model.addConstr(self.y[i] == 0)
            for j in range(nb_nodes):
                self.model.addConstr(self.x_0[i, j] == self.x_0[j, i])
                if i in healthy and j in healthy:
                    self.model.addConstr(self.x_0[i, j] == E[i, j])
                if E[i, j] == 0:
                    self.model.addConstr(self.x_0[i, j] == 0)
                if E[i, j] == 1 and i in tumorous and (j in tumorous):
                    self.model.addConstr(self.x_0[i, j] >= 1 - self.y[i] - self.y[j])
                if E[i, j] == 1 and i in tumorous and (j in healthy):
                    self.model.addConstr(self.x_0[i, j] >= 1 - self.y[i] - self.y[j])

    @timeit_to_dict
    def addGNN(self, gnn: GNN):
        subnetworkIndices = np.where(np.isin(self.instance.new_order, self.subnetworkParcels[gnn.subnetwork]) == 1)[0] if gnn.subnetwork != 'overall' else np.arange(len(self.instance.graph))
        tumorous = np.arange(len(self.instance.tumorous_roi_ids))
        A = self.instance.A.toarray()
        A_removed = A.copy()
        A_removed[tumorous] = 0
        A_removed[:, tumorous] = 0
        A_removed = A_removed[subnetworkIndices]
        A_removed = A_removed[:, subnetworkIndices]
        A = A[subnetworkIndices]
        A = A[:, subnetworkIndices]
        E = self.instance.E.toarray()
        nb_features = E.shape[1]
        nb_nodes = len(subnetworkIndices)
        A[np.arange(nb_nodes), np.arange(nb_nodes)] = 1
        A_removed[np.arange(nb_nodes), np.arange(nb_nodes)] = 1
        deg_inv_sqrt = self.calculate_degree_inverse_sqrt(A_removed)
        deg_inv_sqrt_not_removed = self.calculate_degree_inverse_sqrt(A)
        dAd = deg_inv_sqrt @ A_removed @ deg_inv_sqrt
        dAd_not_removed = deg_inv_sqrt_not_removed @ A @ deg_inv_sqrt_not_removed
        tumorous = np.arange(len(self.instance.tumorous_roi_ids))
        if gnn.targetType not in self.gnnDict:
            self.gnnDict[gnn.targetType] = {}
        if gnn.subnetwork not in self.gnnDict[gnn.targetType]:
            self.gnnDict[gnn.targetType][gnn.subnetwork] = {'gnn': gnn, 'subnetworkIndices': subnetworkIndices}
        else:
            raise ValueError(f'GNN for {gnn.targetType} and {gnn.subnetwork} already exists')
        _state_dict = gnn.sGNN.state_dict()
        _theta1 = _state_dict['gc1.lin.weight'].detach().numpy()
        _bias1 = _state_dict['gc1.bias'].detach().numpy()
        _beta = _state_dict['fc.weight'].detach().numpy()
        _biasLin = _state_dict['fc.bias'].detach().numpy()
        x_1 = self.model.addMVar((nb_nodes, _theta1.shape[0]), vtype=gp.GRB.CONTINUOUS, name=f'x_1_{gnn.targetType}_{gnn.subnetwork}')
        self.gnnDict[gnn.targetType][gnn.subnetwork]['x_1'] = x_1
        x_2 = gp.quicksum((x_1[i, :] for i in range(nb_nodes))) / nb_nodes
        self.gnnDict[gnn.targetType][gnn.subnetwork]['x_2'] = x_2
        if self.is_propagate_bigM:
            E_sub = self.getSubnetwork(E, subnetworkIndices)
            L1, U1 = self.propagate_bigM(nb_nodes, dAd, _theta1, _bias1, isModularity=gnn.targetType == 'modularity')
            M_1 = np.where(-L1 > U1, -L1, U1)
        else:
            M_1 = np.ones(x_1.shape) * 500
        self.gnnDict[gnn.targetType][gnn.subnetwork]['M_1'] = M_1
        z_1 = self.model.addMVar((nb_nodes, _theta1.shape[0]), vtype=gp.GRB.BINARY, name=f'z_1_{gnn.targetType}_{gnn.subnetwork}')
        ones_1 = np.ones(M_1.shape)
        U1 = self.model.addMVar((nb_nodes, nb_features), vtype=gp.GRB.CONTINUOUS, name=f'U1_{gnn.targetType}_{gnn.subnetwork}')
        self.gnnDict[gnn.targetType][gnn.subnetwork]['z_1'] = z_1
        self.gnnDict[gnn.targetType][gnn.subnetwork]['ones_1'] = ones_1
        self.gnnDict[gnn.targetType][gnn.subnetwork]['U1'] = U1
        target = _beta @ x_2 + _biasLin
        self.gnnDict[gnn.targetType][gnn.subnetwork]['target'] = target
        gnnConstrList = []
        if gnn.targetType == 'modularity':
            G = nx.Graph(E)
            self.communities = nx.community.greedy_modularity_communities(G)
            sigma = np.zeros((len(G), len(G)))
            self.sigma = sigma
            for i, community in enumerate(self.communities):
                for node in community:
                    for node2 in community:
                        if node != node2:
                            sigma[node, node2] = 1
                            sigma[node2, node] = 1
            modularity = nx.community.modularity(G, self.communities, weight=None)
            gnnConstrList.append(self.model.addConstr(U1 == dAd @ self.x_0 + dAd @ sigma))
            init_target = gnn.manual_forward(E, sigma, deg_inv_sqrt_not_removed, A, _theta1, _bias1, _beta, _biasLin)
            self.gnnDict[gnn.targetType][gnn.subnetwork]['init_targetGNN'] = init_target
            self.gnnDict[gnn.targetType][gnn.subnetwork]['init_targetNx'] = modularity
            gnnConstrList.append(self.model.addConstr(target <= 1))
        else:
            gnnConstrList.append(self.model.addConstr(U1 == dAd @ self.x_0[subnetworkIndices]))
            sigma = np.zeros((nb_nodes, nb_nodes))
            init_target = gnn.manual_forward(E[subnetworkIndices], sigma, deg_inv_sqrt_not_removed, A, _theta1, _bias1, _beta, _biasLin)
            self.gnnDict[gnn.targetType][gnn.subnetwork]['init_targetGNN'] = init_target
            networkx_target = get_manual_individual_subnetwork_GE(E, self.L, subnetworkIndices)
            self.gnnDict[gnn.targetType][gnn.subnetwork]['init_targetNx'] = networkx_target
            gnnConstrList.append(self.model.addConstr(target <= 1))
        gnnConstrList.append(self.model.addConstr(x_1 >= U1 @ _theta1.T + _bias1))
        gnnConstrList.append(self.model.addConstr(x_1 <= U1 @ _theta1.T + _bias1 + (ones_1 * M_1 - z_1 * M_1)))
        gnnConstrList.append(self.model.addConstr(x_1 <= z_1 * M_1))
        self.gnnDict[gnn.targetType][gnn.subnetwork]['gnnConstrList'] = gnnConstrList

    def setObjective(self, targetType: str, subnetwork: str):
        target = self.gnnDict[targetType][subnetwork]['target']
        self.model.setObjective(10 * target, gp.GRB.MAXIMIZE)

    def addExpectedVolumeConstr(self, thresholdPct: float):
        self.removeExpectedVolumeConstr()
        volumes = self.instance.volumes / 100000
        total_expected_volume = volumes.sum()
        self.expected_volume_constr = self.model.addConstr(gp.quicksum((volumes[i] * self.y[i] for i in range(len(volumes)))) >= total_expected_volume * thresholdPct / 100)
        self.C = total_expected_volume * thresholdPct / 100
        self.lastResPct = thresholdPct
        self.model.update()

    def removeExpectedVolumeConstr(self):
        if self.expected_volume_constr is not None:
            self.model.remove(self.expected_volume_constr)
        self.expected_volume_constr = None
        self.model.update()

    def addTargetConstr(self, targetType: str, subnetwork: str, minValue: float):
        target = self.gnnDict[targetType][subnetwork]['target']
        self.removeTargetConstr(targetType, subnetwork)
        self.gnnDict[targetType][subnetwork]['targetConstr'] = self.model.addConstr(10 * target >= 10 * minValue)
        self.model.update()

    def removeTargetConstr(self, targetType: str, subnetwork: str):
        if 'targetConstr' in self.gnnDict[targetType][subnetwork]:
            self.model.remove(self.gnnDict[targetType][subnetwork]['targetConstr'])
        self.model.update()

    def find_min_node_cut(self, s, t, tumorous_graph_beta, removed_nodes):
        T = len(tumorous_graph_beta)
        nodes_to_duplicate = [i for i in range(T) if i != s and i != t]
        R = len(nodes_to_duplicate)
        new_node_names = []
        for i in range(R):
            new_node_names.append(str(nodes_to_duplicate[i]) + "'")
        new_node_names = dict(list(zip([T + r for r in range(R)], new_node_names)))
        new_graph = nx.DiGraph(np.zeros((T + R, T + R)))
        new_graph = nx.relabel_nodes(new_graph, new_node_names)
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
        min_cut = nx.minimum_cut(new_graph, s, t, capacity='weight')
        S_cut = list(min_cut[1][0])
        T_cut = list(min_cut[1][1])
        min_node_cut = []
        for i in range(len(S_cut)):
            if str(S_cut[i]) + "'" in T_cut:
                min_node_cut.append(S_cut[i])
        return (min_cut[0], min_node_cut)

    @timeit_to_dict
    def solve(self, gnnConfig, objectiveType, objectiveSubnetwork, resPct: float=80):

        def callback_constr(model, where):
            E_beta = self.instance.E_beta
            T = len(self.instance.tumorous_roi_ids)
            if where == gp.GRB.Callback.MIPSOL:
                vals = model.cbGetSolution(model._vars)
                removed_nodes = [i for i in range(T) if vals[i] > 0]
                removed_node_vals = [vals[i] for i in range(T) if vals[i] > 0]
                removed_nodes_graph = nx.from_numpy_array(E_beta[removed_nodes][:, removed_nodes])
                cont_list = list(nx.connected_components(removed_nodes_graph))
                if len(cont_list) != 1:
                    mapped_cont_list = []
                    for cont in cont_list:
                        mapped_cont_list.append([removed_nodes[i] for i in cont])
                for i in range(len(removed_nodes)):
                    for j in range(1, len(removed_nodes)):
                        if E_beta[removed_nodes[i]][removed_nodes[j]] < 1 and i != j:
                            flow, node_cut = self.find_min_node_cut(removed_nodes[i], removed_nodes[j], E_beta[:T][:, :T], removed_nodes)
                            if flow - vals[removed_nodes[i]] - vals[removed_nodes[j]] + 1 < 0:
                                model.cbLazy(gp.quicksum((model._vars[s] for s in node_cut)), gp.GRB.GREATER_EQUAL, model._vars[removed_nodes[i]] + model._vars[removed_nodes[j]] - 1)
                                row = np.zeros(T)
                                for s in node_cut:
                                    row[s] = 1
                                row[removed_nodes[i]] = -1
                                row[removed_nodes[j]] = -1
                            else:
                                pass
        assert (objectiveType, objectiveSubnetwork) in [(config['targetType'], config['subnetwork']) for config in gnnConfig], 'Objective type and subnetwork not found in gnnConfig'
        for key in self.gnnDict.keys():
            for key2 in self.gnnDict[key].keys():
                self.removeTargetConstr(key, key2)
        for config in gnnConfig:
            if config['minValue'] != 0:
                self.addTargetConstr(config['targetType'], config['subnetwork'], config['minValue'])
        self.setObjective(objectiveType, objectiveSubnetwork)
        self.addExpectedVolumeConstr(resPct)
        self.model.Params.lazyConstraints = 1
        self.model.update()
        self.model._vars = self.y
        self.model.optimize(callback_constr)
        if self.model.status == gp.GRB.Status.OPTIMAL:
            return self.create_results()

    def get_solution(self):
        if self.model.status == gp.GRB.Status.OPTIMAL:
            beta = self.instance.E_beta
            E = self.instance.E
            removed_nodes = [i for i in range(len(self.instance.tumorous_roi_ids)) if self.y[i].X > 0]
            removedBetaGNN = beta[removed_nodes].copy()
            removedBetaGNN = removedBetaGNN[:, removed_nodes]
            isConnected = nx.is_connected(nx.Graph(removedBetaGNN))
            removed_volume = sum((self.instance.volumes[i] * self.y[i].X for i in range(len(self.instance.volumes))))
            resultingGraph = E.toarray().copy()
            resultingGraph[removed_nodes] = 0
            resultingGraph[:, removed_nodes] = 0
            _A = resultingGraph.copy()
            np.fill_diagonal(_A, 1)
            objectivesGE = {}
            objectivesMod = {}
            resultingGE = {}
            resultingMod = {}
            resultingGEgnn = {}
            resultingModgnn = {}
            initGEsGNN = {}
            initGEsNx = {}
            initModGNN = {}
            initModNx = {}
            if 'ge' in self.gnnDict:
                initGEsGNN = {key: self.gnnDict['ge'][key]['init_targetGNN'] for key in self.gnnDict['ge'].keys()}
                initGEsNx = {key: self.gnnDict['ge'][key]['init_targetNx'] for key in self.gnnDict['ge'].keys()}
                objectivesGE = {key: self.gnnDict['ge'][key]['target'].getValue() for key in self.gnnDict['ge'].keys()}
                resultingGE = {key: get_manual_individual_subnetwork_GE(resultingGraph, self.L, self.gnnDict['ge'][key]['subnetworkIndices']) for key in self.gnnDict['ge'].keys()}
                resultingGEgnn = {}
                for key in self.gnnDict['ge'].keys():
                    _gnn = self.gnnDict['ge'][key]['gnn']
                    _state_dict = _gnn.sGNN.state_dict()
                    _theta1 = _state_dict['gc1.lin.weight'].detach().numpy()
                    _bias1 = _state_dict['gc1.bias'].detach().numpy()
                    _beta = _state_dict['fc.weight'].detach().numpy()
                    _biasLin = _state_dict['fc.bias'].detach().numpy()
                    subnetworkIndices = self.gnnDict['ge'][key]['subnetworkIndices'] if key != 'overall' else np.arange(len(self.instance.graph))
                    _A_sub = _A[subnetworkIndices]
                    _A_sub = _A_sub[:, subnetworkIndices]
                    _deg_inv_sqrt_sub = self.calculate_degree_inverse_sqrt(_A_sub)
                    resultingGEgnn[key] = _gnn.manual_forward(resultingGraph[subnetworkIndices], None, _deg_inv_sqrt_sub, _A_sub, _theta1, _bias1, _beta, _biasLin)
            if 'modularity' in self.gnnDict:
                initModGNN = {key: self.gnnDict['modularity'][key]['init_targetGNN'] for key in self.gnnDict['modularity'].keys()}
                initModNx = {key: self.gnnDict['modularity'][key]['init_targetNx'] for key in self.gnnDict['modularity'].keys()}
                objectivesMod = {key: self.gnnDict['modularity'][key]['target'].getValue() for key in self.gnnDict['modularity'].keys()}
                resultingMod = {key: nx.community.modularity(nx.Graph(resultingGraph), self.communities, weight=None) for key in self.gnnDict['modularity'].keys()}
                resultingModgnn = {}
                for key in self.gnnDict['modularity'].keys():
                    _gnn = self.gnnDict['modularity'][key]['gnn']
                    _state_dict = _gnn.sGNN.state_dict()
                    _theta1 = _state_dict['gc1.lin.weight'].detach().numpy()
                    _bias1 = _state_dict['gc1.bias'].detach().numpy()
                    _beta = _state_dict['fc.weight'].detach().numpy()
                    _biasLin = _state_dict['fc.bias'].detach().numpy()
                    subnetworkIndices = self.gnnDict['modularity'][key]['subnetworkIndices'] if key != 'overall' else np.arange(len(self.instance.graph))
                    _A_sub = _A[subnetworkIndices]
                    _A_sub = _A_sub[:, subnetworkIndices]
                    _deg_inv_sqrt_sub = self.calculate_degree_inverse_sqrt(_A_sub)
                    resultingModgnn[key] = _gnn.manual_forward(resultingGraph[subnetworkIndices], self.sigma[subnetworkIndices][:, subnetworkIndices], _deg_inv_sqrt_sub, _A_sub, _theta1, _bias1, _beta, _biasLin)
            self.modelResults = {'MIP Objective GEs': objectivesGE, 'MIP Objective Modularities': objectivesMod, 'MIP Removed Volume': removed_volume, 'Initial Modularity Nx': initModNx, 'Initial GE Nx': initGEsNx, 'Initial Modularity GNN': initModGNN, 'Initial GE GNN': initGEsGNN, 'MIP Resulting GEs': resultingGE, 'MIP Resulting Modularities': resultingMod, 'MIP Resulting GNN GEs': resultingGEgnn, 'MIP Resulting GNN Modularities': resultingModgnn, 'MIP Removed Nodes': removed_nodes, 'MIP Is Connected': isConnected}
            return self.modelResults
        else:
            raise ValueError('Model not optimal')

    def create_results(self):
        modelResults = self.get_solution()
        self.resultDict = {}
        self.resultDict['id'] = self.instance.id
        self.resultDict['Initial Expected Volume'] = self.instance.volumes.sum()
        self.resultDict['Initial Voxel Volume'] = self.instance.voxel_volumes.sum()
        self.resultDict['Nb Tumorous Regions'] = len(self.instance.tumorous_roi_ids)
        core_region = self.instance.tumorous_region[self.instance.tumorous_region['tumor_density'] >= 1]
        self.resultDict['Nb Core Regions'] = len(core_region)
        self.resultDict['Expected Core Region Volume'] = core_region['expected_tumorous_volume_density'].sum()
        self.resultDict['Expected Core Region Voxel Volume'] = core_region['nr_voxels'].sum()
        self.resultDict['Resect Pct'] = self.lastResPct if self.lastResPct is not None else None
        self.resultDict['C'] = self.C if self.C is not None else None
        self.resultDict['Base Construction Time'] = self.times['createModelBase'][0]
        for key in self.times.keys():
            if 'addGNN' in key:
                self.resultDict[f'Construction {key} Time'] = self.times[key][0]
        totalConstructionTime = sum((self.resultDict[key] for key in self.resultDict.keys() if 'Construction' in key))
        self.resultDict['Total Construction Time'] = totalConstructionTime
        self.resultDict['MIP Time'] = self.model.Runtime
        self.resultDict['MIP Gap'] = self.model.MIPGap
        for key in modelResults['Initial Modularity Nx'].keys():
            self.resultDict[f'Initial modularity Nx {key}'] = modelResults['Initial Modularity Nx'][key]
        for key in modelResults['Initial GE Nx'].keys():
            self.resultDict[f'Initial ge Nx {key}'] = modelResults['Initial GE Nx'][key]
        for key in modelResults['Initial Modularity GNN'].keys():
            self.resultDict[f'Initial modularity GNN {key}'] = modelResults['Initial Modularity GNN'][key]
        for key in modelResults['Initial GE GNN'].keys():
            self.resultDict[f'Initial ge GNN {key}'] = modelResults['Initial GE GNN'][key]
        for key in modelResults['MIP Objective GEs'].keys():
            self.resultDict[f'MIP ge {key}'] = modelResults['MIP Objective GEs'][key][0]
        for key in modelResults['MIP Objective Modularities'].keys():
            self.resultDict[f'MIP modularity {key}'] = modelResults['MIP Objective Modularities'][key][0]
        for key in modelResults['MIP Resulting GEs'].keys():
            self.resultDict[f'MIP Resulting ge {key}'] = modelResults['MIP Resulting GEs'][key]
        for key in modelResults['MIP Resulting Modularities'].keys():
            self.resultDict[f'MIP Resulting modularity {key}'] = modelResults['MIP Resulting Modularities'][key]
        for key in modelResults['MIP Resulting GNN GEs'].keys():
            self.resultDict[f'MIP Resulting GNN ge {key}'] = modelResults['MIP Resulting GNN GEs'][key]
        for key in modelResults['MIP Resulting GNN Modularities'].keys():
            self.resultDict[f'MIP Resulting GNN modularity {key}'] = modelResults['MIP Resulting GNN Modularities'][key]
        self.resultDict['MIP Removed Volume'] = modelResults['MIP Removed Volume']
        self.resultDict['MIP Removed Nodes'] = modelResults['MIP Removed Nodes']
        self.resultDict['MIP Is Connected'] = modelResults['MIP Is Connected']
        return self.resultDict

    def construct_problem(self, gnnList: list):
        self.createModelBase()
        for gnn in gnnList:
            self.addGNN(gnn)

    def lex_max(self, gnnConfig: list, resectPct: float, objectiveType1: str, objectiveSubnetwork1: str, objectiveType2: str, objectiveSubnetwork2: str):
        _gnnConfig = copy.deepcopy(gnnConfig)
        res1 = self.solve(_gnnConfig, objectiveType1, objectiveSubnetwork1, resectPct)
        if res1 is None:
            return None
        lexDecrease = res1[f'MIP {objectiveType1} {objectiveSubnetwork1}']
        print(f'firstObj: {lexDecrease}')
        print(res1[f'MIP {objectiveType1} {objectiveSubnetwork1}'], res1[f'MIP {objectiveType2} {objectiveSubnetwork2}'])
        for config in _gnnConfig:
            if config['targetType'] == objectiveType1 and config['subnetwork'] == objectiveSubnetwork1:
                config['minValue'] = lexDecrease
        res2 = self.solve(_gnnConfig, objectiveType2, objectiveSubnetwork2, resectPct)
        if res2 is None:
            return None
        print(res2[f'MIP {objectiveType1} {objectiveSubnetwork1}'], res2[f'MIP {objectiveType2} {objectiveSubnetwork2}'])
        return res2
