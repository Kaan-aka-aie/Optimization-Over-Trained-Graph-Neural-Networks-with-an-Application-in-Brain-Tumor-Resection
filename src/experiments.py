"""Explicit single-objective and epsilon-constraint experiment runners."""
import copy
import time
import numpy as np
from generateGraphs import subnetworkParcels
from data import load_instance, metric, write_json
from models import load_gnn


def config(objectives):
    return [dict(targetType=o, subnetwork=s, minValue=0) for o, s in objectives]


class Surrogate:
    def __init__(self, instance, objectives, time_limit=1800):
        from MIP import MIP
        self.instance = instance
        self.objectives = objectives
        self.solver = MIP(instance, subnetworkParcels, mipGap=0)
        self.model = self.solver.model
        self.model.setParam('IntFeasTol', 1e-9)
        self.model.setParam('Aggregate', 0)
        self.model.setParam('NumericFocus', 3)
        self.model.setParam('TimeLimit', time_limit)
        self.solver.construct_problem([load_gnn(*obj) for obj in objectives])

    def solve(self, objective, pct, bounds=None, secondary=None):
        cfg = config(self.objectives)
        for c in cfg:
            c['minValue'] = (bounds or {}).get((c['targetType'], c['subnetwork']), 0)
        if secondary:
            result = self.solver.lex_max(cfg, pct, *objective, *secondary)
        else:
            result = self.solver.solve(cfg, *objective, pct)
        if result is None:
            if self.model.Status == 3: return None
            raise RuntimeError(f'Surrogate solve incomplete: Gurobi status {self.model.Status}')
        return result

    @staticmethod
    def value(result, objective): return result[f'MIP {objective[0]} {objective[1]}']


class Exact:
    def __init__(self, instance, objectives, time_limit=1800):
        from MNRPutils import MNRP_MultiObjective
        self.instance = instance
        self.objectives = objectives
        self.model, self.x, self.u, self.ge, self.constraints, self.construction = MNRP_MultiObjective(
            config(objectives), subnetworkParcels, instance.new_order, instance.E.toarray(),
            instance.volumes, len(instance.volumes), 5, False, False, time_limit / 60)
        self.volume_constraint = None
        self.pct = None

    def solve(self, objective, pct, bounds=None, secondary=None):
        from MNRPutils import MNRPsolve, MNRPaddVolConstr
        if self.pct != pct:
            if self.volume_constraint is not None: self.model.remove(self.volume_constraint)
            self.volume_constraint = MNRPaddVolConstr(self.model, self.x, self.instance.volumes,
                                                    pct / 100 * sum(self.instance.volumes))
            self.pct = pct
        cfg = config(self.objectives)
        for c in cfg: c['minValue'] = (bounds or {}).get((c['targetType'], c['subnetwork']), 0)
        result, removed, self.constraints, elapsed = MNRPsolve(
            objective[1], self.instance, cfg, self.model, self.x, self.ge, self.constraints)
        if result is None:
            if self.model.Status == 3: return None
            raise RuntimeError(f'Exact solve incomplete: Gurobi status {self.model.Status}')
        if secondary:
            b = dict(bounds or {})
            b[objective] = result[objective[1]]
            return self.solve(secondary, pct, b)
        result.update({'MNRP Removed Nodes': removed, 'MNRP Optimization Time': elapsed,
                       'Constr Time': self.construction})
        return result

    @staticmethod
    def value(result, objective): return result[objective[1]]


def single(instance_id, method, output, time_limit=1800):
    objectives = [('ge', s) for s in ('overall', 'Visual_Network', 'Multiple_Demand_Extended_Network')]
    instance = load_instance(instance_id)
    solver = (Surrogate if method == 'surrogate' else Exact)(instance, objectives, time_limit)
    results = {}
    try:
        for obj in objectives:
            result = solver.solve(obj, 80)
            if result is None: raise RuntimeError('Unexpected infeasible single-objective instance')
            results[f'{"mip" if method == "surrogate" else "mnrp"}{obj[0]}-{obj[1]}'] = result
            write_json(output, {str(instance_id): results})
    finally:
        solver.model.dispose()


def frontier(instance_id, objectives, pct, method, output, time_limit=1800, total_limit=7200, digits=4):
    objectives = [tuple(o) for o in objectives]
    instance = load_instance(instance_id)
    # Historical biobjective models included all four surrogates in a shared model.
    all_obj = [('ge', 'overall'), ('ge', 'Visual_Network'), ('modularity', 'overall'),
               ('ge', 'Multiple_Demand_Extended_Network')]
    solver = (Surrogate if method == 'surrogate' else Exact)(
        instance, all_obj if method == 'surrogate' else objectives, time_limit)
    first, second = objectives
    try:
        right = solver.solve(first, pct, secondary=second)
        left = solver.solve(second, pct, secondary=first)
        if left is None or right is None: raise RuntimeError('Infeasible frontier endpoints')
        rows = []
        current = solver.value(left, first)
        start = time.monotonic()
        attempts = 0
        finished = False
        while time.monotonic() - start < total_limit:
            attempts += 1
            row = solver.solve(second, pct, {first: current + 10**(-digits)},
                               secondary=first if method == 'surrogate' else None)
            if row is None:
                finished = True
                break
            next_value = solver.value(row, first)
            if next_value <= current: raise RuntimeError('Epsilon step made no progress')
            rows.append(row)
            current = next_value
            # Save progress on every successful solve.
            write_json(output, {'IterationResults': rows, 'LeftCorner': left, 'RightCorner': right,
                                'complete': False})
        elapsed = time.monotonic() - start
        result = dict(IterationResults=rows, LeftCorner=left, RightCorner=right,
                      ExecutionTime=elapsed, MeanExecutionTime=elapsed/max(attempts,1),
                      ResectPct=pct, ToleranceDigits=digits, ObjectiveType1=first[0],
                      ObjectiveSubnetwork1=first[1], ObjectiveType2=second[0],
                      ObjectiveSubnetwork2=second[1], complete=finished)
        write_json(output, result)
        if not finished: raise RuntimeError('Frontier time limit reached; partial output saved')
    finally:
        solver.model.dispose()
