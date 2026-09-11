"""Validate asset identity and independently re-evaluate saved solutions."""
import json
import hashlib
import platform
import importlib.metadata
import numpy as np
import networkx as nx
from data import ROOT, load_instance, metric, write_json, residual
from models import predict


def run():
    manifest=json.loads((ROOT/'config/provenance.json').read_text())
    for entry in manifest:
        actual=hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest()
        if actual!=entry['sha256']:raise AssertionError(f'Asset changed: {entry["path"]}')
    mapping=json.loads((ROOT/'config/instances.json').read_text())
    ids=sorted(set(mapping['single_objective']+list(mapping['biobjective'].values())+list(mapping['case_study'].values())))
    for i in ids:
        d=load_instance(i);e=d.E.toarray()
        assert e.shape==(360,360) and np.array_equal(e,e.T)
        assert np.array_equal(np.sort(d.new_order),np.arange(360))
        assert len(d.volumes)==len(d.tumorous_roi_ids)
        assert np.all(d.volumes>=0)
        assert np.array_equal(d.A.toarray(),e+np.eye(360))
    reference=json.loads((ROOT/'reference/surrogate_vs_original.json').read_text())
    errors=[];surrogate_errors=[];feasible=0
    for key,records in reference.items():
        d=load_instance(key)
        for name,row in records.items():
            is_surrogate=name.startswith('mip')
            removed=row['MIP Removed Nodes' if is_surrogate else 'MNRP Removed Nodes']
            assert set(removed)<=set(range(len(d.volumes)))
            assert d.volumes[removed].sum()+1e-5>=0.8*d.volumes.sum()
            assert nx.is_connected(nx.Graph(d.E_beta[np.ix_(removed,removed)]))
            feasible+=1
            sub=name.split('ge-',1)[1]
            value=metric(d,removed,'ge',sub)
            if is_surrogate:
                surrogate_errors.append(abs(predict(d,removed,'ge',sub)-row[f'MIP ge {sub}']))
            else:errors.append(abs(value-row[sub]))
        print(f'Validated saved solutions for instance {key}',flush=True)
    # Historical MIP feasibility tolerances can cause small differences; identify, never hide, deviations.
    max_surrogate=max(surrogate_errors);max_exact=max(errors)
    assert max_surrogate<1e-6,(max_surrogate,'Checkpoint or embedding mismatch')
    assert max_exact<1e-6,(max_exact,'Exact objective mismatch')
    stability_path=ROOT/'outputs/stability.json'
    stability_check=None
    if stability_path.exists():
        import openpyxl
        w=openpyxl.load_workbook(ROOT/'reference/workbooks/constant_adjacency_experiment.xlsx',data_only=True)
        rows=list(w['total'].values);expected=[dict(zip(rows[0],r)) for r in rows[1:] if isinstance(r[1],(int,float))]
        actual=json.loads(stability_path.read_text());assert len(actual)==len(expected)==1052
        stability_check=max(abs(a[c]-b[c]) for a,b in zip(actual,expected) for c in ('orig_A','initial_A','fully_resected_A'))
        assert stability_check<1e-8
    versions={n:importlib.metadata.version(n) for n in ['numpy','scipy','pandas','networkx','torch','torch-geometric','gurobipy','openpyxl']}
    report=dict(asset_hashes_checked=len(manifest),instances_checked=len(ids),saved_feasible_plans=feasible,
                max_saved_exact_objective_error=max_exact,max_saved_surrogate_prediction_error=max_surrogate,
                stability_max_prediction_error=stability_check,python=platform.python_version(),versions=versions)
    write_json(ROOT/'outputs/validation.json',report);print(json.dumps(report,indent=2))
