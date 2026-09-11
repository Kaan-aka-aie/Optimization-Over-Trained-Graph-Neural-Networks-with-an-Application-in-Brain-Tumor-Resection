"""Recompute the 1,052 archived stability observations without optimization."""
import json
import numpy as np
from data import ROOT, load_instance, indices, propagation, residual, sigma_matrix, write_json
from models import weights


def run(output, limit=None):
    mapping = json.loads((ROOT/'config/instances.json').read_text())['biobjective']
    pairs = [('modularity','overall'),('ge','Visual_Network'),('ge','Multiple_Demand_Extended_Network')]
    rows = []
    for instance_id in mapping.values():
        d = load_instance(instance_id)
        sigma = sigma_matrix(d)
        for second in pairs:
            filename = f'ge-overallvs{second[0]}-{second[1]}-80.json'
            path = ROOT/'reference/EpsilonResultsFixed'/f'instance_{instance_id}'/filename
            if not path.exists(): continue
            archive = json.loads(path.read_text())
            plans = [archive['LeftCorner']] + archive['IterationResults'] + [archive['RightCorner']]
            for plan_index, plan in enumerate(plans):
                removed = plan['MIP Removed Nodes']
                for obj, sub in [('ge','overall'),second]:
                    ix = indices(d, sub)
                    x = residual(d, removed)[ix]
                    if obj == 'modularity': x = x + sigma
                    theta,bias,beta,bout = weights(obj,sub,'stability')
                    ps = {name:propagation(d,sub,nodes) for name,nodes in
                          [('exact',removed),('fully',range(len(d.volumes))),('initial',[])]}
                    zs = {name:p@x@theta.T+bias for name,p in ps.items()}
                    predictions = {name:float((beta@np.maximum(z,0).mean(0)+bout).item()) for name,z in zs.items()}
                    k = np.linalg.norm(beta,2)*np.linalg.norm(theta,2)*np.linalg.norm(x,'fro')/np.sqrt(len(ix))
                    row = dict(instance=instance_id,obj=obj,subnetwork=sub,source=str(path.relative_to(ROOT)),
                               plan_index=plan_index,removed_nodes=removed,orig_A=predictions['exact'],
                               fully_resected_A=predictions['fully'],initial_A=predictions['initial'])
                    for baseline in ('fully','initial'):
                        delta=zs['exact']-zs[baseline]
                        active=float(np.sum(np.abs(delta*beta)*((zs['exact']>0)|(zs[baseline]>0)))/len(ix))
                        row[baseline+'_error']=abs(predictions['exact']-predictions[baseline])
                        row[baseline+'_active_bound']=active
                        row[baseline+'_lipschitz']=float(k*np.linalg.norm(ps['exact']-ps[baseline],'fro'))
                        # Archived code compared initial versus fully for the initial Lipschitz column.
                        row[baseline+'_archived_lipschitz']=float(k*np.linalg.norm(
                            ps['initial']-ps['fully'] if baseline=='initial' else ps['exact']-ps['fully'],'fro'))
                        assert row[baseline+'_error'] <= active+1e-10
                    rows.append(row)
        print(f'Stability instance {instance_id}: {len(rows)} observations',flush=True)
    lookup = {(r['source'],r['plan_index'],r['obj'],r['subnetwork']):r for r in rows}
    selection = json.loads((ROOT/'config/stability_observations.json').read_text())
    rows = [dict(lookup[(r['source'],r['plan_index'],r['objective'],r['subnetwork'])], observation=r['observation'], workbook_index=r['workbook_index']) for r in selection]
    if limit: rows = rows[:limit]
    write_json(output,rows)
    return rows
