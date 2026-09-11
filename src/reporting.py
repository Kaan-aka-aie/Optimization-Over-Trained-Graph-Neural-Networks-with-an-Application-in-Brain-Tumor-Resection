"""Rebuild numerical tables and figures from saved plans, without a solver."""
import json
import csv
import numpy as np
import openpyxl
from scipy.spatial.distance import cdist
from data import ROOT, load_instance, metric, indices, partition, write_json

OUT=ROOT/'outputs/tables'


def save_csv(name,rows):
    OUT.mkdir(parents=True,exist_ok=True)
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def nondominated(points):
    p=np.asarray(points,float)
    keep=[not np.any(np.all(p>=x,axis=1)&np.any(p>x,axis=1)) for x in p]
    return np.unique(p[keep],axis=0)


def plans(record):return [record['LeftCorner']]+record['IterationResults']+[record['RightCorner']]


def single_rows():
    raw=json.loads((ROOT/'reference/surrogate_vs_original.json').read_text());rows=[]
    for key, record in raw.items():
        instance=load_instance(key)
        for sub in ('overall','Visual_Network','Multiple_Demand_Extended_Network'):
            a=record[f'mipge-{sub}'];b=record[f'mnrpge-{sub}']
            ra=a['MIP Removed Nodes'];rb=b['MNRP Removed Nodes']
            va=metric(instance,ra,'ge',sub);vb=metric(instance,rb,'ge',sub)
            rows.append(dict(instance=int(key),subnetwork=sub,surrogate_prediction=a[f'MIP ge {sub}'],
                             surrogate_true=va,exact_true=vb,gap_pct=100*(vb-va)/vb,
                             surrogate_construction_s=a['Total Construction Time'],surrogate_s=a['MIP Time'],
                             exact_construction_s=b['Constr Time'],exact_s=b['MNRP Optimization Time'],
                             jaccard=len(set(ra)&set(rb))/len(set(ra)|set(rb)),
                             archived_exact_objective=b[sub]))
    save_csv('single_objective.csv',rows)
    return rows


def frontier_rows():
    jobs=json.loads((ROOT/'config/biobjective.json').read_text());rows=[];coordinates=[]
    for job in jobs:
        i=job['instance'];d=load_instance(i);(o1,s1),(o2,s2)=job['objectives']
        folder=ROOT/f'reference/EpsilonResultsFixed/instance_{i}'
        name=f'{o1}-{s1}vs{o2}-{s2}-80.json'
        a=json.loads((folder/name).read_text());b=json.loads((folder/('MNRP-'+name)).read_text())
        denominators=np.array([metric(d,[],o1,s1),metric(d,[],o2,s2)])
        p=[];q=[]
        for method,record,target in [('surrogate',a,p),('exact',b,q)]:
            for n,r in enumerate(plans(record)):
                removed=r['MIP Removed Nodes' if method=='surrogate' else 'MNRP Removed Nodes'] if isinstance(r,dict) else None
                xy=100*np.array([metric(d,removed,o1,s1),metric(d,removed,o2,s2)] if removed is not None else r)/denominators
                target.append(xy)
                coordinates.append(dict(paper_id=job['paper_id'],instance=i,target=s2,method=method,
                                        plan=n,x_pct=xy[0],y_pct=xy[1],removed_nodes=json.dumps(removed)))
        p=nondominated(p);q=nondominated(q);dist=cdist(p,q)
        haus=max(dist.min(0).max(),dist.min(1).max());igd=dist.min(0).mean();diag=np.linalg.norm(q.max(0)-q.min(0))
        rows.append(dict(paper_id=job['paper_id'],instance=i,target=s2,surrogate_s=a['ExecutionTime'],
                         exact_s=b['ExecutionTime'],surrogate_iterations=round(a['ExecutionTime']/a['MeanExecutionTime']),
                         exact_iterations=round(b['ExecutionTime']/b['MeanExecutionTime']),hausdorff=haus,igd=igd,directed_surrogate_to_exact=dist.min(1).max(),diagonal=diag,
                         hausdorff_pct=100*haus/diag if diag>1e-9 else None,igd_pct=100*igd/diag if diag>1e-9 else None))
    save_csv('biobjective.csv',rows);save_csv('biobjective_coordinates.csv',coordinates)
    return rows,coordinates


def workbook_stability():
    w=openpyxl.load_workbook(ROOT/'reference/workbooks/constant_adjacency_experiment.xlsx',data_only=True)
    data=list(w['total'].values);head=data[0]
    rows=[dict(zip(head,r)) for r in data[1:] if isinstance(r[1],(int,float))]
    output=[]
    for label,subset in [('All',rows)]+[(o+' '+s,[r for r in rows if r['obj']==o and r['subnetwork']==s])
                         for o,s in [('ge','overall'),('ge','Visual_Network'),('ge','Multiple_Demand_Extended_Network'),('modularity','overall')]]:
        values={}
        for baseline,source in [('fully','fully_resected_A'),('initial','initial_A')]:
            values[baseline+'_error_pct']=[100*abs(r['orig_A']-r[source])/abs(r['orig_A']) for r in subset]
            bound='error_bound_fully_resectedA' if baseline=='fully' else 'error_bound_initialA'
            lip='error_bound_theoretical_max_error' if baseline=='fully' else 'error_bound_theoretical_max_error_initial'
            values[baseline+'_active_pct']=[100*r[bound]/abs(r['orig_A']) for r in subset]
            values[baseline+'_archived_lipschitz_pct']=[100*r[lip]/abs(r['orig_A']) for r in subset]
        for stat,fn in [('mean',np.mean),('std',lambda x:np.std(x,ddof=1)),('max',np.max)]:
            output.append(dict(category=label,statistic=stat,n=len(subset),**{k:float(fn(v)) for k,v in values.items()}))
    save_csv('stability_archived_summary.csv',output)
    return rows,output


def geometry():
    allrows=[];coords=[];cross=[]
    for i in [168,170]:
        d=load_instance(i);df=d.tumorous_region;communities=partition(d)
        core=df[df.tumor_density>=1];periphery=df[df.tumor_density<1]
        reverse={int(roi):idx for idx,roi in enumerate(d.new_order)}
        metrics=[('ge','overall'),('ge','Visual_Network'),('ge','Multiple_Demand_Extended_Network'),('modularity','overall')]
        initial={s if o=='ge' else o:metric(d,[],o,s,communities) for o,s in metrics}
        for radius in range(int(np.ceil(periphery.distance_to_center.max()))):
            rois=list(core.ROI_ID)+list(periphery[periphery.distance_to_center<=radius].ROI_ID)
            removed=[reverse[int(roi)-1] for roi in rois]
            for obj,sub in metrics:
                value=metric(d,removed,obj,sub,communities)
                allrows.append(dict(instance=i,radius_mm=radius,resect_pct=100*d.volumes[removed].sum()/d.volumes.sum(),
                                    objective=obj,subnetwork=sub,value=value,pct_initial=100*value/initial[sub if obj=='ge' else obj],
                                    removed_nodes=json.dumps(removed)))
        folder=ROOT/f'reference/SurgeryEpsilonResults/instance_{i}'
        for path in sorted(folder.glob('*vs*.json')):
            record=json.loads(path.read_text());method='exact' if path.name.startswith('MNRP-') else 'surrogate'
            objs=[(record[f'ObjectiveType{k}'],record[f'ObjectiveSubnetwork{k}']) for k in (1,2)]
            values=[]
            for n,r in enumerate(plans(record)):
                removed=r['MNRP Removed Nodes' if method=='exact' else 'MIP Removed Nodes'] if isinstance(r,dict) else None
                xy=[100*(metric(d,removed,o,s,communities) if removed is not None else r[k])/initial[s if o=='ge' else o] for k,(o,s) in enumerate(objs)]
                values.append(xy)
                # Retain resection sets for reproducible anatomy mapping.
                coords.append(dict(instance=i,method=method,target=objs[1][0]+' '+objs[1][1],
                                   resect_pct=record['ResectPct'],plan=n,x_pct=xy[0],y_pct=xy[1],
                                   removed_nodes=json.dumps(removed),roi_ids=json.dumps((d.new_order[removed]+1).tolist()) if removed is not None else None,
                                   source=path.name))
    save_csv('geometric_expansion.csv',allrows);save_csv('case_frontier_coordinates.csv',coords)
    # Cross-evaluate every archived plan by ROI identity; Table 21 omits its threshold.
    instances={i:load_instance(i) for i in [168,170]}
    for row in coords:
        if row['removed_nodes']=='null':continue
        source=instances[row['instance']];rois=source.new_order[json.loads(row['removed_nodes'])]
        for target in [168,170]:
            d=instances[target];removed=np.flatnonzero(np.isin(d.new_order,rois)).tolist()
            cross.append(dict(source_instance=row['instance'],target_instance=target,method=row['method'],
                              source=row['source'],plan=row['plan'],resect_pct=row['resect_pct'],
                              overall_ge=metric(d,removed,'ge','overall'),mden_ge=metric(d,removed,'ge','Multiple_Demand_Extended_Network')))
    save_csv('cross_connectome_all_plans.csv',cross)
    table21()
    print('Wrote geometry, case frontiers and cross-connectome evaluations',flush=True)


def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=single_rows();bio,coords=frontier_rows();stability,stats=workbook_stability()
    geometry()
    figures=ROOT/'outputs/figures';figures.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    selected=[r for r in rows if r['subnetwork']=='overall']
    axes[0].scatter([r['exact_true'] for r in selected],[r['surrogate_true'] for r in selected]);axes[0].plot([.4,.65],[.4,.65],color='gray')
    axes[0].set(xlabel='Exact GE',ylabel='GE of surrogate plan')
    axes[1].hist([r['gap_pct'] for r in selected],bins=15);axes[1].set(xlabel='Objective gap (%)',ylabel='Instances')
    fig.savefig(figures/'figure3_exact_comparison.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained');ax.scatter([r['surrogate_s'] for r in bio],[r['exact_s'] for r in bio]);ax.plot([0,2200],[0,2200],color='gray');ax.set(xlabel='MGBTRP frontier time (s)',ylabel='Exact frontier time (s)');fig.savefig(figures/'figure4_frontier_time.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for method in ['exact','surrogate']:
        p=nondominated([[r['x_pct'],r['y_pct']] for r in coords if r['paper_id']==20 and r['method']==method]);ax.plot(p[:,0],p[:,1],'-o',label=method)
    ax.set(xlabel='Whole-brain GE (% of initial)',ylabel='Visual GE (% of initial)');ax.legend();fig.savefig(figures/'figure5_instance20.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for name,col in [('Fully resected','fully_resected_A'),('Initial','initial_A')]:
        e=[100*abs(r['orig_A']-r[col])/abs(r['orig_A']) for r in stability];axes[0].hist(e,bins=40,density=True,histtype='step',label=name)
    axes[0].set(xlabel='Realized error (%)',ylabel='Density');axes[0].legend()
    x=[abs(r['orig_A']-r['fully_resected_A']) for r in stability];y=[r['error_bound_fully_resectedA'] for r in stability];axes[1].scatter(x,y,s=5);axes[1].plot([0,max(y)],[0,max(y)],color='gray');axes[1].set(xlabel='Realized error',ylabel='Active-subspace bound');fig.savefig(figures/'figure2_stability.png',dpi=200);plt.close(fig)
    summary=dict(single_instances=len(set(r['instance'] for r in rows)),single_comparisons=len(rows),
                 whole_brain_mean_gap_pct=float(np.mean([r['gap_pct'] for r in selected])),
                 whole_brain_max_gap_pct=max(r['gap_pct'] for r in selected),
                 biobjective_runs=len(bio),stability_observations=len(stability),
                 biobjective_mean_surrogate_s=float(np.mean([r['surrogate_s'] for r in bio])),
                 biobjective_mean_exact_s=float(np.mean([r['exact_s'] for r in bio])))
    paper_workbook_rows()
    additional_figures(rows)
    write_json(ROOT/'outputs/summary.json',summary);print(json.dumps(summary,indent=2))


def table21():
    pct=69.69441243166938
    instances={i:load_instance(i) for i in (168,170)}
    rows=[]
    for method in ('exact','surrogate'):
        for objective,corner in [('overall','RightCorner'),('Multiple_Demand_Extended_Network','LeftCorner')]:
            row=dict(method=method,objective=objective,resect_pct=pct)
            for source in (168,170):
                path=ROOT/f'reference/SurgeryEpsilonResults/instance_{source}/{"MNRP-" if method=="exact" else ""}ge-overallvsge-Multiple_Demand_Extended_Network-{pct}.json'
                record=json.loads(path.read_text())
                plan=record['IterationResults'][-1] if method=='surrogate' and source==168 and corner=='RightCorner' else record[corner]
                removed=plan['MIP Removed Nodes' if method=='surrogate' else 'MNRP Removed Nodes']
                rois=instances[source].new_order[removed]
                for target in (168,170):
                    d=instances[target];selected=np.flatnonzero(np.isin(d.new_order,rois)).tolist()
                    row[f'evaluate_{target}_plan_{source}']=metric(d,selected,'ge',objective)
            rows.append(row)
    save_csv('table21_cross_connectome.csv',rows)
    return rows


def paper_workbook_rows():
    w=openpyxl.load_workbook(ROOT/'reference/workbooks/Bi-Obj Results Jun10.xlsx',data_only=True)
    data=list(w['Consolidated for Paper'].values)
    rows=[dict(zip(data[0],r)) for r in data[1:] if isinstance(r[0],(int,float)) and r[2]=='ge']
    save_csv('biobjective_archived_paper_sheet.csv',rows)


def additional_figures(rows):
    import matplotlib.pyplot as plt
    figures=ROOT/'outputs/figures'
    fig,axes=plt.subplots(3,2,figsize=(10,12),layout='constrained')
    for k,sub in enumerate(('overall','Visual_Network','Multiple_Demand_Extended_Network')):
        selected=[r for r in rows if r['subnetwork']==sub]
        x=[r['exact_true'] for r in selected];y=[r['surrogate_true'] for r in selected]
        axes[k,0].scatter(x,y);lo=min(x+y);hi=max(x+y);axes[k,0].plot([lo,hi],[lo,hi],color='gray');axes[k,0].set(xlabel='Exact GE',ylabel='GE of surrogate plan',title=sub)
        x=[r['surrogate_s']+r['surrogate_construction_s'] for r in selected];y=[r['exact_s']+r['exact_construction_s'] for r in selected]
        axes[k,1].scatter(x,y);hi=max(x+y);axes[k,1].plot([0,hi],[0,hi],color='gray');axes[k,1].set(xlabel='Surrogate total time (s)',ylabel='Exact total time (s)')
    fig.savefig(figures/'figure9_objectives_and_times.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained');ax.hist([r['jaccard'] for r in rows],bins=20,density=True);ax.set(xlabel='Jaccard similarity',ylabel='Density');fig.savefig(figures/'figure10_jaccard.png',dpi=200);plt.close(fig)
