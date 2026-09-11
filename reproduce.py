#!/usr/bin/env python3
"""Reviewer entry point. All paths resolve relative to this folder."""
from pathlib import Path
import argparse
import sys
import json
import os
ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/.matplotlib'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('verify',help='Check immutable asset hashes and archived numerical results')
    sub.add_parser('summarize',help='Rebuild tables and plots from archived results; no solver')
    stability=sub.add_parser('stability',help='Recompute fixed-propagation errors and bounds')
    stability.add_argument('--limit',type=int)
    single=sub.add_parser('single',help='Rerun GE optimization on the saved instances')
    single.add_argument('--instance',type=int,nargs='+')
    single.add_argument('--method',choices=['surrogate','exact','both'],default='surrogate')
    single.add_argument('--time-limit',type=float,default=1800)
    bio=sub.add_parser('biobjective',help='Rerun the 27 mapped synthetic frontier experiments')
    bio.add_argument('--paper-id',type=int,nargs='+')
    bio.add_argument('--method',choices=['surrogate','exact','both'],default='surrogate')
    bio.add_argument('--time-limit',type=float,default=1800)
    case=sub.add_parser('case-study',help='Rerun patient and HCP frontiers at archived thresholds')
    case.add_argument('--instance',type=int,choices=[168,170],nargs='+',default=[168,170])
    case.add_argument('--method',choices=['surrogate','exact','both'],default='surrogate')
    case.add_argument('--time-limit',type=float,default=1800)
    sub.add_parser('geometry',help='Recompute geometric expansion and cross-connectome evaluation')
    args=parser.parse_args()
    from data import write_json
    mapping=json.loads((ROOT/'config/instances.json').read_text())
    if args.command=='verify':
        from validation import run
        run()
    elif args.command=='summarize':
        from reporting import run
        run()
    elif args.command=='stability':
        from stability import run
        run(ROOT/'outputs/stability.json',args.limit)
    elif args.command=='geometry':
        from reporting import geometry
        geometry()
    else:
        from experiments import single,frontier
        methods=['surrogate','exact'] if args.method=='both' else [args.method]
        if args.command=='single':
            ids=args.instance or mapping['single_objective']
            if not set(ids)<=set(mapping['single_objective']): parser.error('Instance is not in the single-objective cohort')
            for i in ids:
                for method in methods:
                    single(i,method,ROOT/f'outputs/reruns/single_{i}_{method}.json',args.time_limit)
        elif args.command=='biobjective':
            jobs=json.loads((ROOT/'config/biobjective.json').read_text())
            if args.paper_id and not set(args.paper_id)<=set(range(1,21)): parser.error('Paper ID must be 1..20')
            for job in jobs:
                if args.paper_id and job['paper_id'] not in args.paper_id: continue
                for method in methods:
                    label=job['objectives'][1][1]
                    frontier(job['instance'],job['objectives'],job['resect_pct'],method,
                             ROOT/f'outputs/reruns/biobjective_{job["paper_id"]}_{label}_{method}.json',
                             args.time_limit,digits=job['epsilon_digits'])
        else:
            for i in args.instance:
                folder=ROOT/f'reference/SurgeryEpsilonResults/instance_{i}'
                for path in sorted(folder.glob('ge-overallvs*.json')):
                    record=json.loads(path.read_text())
                    objectives=[(record[f'ObjectiveType{k}'],record[f'ObjectiveSubnetwork{k}']) for k in (1,2)]
                    for method in methods:
                        if method=='exact' and any(o=='modularity' for o,s in objectives):continue
                        frontier(i,objectives,record['ResectPct'],method,
                                 ROOT/f'outputs/reruns/case_{i}_{method}_{path.name}',args.time_limit,
                                 digits=record['ToleranceDigits'])

if __name__=='__main__': main()
