#!/usr/bin/env python3
"""Predeclared simulation-only QCD 2b->3b/ge4 transfer closure diagnostic."""
import csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np,pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'artifacts/hh4b_cms_sensitivity_gap_v3'
PLAN=ROOT/'docs/checkpoints/hh4b_train_common_table_full_output_freeze_20260805_v1/evidence/production_manifest/full_464_production_plan.tsv'
TRANSFER={'r_hh_125_125':[0,20,40,80,float('inf')],'mhh':[0,300,500,float('inf')],'ht_candidate_jets':[0,250,400,float('inf')]}
AUDIT={'mbb1':[0,60,100,140,200,float('inf')],'mbb2':[0,60,100,140,200,float('inf')],**TRANSFER,'h1_pt':[0,100,200,400,float('inf')],'h2_pt':[0,100,200,400,float('inf')],'max_drbb':[0,1,2,3,float('inf')]}
def ix(x,e): return int(np.searchsorted(e,x,side='right')-1)
def main():
 rows=[]; cols=['source_fold','n_selected_bjets','resolved_selection_contribution_weight','mbb1','mbb2','r_hh_125_125','mhh','ht_candidate_jets','h1_pt','h2_pt','drbb1','drbb2']
 for r in csv.DictReader(PLAN.open(),delimiter='\t'):
  if r['process_or_mode']!='qcd_hardqcd': continue
  d=pq.read_table(r['final_resolved_path'],columns=cols).to_pydict()
  for z in zip(*[d[c] for c in cols]):
   x=dict(zip(cols,z)); x['max_drbb']=max(x['drbb1'],x['drbb2']); rows.append(x)
 result=[]
 modes=[('2b_to_3b',lambda n:n==2,lambda n:n==3,'direct'),('2b_to_ge4',lambda n:n==2,lambda n:n>=4,'direct'),('3b_to_ge4',lambda n:n==3,lambda n:n>=4,'direct'),('sequential_2b_3b_ge4',lambda n:n==2,lambda n:n>=4,'sequential'),('joint_2b3b_to_ge4',lambda n:n in (2,3),lambda n:n>=4,'direct')]
 for label,is_source,is_target,kind in modes:
  num=defaultdict(float); den=defaultdict(float); mid=defaultdict(float)
  for x in rows:
   if x['source_fold']>2: continue
   key=tuple(ix(x[v],TRANSFER[v]) for v in TRANSFER); w=x['resolved_selection_contribution_weight']
   if is_source(x['n_selected_bjets']): den[key]+=w
   if is_target(x['n_selected_bjets']): num[key]+=w
   if x['n_selected_bjets']==3: mid[key]+=w
  if kind=='sequential':
   r23={k:mid[k]/v for k,v in den.items() if v>0}; r34={k:num[k]/v for k,v in mid.items() if v>0}; ratio={k:r23[k]*r34.get(k,0.) for k in r23}
  else: ratio={k:num[k]/v for k,v in den.items() if v>0}
  closure=[x for x in rows if x['source_fold']>2]; source=[x for x in closure if is_source(x['n_selected_bjets'])]; truth=[x for x in closure if is_target(x['n_selected_bjets'])]
  py=sum(x['resolved_selection_contribution_weight']*ratio.get(tuple(ix(x[v],TRANSFER[v]) for v in TRANSFER),0.) for x in source); ty=sum(x['resolved_selection_contribution_weight'] for x in truth)
  tw2=sum(x['resolved_selection_contribution_weight']**2 for x in truth); fold_ratios=[]
  for fold in (3,4):
   sf=[x for x in source if x['source_fold']==fold]; tf=[x for x in truth if x['source_fold']==fold]; p=sum(x['resolved_selection_contribution_weight']*ratio.get(tuple(ix(x[v],TRANSFER[v]) for v in TRANSFER),0.) for x in sf); t=sum(x['resolved_selection_contribution_weight'] for x in tf); fold_ratios.append(p/t if t else None)
  result.append({'transfer':label,'distribution':'yield','predicted':py,'target':ty,'predicted_over_target':py/ty if ty else None,'shape_l1':None,'development_nonzero_cells':len(ratio),'closure_source_rows':len(source),'closure_target_rows':len(truth),'closure_target_neff':ty*ty/tw2 if tw2 else 0.,'fold3_ratio':fold_ratios[0],'fold4_ratio':fold_ratios[1]})
  for var,edges in AUDIT.items():
   pred=np.zeros(len(edges)-1); obs=np.zeros(len(edges)-1)
   for x in source:
    j=ix(x[var],edges)
    if 0<=j<len(pred): pred[j]+=x['resolved_selection_contribution_weight']*ratio.get(tuple(ix(x[v],TRANSFER[v]) for v in TRANSFER),0.)
   for x in truth:
    j=ix(x[var],edges)
    if 0<=j<len(obs): obs[j]+=x['resolved_selection_contribution_weight']
   pn=pred/pred.sum() if pred.sum() else pred; on=obs/obs.sum() if obs.sum() else obs
   result.append({'transfer':label,'distribution':var,'predicted':pred.sum(),'target':obs.sum(),'predicted_over_target':pred.sum()/obs.sum() if obs.sum() else None,'shape_l1':np.abs(pn-on).sum()/2,'development_nonzero_cells':len(ratio),'closure_source_rows':len(source),'closure_target_rows':len(truth),'closure_target_neff':ty*ty/tw2 if tw2 else 0.,'fold3_ratio':fold_ratios[0],'fold4_ratio':fold_ratios[1]})
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'lower_tag_transfer_closure.tsv').open('w',newline='') as f:
  w=csv.DictWriter(f,list(result[0]),delimiter='\t',lineterminator='\n'); w.writeheader(); w.writerows(result)
 y=[x for x in result if x['distribution']=='yield']; clean=lambda d:{k:[('inf' if math.isinf(v) else v) for v in z] for k,z in d.items()}; data={'schema_version':1,'label':'simulation-only lower-tag transfer diagnostic','development_source_folds':[0,1,2],'closure_source_folds':[3,4],'transfer_features':clean(TRANSFER),'audit_features':clean(AUDIT),'results':y,'validation_payloads_opened':0,'test_payloads_opened':0}
 (OUT/'lower_tag_transfer_summary.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n'); print(json.dumps(data,indent=2,allow_nan=False))
if __name__=='__main__': main()
