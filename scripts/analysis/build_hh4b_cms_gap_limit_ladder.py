#!/usr/bin/env python3
"""Build the frozen train-only R_HH<34 expected-limit ladder."""
import csv,hashlib,json
from pathlib import Path
import pyarrow.parquet as pq
from hh4b_expected_limit import expected_upper_limit
ROOT=Path(__file__).resolve().parents[2]
METRICS=ROOT/'artifacts/hh4b_cut_baseline/train_performance/tables/fixed_selection_source_metrics.tsv'
INVENTORY=ROOT/'artifacts/hh4b_cut_baseline/train_performance/manifests/fold_table_inventory.tsv'
OUT=ROOT/'artifacts/hh4b_cms_sensitivity_gap_v1'; EDGES=[0.,10.,20.,30.,34.]
def categories():
 d={c:{k:0. for k in ('s','b','s2','b2')} for c in ('exact3tag','ge4tag')}
 for r in csv.DictReader(METRICS.open(),delimiter='\t'):
  if r['selection_id']!='historical_rhh125125_lt34' or r['scope'] not in d: continue
  p='s' if r['sample_class']=='signal' else 'b'; d[r['scope']][p]+=float(r['selected_signed_yield']); d[r['scope']][p+'2']+=float(r['selected_sumw2'])
 return d
def shapes():
 d={(c,i):{k:0. for k in ('s','b','s2','b2')} for c in ('exact3tag','ge4tag') for i in range(4)}
 for item in csv.DictReader(INVENTORY.open(),delimiter='\t'):
  p=Path(item['path'])
  if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']: raise RuntimeError(f'checksum mismatch: {p}')
  cols=pq.read_table(p,columns=['sample_class','physical_evaluation_eligible','r_hh_125_125','resolved_selection_contribution_weight']).to_pydict()
  for cl,ok,r,w in zip(cols['sample_class'],cols['physical_evaluation_eligible'],cols['r_hh_125_125'],cols['resolved_selection_contribution_weight']):
   if not ok or w is None or r is None or not 0<=r<34: continue
   x=d[(item['category_id'],min(3,int(r//10)))]; q='s' if cl=='signal' else 'b'; x[q]+=w; x[q+'2']+=w*w
 return d
def rec(name,x):
 s=[z['s'] for z in x]; b=[z['b'] for z in x]; v=[z['b2'] for z in x]
 return {'stage':name,'bins':len(x),'signal_yield':sum(s),'background_yield':sum(b),'background_sumw2':sum(v),'background_neff':sum(b)**2/sum(v),'mu95_stat_only':expected_upper_limit(s,b),'mu95_finite_mc_aware':expected_upper_limit(s,b,v)}
def main():
 c=categories(); h=shapes(); pooled=[{k:sum(x[k] for x in c.values()) for k in ('s','b','s2','b2')}]
 ladder=[rec('pooled_one_bin',pooled),rec('exact3tag_ge4tag',list(c.values())),rec('exact3tag_ge4tag_rhh_fixed_shape',[h[k] for k in sorted(h)])]
 for m in ('mu95_stat_only','mu95_finite_mc_aware'):
  ladder[1]['gain_vs_pooled_'+m]=ladder[0][m]/ladder[1][m]; ladder[2]['gain_vs_categories_'+m]=ladder[1][m]/ladder[2][m]
 data={'schema_version':1,'label':'Delphes simulation expected-limit diagnostic','selection':'R_HH(125,125) < 34','shape_binning':{'variable':'r_hh_125_125','edges':EDGES,'rule':'predeclared fixed physics intervals'},'inference':{'method':'asymptotic median background-only CLs','target_sqrt_qmu':1.959963984540054,'finite_mc_model':'independent Gaussian-constrained background per bin; sigma=sqrt(sumw2)','experimental_systematics':'none'},'validation_payloads_opened':0,'test_payloads_opened':0,'ladder':ladder}
 OUT.mkdir(parents=True,exist_ok=True); (OUT/'expected_limit_ladder.json').write_text(json.dumps(data,indent=2)+'\n')
 with (OUT/'expected_limit_ladder.tsv').open('w',newline='') as f:
  fields=['stage','bins','signal_yield','background_yield','background_sumw2','background_neff','mu95_stat_only','mu95_finite_mc_aware']; w=csv.DictWriter(f,fields,delimiter='\t',extrasaction='ignore',lineterminator='\n'); w.writeheader(); w.writerows(ladder)
 print(json.dumps(data,indent=2))
if __name__=='__main__': main()
