#!/usr/bin/env python3
"""Audit immutable broad train tables and close their 3b/4b subsets."""
import csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'artifacts/hh4b_cms_sensitivity_gap_v2'
PLAN=ROOT/'docs/checkpoints/hh4b_train_common_table_full_output_freeze_20260805_v1/evidence/production_manifest/full_464_production_plan.tsv'
FOLDS=ROOT/'artifacts/hh4b_cut_baseline/train_performance/manifests/fold_table_inventory.tsv'
PROCESSES=['ggf_hh4b','qcd_hardqcd','ttbar_inclusive','zbbbb','tth_hbb']
def bucket(n): return str(n) if n<4 else 'ge4'
def main():
 plan=[r for r in csv.DictReader(PLAN.open(),delimiter='\t') if r['auxiliary_qcd']=='False']
 chosen={p:min((r for r in plan if r['process_or_mode']==p),key=lambda r:int(r['production_row_index'])) for p in PROCESSES}
 frozen=defaultdict(lambda:[0,0.,0.])
 for f in csv.DictReader(FOLDS.open(),delimiter='\t'):
  d=pq.read_table(f['path'],columns=['source_uid','resolved_selection_contribution_weight']).to_pydict()
  for u,w in zip(d['source_uid'],d['resolved_selection_contribution_weight']):
   frozen[(u,f['category_id'])][0]+=1; frozen[(u,f['category_id'])][1]+=w; frozen[(u,f['category_id'])][2]+=w*w
 summary=defaultdict(lambda:{'rows':0,'yield':0.,'sumw2':0.,'negative_rows':0}); canary=[]; manifest=[]
 cols=['source_uid','source_fold','event_uid','sample_class','process_or_mode','physical_evaluation_eligible','n_selected_bjets','candidate_tagged_jet_count','resolved_selection_contribution_weight','mbb1','mbb2','r_hh_125_125','mhh','ht_candidate_jets','h1_pt','h2_pt','drbb1','drbb2']
 for r in plan:
  path=Path(r['final_resolved_path']); t=pq.read_table(path,columns=cols); d=t.to_pydict(); uid=r['source_uid']
  manifest.append({'source_uid':uid,'source_fold':r['optimized_fold_k5'],'process_or_mode':r['process_or_mode'],'rows':t.num_rows,'path':str(path),'source_parquet_sha256':r['source_parquet_sha256'],'preserved_columns':','.join(cols)})
  local=defaultdict(lambda:[0,0.,0.])
  for n,w in zip(d['n_selected_bjets'],d['resolved_selection_contribution_weight']):
   b=bucket(n); z=summary[(r['sample_class'],r['process_or_mode'],b)]; z['rows']+=1; z['yield']+=w; z['sumw2']+=w*w; z['negative_rows']+=w<0
   if b in ('3','ge4'): local['exact3tag' if b=='3' else 'ge4tag'][0]+=1; local['exact3tag' if b=='3' else 'ge4tag'][1]+=w; local['exact3tag' if b=='3' else 'ge4tag'][2]+=w*w
  if r is chosen.get(r['process_or_mode']):
   for cat in ('exact3tag','ge4tag'):
    a=local[cat]; b=frozen[(uid,cat)]; canary.append({'process_or_mode':r['process_or_mode'],'source_uid':uid,'category':cat,'existing_broad_rows':a[0],'frozen_fold_rows':b[0],'yield_difference':a[1]-b[1],'sumw2_difference':a[2]-b[2],'closure_pass':a[0]==b[0] and abs(a[1]-b[1])<1e-8 and abs(a[2]-b[2])<1e-6})
 OUT.mkdir(parents=True,exist_ok=True)
 def write(name,rows):
  with (OUT/name).open('w',newline='') as f:
   w=csv.DictWriter(f,list(rows[0]),delimiter='\t',lineterminator='\n'); w.writeheader(); w.writerows(rows)
 dist=[{'sample_class':k[0],'process_or_mode':k[1],'btag_bin':k[2],**v,'neff':v['yield']**2/v['sumw2'] if v['sumw2'] else 0.} for k,v in sorted(summary.items())]
 write('lower_tag_train_distribution.tsv',dist); write('lower_tag_existing_product_manifest.tsv',manifest); write('lower_tag_five_source_canary.tsv',canary)
 data={'schema_version':1,'physical_sources':len(plan),'resolved_rows':sum(x['rows'] for x in manifest),'canary_sources':len(chosen),'canary_category_checks':len(canary),'canary_closure_pass':all(x['closure_pass'] for x in canary),'new_extraction_required':False,'full_condor_campaign_required':False,'validation_payloads_opened':0,'test_payloads_opened':0}
 (OUT/'lower_tag_feasibility_summary.json').write_text(json.dumps(data,indent=2)+'\n'); print(json.dumps(data,indent=2))
if __name__=='__main__': main()
