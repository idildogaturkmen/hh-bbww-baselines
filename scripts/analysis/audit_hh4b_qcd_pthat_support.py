#!/usr/bin/env python3
"""Aggregate physical train QCD support by authoritative campaign/pTHat."""
import csv,json
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'artifacts/hh4b_cms_sensitivity_gap_v3'
PLAN=ROOT/'docs/checkpoints/hh4b_train_common_table_full_output_freeze_20260805_v1/evidence/production_manifest/full_464_production_plan.tsv'
HARD=ROOT/'docs/checkpoints/hh4b_physical_normalization_hard_qcd_global_weight_contract_freeze_20260731_v1/hard_qcd_shard_weight_contract.tsv'
def fresh(): return {'physical_sources':0,'generated_events':0,'broad_resolved_rows':0,'2b_rows':0,'3b_rows':0,'ge4_rows':0,'rhh34_rows':0,'rhh34_yield':0.,'rhh34_sumw2':0.,'maximum_physical_weight':0.}
def main():
 hard={r['source_tag']:r for r in csv.DictReader(HARD.open(),delimiter='\t') if r['production_included']=='True'}; agg=defaultdict(fresh); critical=defaultdict(lambda:{'rows':0,'yield':0.,'sumw2':0.})
 for r in csv.DictReader(PLAN.open(),delimiter='\t'):
  if r['process_or_mode']!='qcd_hardqcd': continue
  meta=next(x for tag,x in hard.items() if tag in r['source_uid']); key=(meta['campaign'],meta['pthat_min_GeV'],meta['pthat_max_GeV']); z=agg[key]; z['physical_sources']+=1; z['generated_events']+=int(r['generated_events'])
  cols=['n_selected_bjets','r_hh_125_125','resolved_selection_contribution_weight']; d=pq.read_table(r['final_resolved_path'],columns=cols).to_pydict(); z['broad_resolved_rows']+=len(d['n_selected_bjets'])
  for n,rhh,w in zip(d['n_selected_bjets'],d['r_hh_125_125'],d['resolved_selection_contribution_weight']):
   if n==2:z['2b_rows']+=1
   if n==3:z['3b_rows']+=1
   if n>=4:z['ge4_rows']+=1
   if n>=3 and rhh<34:
    z['rhh34_rows']+=1; z['rhh34_yield']+=w; z['rhh34_sumw2']+=w*w; z['maximum_physical_weight']=max(z['maximum_physical_weight'],abs(w))
   if n>=4 and rhh<34:
    lo=0 if rhh<10 else 10 if rhh<20 else 20 if rhh<30 else 30
    q=critical[(key,lo)]; q['rows']+=1; q['yield']+=w; q['sumw2']+=w*w
 total=sum(x['rhh34_yield'] for x in agg.values()); rows=[]
 for (camp,lo,hi),z in sorted(agg.items()): rows.append({'campaign':camp,'pthat_min_GeV':lo,'pthat_max_GeV':hi,**z,'rhh34_neff':z['rhh34_yield']**2/z['rhh34_sumw2'] if z['rhh34_sumw2'] else 0.,'fraction_total_selected_qcd_yield':z['rhh34_yield']/total,'typical_selected_physical_weight':z['rhh34_yield']/z['rhh34_rows'] if z['rhh34_rows'] else 0.})
 crit=[]
 for ((camp,plo,phi),rlo),z in sorted(critical.items()): crit.append({'campaign':camp,'pthat_min_GeV':plo,'pthat_max_GeV':phi,'rhh_low':rlo,'rhh_high':34 if rlo==30 else rlo+10,**z,'neff':z['yield']**2/z['sumw2'] if z['sumw2'] else 0.})
 OUT.mkdir(parents=True,exist_ok=True)
 for name,data in [('qcd_pthat_campaign_support.tsv',rows),('qcd_pthat_ge4_rhh_support.tsv',crit)]:
  with (OUT/name).open('w',newline='') as f: w=csv.DictWriter(f,list(data[0]),delimiter='\t',lineterminator='\n'); w.writeheader(); w.writerows(data)
 summary={'schema_version':1,'strata':len(rows),'campaigns':len(set(x['campaign'] for x in rows)),'selected_qcd_yield':total,'dominant_strata':sorted(rows,key=lambda x:x['rhh34_yield'],reverse=True)[:5],'validation_payloads_opened':0,'test_payloads_opened':0}; (OUT/'qcd_pthat_support_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
