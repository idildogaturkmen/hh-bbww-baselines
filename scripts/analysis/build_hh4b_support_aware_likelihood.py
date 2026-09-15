#!/usr/bin/env python3
"""Deterministic adjacent merging based only on QCD Neff thresholds."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; IN=ROOT/'artifacts/hh4b_cms_sensitivity_gap_v2/qcd_tail_bin_support.tsv'; OUT=ROOT/'artifacts/hh4b_cms_sensitivity_gap_v3'
import sys; sys.path.insert(0,str(ROOT/'scripts/analysis'))
from hh4b_expected_limit import expected_upper_limit
def merge(a,b): return {k:(a[k]+b[k] if k in ('signal_yield','background_yield','background_sumw2','qcd_yield','qcd_sumw2','qcd_rows') else a[k]) for k in a}
def main():
 raw=[]
 for r in csv.DictReader(IN.open(),delimiter='\t'): raw.append({'category':r['category'],'low':float(r['rhh_low']),'high':float(r['rhh_high']),'signal_yield':float(r['signal_yield']),'background_yield':float(r['background_yield']),'background_sumw2':float(r['background_sumw2']),'qcd_yield':float(r['qcd_yield']),'qcd_sumw2':float(r['qcd_sumw2']),'qcd_rows':int(r['qcd_rows'])})
 results=[]; binsout=[]
 for threshold in (5,10,20):
  merged=[]
  for cat in ('exact3tag','ge4tag'):
   acc=None
   for x in [z for z in raw if z['category']==cat]:
    acc=x.copy() if acc is None else merge(acc,x); acc['high']=x['high']
    neff=acc['qcd_yield']**2/acc['qcd_sumw2'] if acc['qcd_sumw2'] else 0.
    if neff>=threshold: merged.append(acc); acc=None
   if acc is not None:
    prior=next((i for i in range(len(merged)-1,-1,-1) if merged[i]['category']==cat),None)
    if prior is None: merged.append(acc)
    else:
     low=merged[prior]['low']; merged[prior]=merge(merged[prior],acc); merged[prior]['low']=low; merged[prior]['high']=acc['high']
  mu=expected_upper_limit([x['signal_yield'] for x in merged],[x['background_yield'] for x in merged]); achievable=all((x['qcd_yield']**2/x['qcd_sumw2'] if x['qcd_sumw2'] else 0.)>=threshold for x in merged)
  results.append({'minimum_qcd_neff_threshold':threshold,'merged_bins':len(merged),'threshold_achievable_all_bins':achievable,'mu95_stat_only_support_merge':mu,'gain_vs_two_channel':44.32758473485289/mu,'fraction_nominal_shape_gain_retained':(44.32758473485289-mu)/(44.32758473485289-31.95647727235911)})
  for x in merged: binsout.append({'threshold':threshold,**x,'qcd_neff':x['qcd_yield']**2/x['qcd_sumw2'] if x['qcd_sumw2'] else 0.})
 OUT.mkdir(parents=True,exist_ok=True)
 for name,data in [('support_aware_likelihood_summary.tsv',results),('support_aware_merged_bins.tsv',binsout)]:
  with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,list(data[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(data)
 payload={'schema_version':1,'rule':'left-to-right adjacent accumulation until QCD Neff threshold; merge terminal residual backward; no sensitivity input','results':results,'validation_payloads_opened':0,'test_payloads_opened':0};(OUT/'support_aware_likelihood_summary.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
