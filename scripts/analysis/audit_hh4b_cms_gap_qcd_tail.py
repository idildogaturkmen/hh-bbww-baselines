#!/usr/bin/env python3
"""Audit frozen train-only RHH shape bins and authoritative QCD sources."""
import csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq
from hh4b_expected_limit import expected_upper_limit,expected_upper_limit_gamma
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'artifacts/hh4b_cms_sensitivity_gap_v2'; EDGES=[0.,10.,20.,30.,34.]
INV=ROOT/'artifacts/hh4b_cut_baseline/train_performance/manifests/fold_table_inventory.tsv'
MAN=ROOT/'docs/checkpoints/hh4b_train_464_authoritative_broad_ml_access_manifest_20260804_v1/train_464_authoritative_broad_ml_access_manifest_v5.tsv'
HARD=ROOT/'docs/checkpoints/hh4b_physical_normalization_hard_qcd_global_weight_contract_freeze_20260731_v1/hard_qcd_shard_weight_contract.tsv'
def blank(): return {'signal_rows':0,'background_rows':0,'qcd_rows':0,'signal_yield':0.,'background_yield':0.,'qcd_yield':0.,'background_sumw2':0.,'qcd_sumw2':0.,'negative_rows':0,'negative_yield':0.,'events':[],'sources':defaultdict(float)}
def main():
 bins={(c,i):blank() for c in ('exact3tag','ge4tag') for i in range(4)}; source=defaultdict(lambda:{'exact3_rows':0,'ge4_rows':0,'selected_rows':0,'selected_yield':0.,'selected_sumw2':0.,'max_abs_weight':0.,'resolved_rows':0,'resolved_yield':0.})
 for item in csv.DictReader(INV.open(),delimiter='\t'):
  p=Path(item['path'])
  if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']: raise RuntimeError(f'checksum mismatch {p}')
  cols=['source_uid','sample_class','process_or_mode','physical_evaluation_eligible','r_hh_125_125','resolved_selection_contribution_weight']
  d=pq.read_table(p,columns=cols).to_pydict()
  for uid,cl,proc,ok,r,w in zip(*[d[x] for x in cols]):
   if not ok or w is None: continue
   if proc=='qcd_hardqcd':
    z=source[uid]; z['resolved_rows']+=1; z['resolved_yield']+=w; z['exact3_rows' if item['category_id']=='exact3tag' else 'ge4_rows']+=1
   if r is None or not 0<=r<34: continue
   z=bins[(item['category_id'],min(3,int(r//10)))]; q=proc=='qcd_hardqcd'
   if cl=='signal': z['signal_rows']+=1; z['signal_yield']+=w
   else:
    z['background_rows']+=1; z['background_yield']+=w; z['background_sumw2']+=w*w; z['sources'][uid]+=w; z['events'].append(abs(w))
    if q: z['qcd_rows']+=1; z['qcd_yield']+=w; z['qcd_sumw2']+=w*w
    if w<0: z['negative_rows']+=1; z['negative_yield']+=w
   if q:
    x=source[uid]; x['selected_rows']+=1; x['selected_yield']+=w; x['selected_sumw2']+=w*w; x['max_abs_weight']=max(x['max_abs_weight'],abs(w))
 rows=[]
 for (cat,i),z in bins.items():
  fr=sorted((y/z['background_yield'],u) for u,y in z['sources'].items())[::-1]; s,b=z['signal_yield'],z['background_yield']
  qneff=z['qcd_yield']**2/z['qcd_sumw2'] if z['qcd_sumw2'] else 0.
  rows.append({'category':cat,'rhh_low':EDGES[i],'rhh_high':EDGES[i+1],**{k:v for k,v in z.items() if k not in ('events','sources')},'nonqcd_yield':b-z['qcd_yield'],'background_neff':b*b/z['background_sumw2'],'qcd_neff':qneff,'s_over_b':s/b,'small_signal_information_s2_over_b':s*s/b,'contributing_source_groups':len(z['sources']),'largest_source_fraction':fr[0][0],'top5_source_fractions':','.join(f'{x[0]:.12g}' for x in fr[:5]),'largest_event_abs_weight_fraction':max(z['events'])/sum(z['events']),'importance_qcd_yield':z['qcd_yield'],'ordinary_physical_qcd_yield':0.,'support':'UNSUPPORTED' if (qneff<20 or fr[0][0]>.5) else 'LIMITED' if qneff<100 else 'SUPPORTED'})
 total_info=sum(x['small_signal_information_s2_over_b'] for x in rows)
 for x in rows: x['fraction_small_signal_information']=x['small_signal_information_s2_over_b']/total_info
 hard={r['source_tag']:r for r in csv.DictReader(HARD.open(),delimiter='\t') if r['production_included']=='True'}
 srcrows=[]
 for m in csv.DictReader(MAN.open(),delimiter='\t'):
  if 'qcd' not in m['process_or_mode']: continue
  uid=m['source_uid']; x=source[uid]; role='importance_tail_physical' if m['process_or_mode']=='qcd_hardqcd' else 'auxiliary_nonphysical'
  meta=next((r for tag,r in hard.items() if tag in m['source_locator']),None)
  na='not_applicable'; srcrows.append({'source_uid':uid,'process_or_mode':m['process_or_mode'],'source_role':role,'generated_events':m['generated_events'],'resolved_exact3_ge4_rows':x['resolved_rows'],'exact3_rows':x['exact3_rows'],'ge4_rows':x['ge4_rows'],'resolved_exact3_ge4_expected_yield':x['resolved_yield'],'rhh34_selected_rows':x['selected_rows'],'selected_yield':x['selected_yield'],'selected_sumw2':x['selected_sumw2'],'selected_neff':x['selected_yield']**2/x['selected_sumw2'] if x['selected_sumw2'] else 0.,'typical_selected_weight':x['selected_yield']/x['selected_rows'] if x['selected_rows'] else 0.,'maximum_absolute_selected_weight':x['max_abs_weight'],'selected_fraction_of_resolved':x['selected_rows']/x['resolved_rows'] if x['resolved_rows'] else 0.,'fraction_total_selected_physical_qcd':x['selected_yield']/sum(y['selected_yield'] for y in source.values()),'campaign':meta['campaign'] if meta else 'not_applicable_auxiliary','pthat_min_GeV':meta['pthat_min_GeV'] if meta else na, 'pthat_max_GeV':meta['pthat_max_GeV'] if meta else na, 'sigma_gen_pb':meta['sigma_gen_pb'] if meta else na, 'normalization_denominator_type':meta['normalization_denominator_type'] if meta else na, 'normalization_denominator':meta['normalization_denominator'] if meta else na, 'production_mixture_fraction':meta['production_mixture_fraction'] if meta else na, 'run2_yield_coefficient_per_generator_weight':meta['run2_yield_coefficient_per_generator_weight'] if meta else na})
 OUT.mkdir(parents=True,exist_ok=True)
 def write(name,data):
  with (OUT/name).open('w',newline='') as f:
   w=csv.DictWriter(f,list(data[0]),delimiter='\t',lineterminator='\n'); w.writeheader(); w.writerows(data)
 write('qcd_tail_bin_support.tsv',rows); write('qcd_source_inventory.tsv',srcrows)
 ss=[x['signal_yield'] for x in rows]; bb=[x['background_yield'] for x in rows]; vv=[x['background_sumw2'] for x in rows]
 summary={'schema_version':1,'shape_bins':len(rows),'all_bins_positive':all(x>0 for x in bb),'mu95_stat_only':expected_upper_limit(ss,bb),'mu95_gaussian_mc':expected_upper_limit(ss,bb,vv),'mu95_gamma_effective_count':expected_upper_limit_gamma(ss,bb,vv),'physical_qcd_sources':sum(x['source_role']=='importance_tail_physical' for x in srcrows),'auxiliary_nonphysical_qcd_sources':sum(x['source_role']=='auxiliary_nonphysical' for x in srcrows),'ordinary_physical_qcd_sources':0,'selected_physical_qcd_yield':sum(x['selected_yield'] for x in srcrows if x['source_role']=='importance_tail_physical'),'selected_physical_qcd_rows':sum(x['rhh34_selected_rows'] for x in srcrows if x['source_role']=='importance_tail_physical'),'validation_payloads_opened':0,'test_payloads_opened':0}
 (OUT/'qcd_tail_audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
