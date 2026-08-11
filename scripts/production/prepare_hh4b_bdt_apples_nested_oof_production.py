#!/usr/bin/env python3
"""Build, never submit, the immutable ten-job global-BDT production package."""
from __future__ import annotations
import csv,hashlib,json,shutil
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
STORE=Path('/uscms_data/d3/iturkmen/hh4b_delphes')
CAMPAIGN='hh4b_bdt_apples_nested_oof10_v2_20260811'
OUT=STORE/'condor_submit'/CAMPAIGN
PILOT=STORE/'condor_submit'/'hh4b_bdt_apples_full_budget_pilot_outer0_blind_v2_20260811'
REMOTE='/store/user/iturkmen/hh4b_delphes/run2_13tev/bdt_apples_to_apples/hh4b_bdt_apples_full_budget_pilot_outer0_blind_v2_20260811'
FILES=[REPO/'scripts/analysis/run_hh4b_bdt_apples_to_apples_outer.py',REPO/'scripts/analysis/hh4b_bdt_apples_to_apples_common.py',REPO/'configs/baselines/hh4b_bdt_apples_to_apples_v1.json',REPO/'docs/analysis/hh4b_bdt_apples_to_apples_v1_features.tsv',PILOT/'portable_primary_plan.tsv',PILOT/'physical_weight_authorization_registry_464.tsv']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if OUT.exists(): raise RuntimeError('production namespace already exists')
 OUT.mkdir(parents=True)
 for p in FILES: shutil.copy2(p,OUT/p.name)
 jobs=[(f,v) for f in range(5) for v in ('global_mass_aware','global_mass_plane_blind')]
 with (OUT/'jobs.tsv').open('w',newline='') as h:
  w=csv.writer(h,delimiter='\t',lineterminator='\n'); w.writerows(jobs)
 wrapper=OUT/'run_production.sh'
 wrapper.write_text(f'''#!/usr/bin/bash
set -euo pipefail
outer="$1"; variant="$2"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
xrdcp --force root://cmseos.fnal.gov/{REMOTE}/primary_resolved_441.tar data.tar
xrdcp --force root://cmseos.fnal.gov/{REMOTE}/python39_bdt_cpu_runtime.tar.gz runtime.tar.gz
tar -xf data.tar
tar -xzf runtime.tar.gz
export PYTHONPATH="$PWD/site-packages"
/usr/bin/time -v -o "time_${{outer}}_${{variant}}.txt" /usr/bin/python3 run_hh4b_bdt_apples_to_apples_outer.py --outer-fold "$outer" --variant "$variant" --output-dir result --protocol hh4b_bdt_apples_to_apples_v1.json --plan portable_primary_plan.tsv --authorization physical_weight_authorization_registry_464.tsv --feature-registry hh4b_bdt_apples_to_apples_v1_features.tsv
tar -czf "result_${{outer}}_${{variant}}.tar.gz" result "time_${{outer}}_${{variant}}.txt"
'''); wrapper.chmod(0o755)
 submit=OUT/'production.submit'
 submit.write_text(f'''universe = vanilla
executable = {wrapper}
arguments = $(outer_fold) $(variant)
initialdir = {OUT}
log = logs/$(outer_fold)_$(variant).condor.log
output = logs/$(outer_fold)_$(variant).stdout
error = logs/$(outer_fold)_$(variant).stderr
request_cpus = 8
request_memory = 6144MB
request_disk = 5242880KB
+JobBatchName = "{CAMPAIGN}"
+CampaignId = "{CAMPAIGN}"
+OuterFold = $(outer_fold)
+BdtVariant = "$(variant)"
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = run_hh4b_bdt_apples_to_apples_outer.py,hh4b_bdt_apples_to_apples_common.py,hh4b_bdt_apples_to_apples_v1.json,hh4b_bdt_apples_to_apples_v1_features.tsv,portable_primary_plan.tsv,physical_weight_authorization_registry_464.tsv
transfer_output_files = result_$(outer_fold)_$(variant).tar.gz,time_$(outer_fold)_$(variant).txt
on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
queue outer_fold,variant from jobs.tsv
''')
 (OUT/'logs').mkdir()
 inventory={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.iterdir() if p.is_file()}
 contract={'status':'built_not_submitted','campaign':CAMPAIGN,'job_count':10,'outer_folds':list(range(5)),'variants':['global_mass_aware','global_mass_plane_blind'],'candidate_count':8,'inner_folds':4,'rows':1042397,'sources':441,'signal_rows':90638,'background_rows':951759,'pilot_cluster':3798575,'pilot_audit':'PASS','pilot_included_in_science':False,'validation_payloads_opened':0,'test_payloads_opened':0,'resources':{'cpus':8,'memory_mb':6144,'disk_kib':5242880},'remote_assets':REMOTE,'inventory':inventory}
 (OUT/'production_package_contract.json').write_text(json.dumps(contract,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':'built_not_submitted','path':str(OUT)}))
if __name__=='__main__': main()
