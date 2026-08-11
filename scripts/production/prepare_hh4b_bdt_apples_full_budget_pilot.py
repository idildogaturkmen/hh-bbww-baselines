#!/usr/bin/env python3
"""Build (but never submit) the frozen full-budget global-BDT pilot package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
STORE = Path("/uscms_data/d3/iturkmen/hh4b_delphes")
CAMPAIGN = "hh4b_bdt_apples_full_budget_pilot_outer0_blind_v1_20260811"
PACKAGE = STORE / "condor_submit" / CAMPAIGN
ASSETS = STORE / "bdt_apples_to_apples" / "portable_assets" / CAMPAIGN
PROTOCOL = REPO / "configs/baselines/hh4b_bdt_apples_to_apples_v1.json"
FEATURES = REPO / "docs/analysis/hh4b_bdt_apples_to_apples_v1_features.tsv"
PLAN = REPO / "docs/checkpoints/hh4b_train_common_table_full_output_freeze_20260805_v1/evidence/production_manifest/full_464_production_plan.tsv"
AUTH = REPO / "docs/checkpoints/hh4b_train_run2_138fb_physical_authorization_freeze_20260806_v1/evidence/authorization/physical_weight_authorization_registry_464.tsv"
VENV_SITE = Path("/uscms_data/d3/iturkmen/venvs/hh4b-bdt-v1/lib/python3.9/site-packages")
BASE_SITE = Path("/uscms_data/d3/iturkmen/hh4b_delphes/runtime/hh4b_broad_ml_py39_v1/lib/python3.9/site-packages")
REMOTE_ROOT = f"/store/user/iturkmen/hh4b_delphes/run2_13tev/bdt_apples_to_apples/{CAMPAIGN}"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_tree(archive: tarfile.TarFile, source: Path, target: str) -> None:
    archive.add(source, arcname=target, recursive=True)


def build_runtime(path: Path) -> None:
    base_names = ["pandas", "pandas-2.3.3.dist-info", "pyarrow", "pyarrow-21.0.0.dist-info", "pytz", "python_dateutil-2.9.0.post0.dist-info", "dateutil", "six.py", "six-1.17.0.dist-info", "tzdata", "tzdata-2025.2.dist-info"]
    overlay_names = ["numpy", "numpy.libs", "numpy-1.26.4.dist-info", "scipy", "scipy.libs", "scipy-1.13.1.dist-info", "sklearn", "scikit_learn.libs", "scikit_learn-1.6.1.dist-info", "joblib", "joblib-1.5.3.dist-info", "threadpoolctl.py", "threadpoolctl-3.6.0.dist-info", "xgboost", "xgboost.libs", "xgboost-2.1.4.dist-info"]
    with tarfile.open(path, "w:gz", compresslevel=6) as archive:
        for name in base_names:
            source = BASE_SITE / name
            if source.exists():
                add_tree(archive, source, f"site-packages/{name}")
        for name in overlay_names:
            source = VENV_SITE / name
            require(source.exists(), f"missing frozen runtime component: {source}")
            add_tree(archive, source, f"site-packages/{name}")


def build_data(path: Path, portable_plan: Path) -> None:
    with PLAN.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    with AUTH.open() as handle:
        auth = {row["source_uid"]: row for row in csv.DictReader(handle, delimiter="\t")}
    primary = [row for row in rows if row["auxiliary_qcd"] == "False" and auth[row["source_uid"]]["physical_weight_application_authorized"] == "True"]
    require(len(primary) == 441, "primary source count changed")
    fieldnames = list(rows[0])
    with portable_plan.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if row in primary:
                row = dict(row)
                row["final_resolved_path"] = f"resolved/{Path(row['final_resolved_path']).name}"
            writer.writerow(row)
    with tarfile.open(path, "w") as archive:
        for row in primary:
            source = Path(row["final_resolved_path"])
            require(source.is_file(), f"missing resolved table: {source}")
            archive.add(source, arcname=f"resolved/{source.name}", recursive=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    require(args.build, "pass --build explicitly")
    require(not PACKAGE.exists() and not ASSETS.exists(), "pilot package namespace already exists")
    PACKAGE.mkdir(parents=True)
    ASSETS.mkdir(parents=True)
    shutil.copy2(PROTOCOL, PACKAGE / PROTOCOL.name)
    shutil.copy2(FEATURES, PACKAGE / FEATURES.name)
    shutil.copy2(AUTH, PACKAGE / "physical_weight_authorization_registry_464.tsv")
    shutil.copy2(REPO / "scripts/analysis/run_hh4b_bdt_apples_to_apples_outer.py", PACKAGE)
    shutil.copy2(REPO / "scripts/analysis/hh4b_bdt_apples_to_apples_common.py", PACKAGE)
    portable_plan = PACKAGE / "portable_primary_plan.tsv"
    data_archive = ASSETS / "primary_resolved_441.tar"
    runtime_archive = ASSETS / "python39_bdt_cpu_runtime.tar.gz"
    build_data(data_archive, portable_plan)
    build_runtime(runtime_archive)
    wrapper = PACKAGE / "run_pilot.sh"
    wrapper.write_text(f'''#!/usr/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
xrdcp --force root://cmseos.fnal.gov/{REMOTE_ROOT}/primary_resolved_441.tar data.tar
xrdcp --force root://cmseos.fnal.gov/{REMOTE_ROOT}/python39_bdt_cpu_runtime.tar.gz runtime.tar.gz
tar -xf data.tar
tar -xzf runtime.tar.gz
export PYTHONPATH="$PWD/site-packages"
/usr/bin/time -v -o pilot_time.txt /usr/bin/python3 run_hh4b_bdt_apples_to_apples_outer.py --outer-fold 0 --variant global_mass_plane_blind --output-dir pilot_result --protocol {PROTOCOL.name} --plan portable_primary_plan.tsv --authorization physical_weight_authorization_registry_464.tsv
tar -czf pilot_result.tar.gz pilot_result pilot_time.txt
''', encoding="utf-8")
    wrapper.chmod(0o755)
    submit = PACKAGE / "pilot.submit"
    submit.write_text(f'''universe = vanilla
executable = {wrapper}
initialdir = {PACKAGE}
log = pilot.condor.log
output = pilot.stdout
error = pilot.stderr
request_cpus = 8
request_memory = 16384MB
request_disk = 8192MB
+JobBatchName = "{CAMPAIGN}"
+CampaignId = "{CAMPAIGN}"
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = run_hh4b_bdt_apples_to_apples_outer.py,hh4b_bdt_apples_to_apples_common.py,{PROTOCOL.name},{FEATURES.name},portable_primary_plan.tsv,physical_weight_authorization_registry_464.tsv
transfer_output_files = pilot_result.tar.gz,pilot_time.txt
on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
queue 1
''', encoding="utf-8")
    inventory = []
    for path in sorted([*PACKAGE.iterdir(), *ASSETS.iterdir()]):
        inventory.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    contract = {"campaign": CAMPAIGN, "role": "full_budget_operational_pilot_not_scientific_input", "outer_fold": 0, "variant": "global_mass_plane_blind", "candidate_count": 8, "inner_folds": 4, "cpus": 8, "memory_mb": 16384, "disk_mb": 8192, "remote_root": REMOTE_ROOT, "validation_payloads_opened": 0, "test_payloads_opened": 0, "production_submission": False, "inventory": inventory}
    (PACKAGE / "pilot_package_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "pilot_package_built_not_submitted", "package": str(PACKAGE), "assets": str(ASSETS)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
