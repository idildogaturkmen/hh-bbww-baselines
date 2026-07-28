#!/usr/bin/env bash
# Worker entry point for the frozen HH4b ttbar7 full-chain scaleout contract.
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 MEMBER_INDEX MEMBER_NAME SEED GENERATED_EVENTS" >&2
  exit 64
fi

MEMBER_INDEX="$1"
MEMBER_NAME="$2"
SEED="$3"
GENERATED_EVENTS="$4"

case "${MEMBER_INDEX}:${MEMBER_NAME}:${SEED}:${GENERATED_EVENTS}" in
  324:ttbar_100k_shard001:105001:10000) ;;
  325:ttbar_100k_shard002:105002:10000) ;;
  347:ttbar_100k_shard004:105004:10000) ;;
  348:ttbar_100k_shard005:105005:10000) ;;
  327:ttbar_100k_shard007:105007:10000) ;;
  328:ttbar_100k_shard008:105008:10000) ;;
  329:ttbar_100k_shard009:105009:10000) ;;
  *)
    echo "ERROR: arguments do not match a frozen member contract" >&2
    exit 65
    ;;
esac

SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
cd "$SCRATCH"

PAYLOAD_TARBALL="${HH4B_TTBAR7_INPUT_TARBALL:-hh4b_ttbar7_exact_regeneration_scaleout_inputs.tar.gz}"
PYENV_TARBALL="${HH4B_TTBAR7_PYENV_TARBALL:-python39_site_packages.tar.gz}"

EXPECTED_PYENV_SHA256="6f47dbb493681b424342045f38cd514a56dba056e41ce2cce04790d99ce515cd"

LCG_SETUP="/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh"
BASE_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

OUTPUT_TAG="${MEMBER_NAME}_pythia8_delphes"
RUN_NAME="run_ttbar_100k_${MEMBER_NAME##*shard}"
OUTDIR="$SCRATCH/output"
WORKDIR="$SCRATCH/work"

mkdir -p "$OUTDIR"/{lhe,hepmc,root,parquet,logs,receipts,checksums} "$WORKDIR"

finalize() {
  local status=$?

  trap - EXIT
  set +e

  {
    echo "exit_status=$status"
    echo "campaign=hh4b_ttbar7_exact_regeneration_scaleout_20260728_v1"
    echo "member_index=$MEMBER_INDEX"
    echo "member_name=$MEMBER_NAME"
    echo "seed=$SEED"
    echo "full_chain_regeneration=true"
    echo "scaleout_job=true"
    echo "finished_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  } > "$OUTDIR/receipts/${MEMBER_NAME}_final_status.txt"

  exit "$status"
}
trap finalize EXIT

if [[ ! -s "$PAYLOAD_TARBALL" ]]; then
  echo "ERROR: missing transfer payload: $PAYLOAD_TARBALL" >&2
  exit 66
fi

if [[ ! -s "$PYENV_TARBALL" ]]; then
  echo "ERROR: missing portable Python environment: $PYENV_TARBALL" >&2
  exit 66
fi

if [[ "$(sha256sum "$PYENV_TARBALL" | awk '{print $1}')" != "$EXPECTED_PYENV_SHA256" ]]; then
  echo "ERROR: portable Python environment SHA256 mismatch" >&2
  exit 66
fi

if [[ ! -r "$LCG_SETUP" ]]; then
  echo "ERROR: missing LCG setup: $LCG_SETUP" >&2
  exit 67
fi

tar -xzf "$PAYLOAD_TARBALL"
PAYLOAD="$SCRATCH/payload"
if [[ ! -s "$PAYLOAD/SHA256SUMS" ]]; then
  echo "ERROR: payload checksum manifest missing" >&2
  exit 68
fi
(cd "$PAYLOAD" && sha256sum -c SHA256SUMS)

PYTHON_ENV_ROOT="$WORKDIR/python39_environment"
mkdir -p "$PYTHON_ENV_ROOT"
tar -xzf "$PYENV_TARBALL" -C "$PYTHON_ENV_ROOT"

PYTHON_SITE="$PYTHON_ENV_ROOT/site-packages"
PORTABLE_PYTHON="/usr/bin/python3"

if [[ ! -d "$PYTHON_SITE" ]]; then
  echo "ERROR: portable site-packages directory is absent" >&2
  exit 68
fi

if [[ ! -s "$PAYLOAD/bootstrap/sitecustomize.py" ]]; then
  echo "ERROR: payload bootstrap/sitecustomize.py is absent" >&2
  exit 68
fi

if [[ ! -s "$PAYLOAD/repo/scripts/delphes/write_parquet_from_pickle.py" ]]; then
  echo "ERROR: isolated Parquet writer is absent" >&2
  exit 68
fi

env   LD_LIBRARY_PATH="$BASE_LD_LIBRARY_PATH"   PYTHONNOUSERSITE=1   PYTHONPATH="$PYTHON_SITE:$PAYLOAD/bootstrap"   "$PORTABLE_PYTHON" - <<'PYENV'
import awkward
import numpy
import pandas
import pyarrow
import uproot

assert numpy.__version__ == "1.26.4"
assert awkward.__version__ == "2.6.4"
assert uproot.__version__ == "5.3.7"
assert pandas.__version__ == "2.2.2"
assert pyarrow.__version__ == "15.0.0"

print("FULL_CHAIN_PORTABLE_ENVIRONMENT_PASS")
PYENV

set +u
# shellcheck disable=SC1090
source "$LCG_SETUP"
set -u

export HH4B_REPO="$PAYLOAD/repo"
export HH4B_STORE="$SCRATCH/store"
export HH4B_SOFTWARE="$PAYLOAD/software"
export DELPHES_DIR="$PAYLOAD/Delphes"
export LD_LIBRARY_PATH="$DELPHES_DIR:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

for command_name in python3 gzip sha256sum tar; do
  command -v "$command_name" >/dev/null
done
ldd "$HH4B_SOFTWARE/bin/lhe_to_hepmc3" | tee "$OUTDIR/logs/${MEMBER_NAME}_converter_ldd.log"
ldd "$DELPHES_DIR/DelphesHepMC3" | tee "$OUTDIR/logs/${MEMBER_NAME}_delphes_ldd.log"
if grep -q "not found" "$OUTDIR/logs/${MEMBER_NAME}_converter_ldd.log" \
  "$OUTDIR/logs/${MEMBER_NAME}_delphes_ldd.log"; then
  echo "ERROR: unresolved runtime library" >&2
  exit 69
fi

PROCESS_DIR="$WORKDIR/TTbar_smoke"
cp -a "$PAYLOAD/mg5_template" "$PROCESS_DIR"
mkdir -p "$PROCESS_DIR/Events" "$PROCESS_DIR/HTML"

test "$(sha256sum "$PROCESS_DIR/Cards/proc_card_mg5.dat" | awk '{print $1}')" \
  = "9c771a8b9c596f02c9a21953936a2bacb2268f04253d38ef974e6ff6bea8bbf8"
test "$(sha256sum "$PROCESS_DIR/Cards/param_card.dat" | awk '{print $1}')" \
  = "5315ff912514491596e9720b0540dbacd61d85d66dec2b4c94eaee0c15a52b41"
test "$(sha256sum "$PROCESS_DIR/Cards/run_card.dat" | awk '{print $1}')" \
  = "5b6b7f660badb79d25218fb4920c801794e62861fb8cf53311d75588786ef04d"

python3 - "$PROCESS_DIR/Cards/run_card.dat" "$GENERATED_EVENTS" "$SEED" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
events = sys.argv[2]
seed = sys.argv[3]
text = path.read_text()
settings = {
    "nevents": events,
    "iseed": seed,
    "ptb": "25.0",
    "etab": "2.7",
    "drbb": "0.4",
    "drbj": "0.4",
    "ptj": "20.0",
    "etaj": "5.0",
}
for key, value in settings.items():
    pattern = rf"(?m)^.*=\s*{re.escape(key)}\b.*$"
    replacement = f" {value} = {key} ! frozen exact-regeneration override"
    if not re.search(pattern, text):
        raise SystemExit(f"missing run-card key: {key}")
    text = re.sub(pattern, replacement, text)
path.write_text(text)
PY

(
  cd "$PROCESS_DIR"
  ./bin/generate_events "$RUN_NAME" -f \
    2>&1 | tee "$OUTDIR/logs/${MEMBER_NAME}_generate_events.log"
)

SOURCE_LHE_GZ="$PROCESS_DIR/Events/$RUN_NAME/unweighted_events.lhe.gz"
if [[ ! -s "$SOURCE_LHE_GZ" ]]; then
  echo "ERROR: generator did not produce LHE" >&2
  exit 70
fi
LHE_COUNT="$(gzip -cd "$SOURCE_LHE_GZ" | grep -c '<event>')"
if [[ "$LHE_COUNT" != "$GENERATED_EVENTS" ]]; then
  echo "ERROR: LHE count $LHE_COUNT != $GENERATED_EVENTS" >&2
  exit 71
fi

LHE_GZ="$OUTDIR/lhe/${MEMBER_NAME}_unweighted_events.lhe.gz"
LHE="$WORKDIR/${MEMBER_NAME}_unweighted_events.lhe"
HEPMC="$OUTDIR/hepmc/${MEMBER_NAME}_pythia8.hepmc"
ROOT_OUT="$OUTDIR/root/${MEMBER_NAME}_pythia8_delphes.root"
CANDIDATE="$OUTDIR/parquet/${MEMBER_NAME}_pythia8_delphes_canonical72.parquet"
cp "$SOURCE_LHE_GZ" "$LHE_GZ"
gzip -cd "$LHE_GZ" > "$LHE"

"$HH4B_SOFTWARE/bin/lhe_to_hepmc3" "$LHE" "$HEPMC" "$GENERATED_EVENTS" \
  2>&1 | tee "$OUTDIR/logs/${MEMBER_NAME}_lhe_to_hepmc3.log"
if ! grep -q "This is PYTHIA version 8.312" \
  "$OUTDIR/logs/${MEMBER_NAME}_lhe_to_hepmc3.log"; then
  echo "ERROR: Pythia 8.312 completion evidence missing" >&2
  exit 72
fi

"$DELPHES_DIR/DelphesHepMC3" \
  "$HH4B_REPO/cards/delphes/delphes_card_CMS_lpc.tcl" \
  "$ROOT_OUT" "$HEPMC" \
  2>&1 | tee "$OUTDIR/logs/${MEMBER_NAME}_pythia8_delphes.log"
if ! grep -q "\\*\\* Exiting" "$OUTDIR/logs/${MEMBER_NAME}_pythia8_delphes.log"; then
  echo "ERROR: Delphes completion evidence missing" >&2
  exit 73
fi

env \
  LD_LIBRARY_PATH="$BASE_LD_LIBRARY_PATH" \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="$PYTHON_SITE:$PAYLOAD/bootstrap" \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  "$PORTABLE_PYTHON" "$HH4B_REPO/scripts/delphes/reconstruct_hh4b_candidates_v2.py" \
  --input "$ROOT_OUT" \
  --out "$CANDIDATE" \
  --sample "$OUTPUT_TAG" \
  --target-mass 125.0 \
  --jet-pt-min 30.0 \
  --jet-eta-max 2.5 \
  --btag-min 0.0 \
  --max-bjets-for-pairing 8 \
  --higgs-ordering pt \
  2>&1 | tee "$OUTDIR/logs/${MEMBER_NAME}_canonical72.log"

env \
  LD_LIBRARY_PATH="$BASE_LD_LIBRARY_PATH" \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="$PYTHON_SITE:$PAYLOAD/bootstrap" \
  "$PORTABLE_PYTHON" - "$ROOT_OUT" "$CANDIDATE" "$HH4B_REPO/schema/canonical72_columns.tsv" \
  "$GENERATED_EVENTS" "$OUTPUT_TAG" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import uproot

root_path = Path(sys.argv[1])
candidate_path = Path(sys.argv[2])
schema_path = Path(sys.argv[3])
expected_events = int(sys.argv[4])
expected_sample = sys.argv[5]

with uproot.open(root_path) as source:
    entries = int(source["Delphes"].num_entries)
if entries != expected_events:
    raise SystemExit(f"Delphes entries {entries} != {expected_events}")

schema_rows = [
    line.split("\t")
    for line in schema_path.read_text().splitlines()[1:]
    if line.strip()
]
expected_columns = [row[1] for row in schema_rows]
expected_types = {row[1]: row[2] for row in schema_rows}
table = pq.read_table(candidate_path)
if table.column_names != expected_columns or len(table.column_names) != 72:
    raise SystemExit("canonical-72 schema/order mismatch")
observed_types = {field.name: str(field.type) for field in table.schema}
if observed_types != expected_types:
    raise SystemExit("canonical-72 Arrow type mismatch")
frame = table.to_pandas()
if frame.duplicated(["sample", "event"]).any():
    raise SystemExit("duplicate sample/event key")
if len(frame) and set(frame["sample"]) != {expected_sample}:
    raise SystemExit("source/member identity mismatch")
for column in frame.select_dtypes(include=[np.number]).columns:
    if not np.isfinite(frame[column].to_numpy()).all():
        raise SystemExit(f"nonfinite values in {column}")
print(json.dumps({"root_entries": entries, "candidate_rows": len(frame)}))
PY

(
  cd "$OUTDIR"
  find lhe hepmc root parquet logs -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum > "checksums/${MEMBER_NAME}_SHA256SUMS"
)

env \
  LD_LIBRARY_PATH="$BASE_LD_LIBRARY_PATH" \
  "$PORTABLE_PYTHON" - "$OUTDIR" "$MEMBER_INDEX" "$MEMBER_NAME" "$SEED" \
  "$GENERATED_EVENTS" "$OUTPUT_TAG" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
member = sys.argv[3]

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

artifacts = {}
for path in sorted(
    list((output / "lhe").glob("*"))
    + list((output / "hepmc").glob("*"))
    + list((output / "root").glob("*"))
    + list((output / "parquet").glob("*"))
):
    artifacts[str(path.relative_to(output))] = {
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    }
receipt = {
    "campaign": "hh4b_ttbar7_exact_regeneration_scaleout_20260728_v1",
    "member_index": int(sys.argv[2]),
    "member_name": member,
    "seed": int(sys.argv[4]),
    "generated_events": int(sys.argv[5]),
    "sample_identity": sys.argv[6],
    "artifacts": artifacts,
    "byte_identical_root_claimed": False,
    "portable_python_environment_sha256":
        "6f47dbb493681b424342045f38cd514a56dba056e41ce2cce04790d99ce515cd",
    "full_chain_regeneration": True,
    "scaleout_job": True,
    "validated_canary_member": "ttbar_100k_shard003",
    "validated_canary_job": "3655099.0",
    "member_validation_status":
        "pass_pending_cross_member_scaleout_validation",
}
(output / "receipts" / f"{member}_receipt.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n"
)
PY

echo "Member production complete: $MEMBER_NAME"
