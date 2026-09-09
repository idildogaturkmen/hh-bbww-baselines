#!/usr/bin/env bash
# Fail-closed master postflight gate for the exact_2M ParT production.
#
# Composes existing, already-tested project scripts rather than duplicating
# them (project convention):
#   1. census_2m.py                    -- H5/receipt presence, dup/orphan detection, checkpoint hash
#   2. aggregate_embedding_audit.py    -- O(n_shards) manifest-window disjointness + uniqueness proof
#   3. cross_check_manifest_counts.py  -- NEW: manifest's own declared counts match FROZEN_FACTS.json
#   4. verify_shard_checksums.py       -- NEW: re-hash shard content vs receipt.output_sha256 (sample by default)
#
# Read-only throughout. Never writes under any production shards/ directory.
# Exits 0 only if every step passes. Run this fresh -- do not reuse a stale
# census from a prior session.
#
# Every step is run with `set +e` around it and its exit code captured
# explicitly (a step failing here is an EXPECTED outcome while production
# is incomplete, not a script bug -- `set -e` must not abort the gate
# before all four steps have been attempted and recorded).
set -uo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FROZEN_FACTS="${PKG_ROOT}/configs/FROZEN_FACTS.json"
CENSUS_SCRIPT="/uscms_data/d3/iturkmen/hh4b_delphes/track_b_part_postflight_2m_20260907_v1/census/census_2m.py"
AGGREGATE_SCRIPT="/uscms_data/d3/iturkmen/hh4b_delphes/track_b_part_postproduction_tools_20260826_v2/code/aggregate_embedding_audit.py"
CROSS_CHECK_SCRIPT="${PKG_ROOT}/scripts/postflight/cross_check_manifest_counts.py"
CHECKSUM_SCRIPT="${PKG_ROOT}/scripts/postflight/verify_shard_checksums.py"

MANIFEST="/uscms_data/d3/iturkmen/hh4b_delphes/track_b_part_full_embedding_production_exact_2M_20260824_v1/manifest/SHARD_MANIFEST.json"
EOS_ROOT="root://cmseos.fnal.gov//store/user/iturkmen/hh4b_delphes/track_b_part_full_embedding_production_exact_2M_20260824_v1/shards"
CKPT_SHA256="61e752f80d7c237d4b18b97705df416a8518dd9e3d5a78a8bbdeebadd787fec0"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${1:-${PKG_ROOT}/logs/postflight_${TS}}"
CHECKSUM_SAMPLE_N="${CHECKSUM_SAMPLE_N:-20}"   # set env CHECKSUM_SAMPLE_N=full for a full pre-freeze run

mkdir -p "${OUT_DIR}"
echo "=== Postflight gate ${TS} -> ${OUT_DIR} ==="

for f in "${FROZEN_FACTS}" "${CENSUS_SCRIPT}" "${AGGREGATE_SCRIPT}" "${MANIFEST}"; do
  if [ ! -f "${f}" ]; then
    echo "FATAL: required file missing: ${f}" >&2
    exit 10
  fi
done

echo "--- Step 1/4: fresh EOS census (read-only xrdfs ls) ---"
xrdfs cmseos.fnal.gov ls "${EOS_ROOT}/train" > "${OUT_DIR}/train_listing.txt"
xrdfs cmseos.fnal.gov ls "${EOS_ROOT}/val"   > "${OUT_DIR}/val_listing.txt"
set +e
python3 "${CENSUS_SCRIPT}" \
  "${OUT_DIR}/train_listing.txt" "${OUT_DIR}/val_listing.txt" \
  "${CKPT_SHA256}" \
  "${OUT_DIR}/census.json"
CENSUS_RC=$?
set -e
python3 -c "import json; d=json.load(open('${OUT_DIR}/census.json')); print('train:',d['train']['complete_pairs'],'/',d['train']['expected'],' val:',d['val']['complete_pairs'],'/',d['val']['expected'],' OVERALL_CENSUS_PASS:',d['OVERALL_CENSUS_PASS'])"

if [ "${CENSUS_RC}" -ne 0 ]; then
  echo "Census FAILED -- production is incomplete. Steps 2-4 still run (for diagnostic value) but the gate is already FAIL."
fi

echo "--- Step 2/4: aggregate disjointness/uniqueness audit (memory-bounded, O(n_shards)) ---"
set +e
python3 "${AGGREGATE_SCRIPT}" \
  --manifest "${MANIFEST}" \
  --eos-root "${EOS_ROOT}" \
  --out "${OUT_DIR}/aggregate_audit.json"
AGG_RC=$?
set -e
python3 -c "import json; d=json.load(open('${OUT_DIR}/aggregate_audit.json')); print('AGGREGATE_PASS:', d.get('AGGREGATE_PASS'))" 2>/dev/null || echo "AGGREGATE_PASS: <could not parse output -- see ${OUT_DIR}/aggregate_audit.json>"

echo "--- Step 3/4: manifest declared-count cross-check vs FROZEN_FACTS.json ---"
set +e
python3 "${CROSS_CHECK_SCRIPT}" \
  --manifest "${MANIFEST}" \
  --frozen-facts "${FROZEN_FACTS}" \
  --census-json "${OUT_DIR}/census.json" \
  --out "${OUT_DIR}/manifest_cross_check.json"
CROSS_RC=$?
set -e

echo "--- Step 4/4: checksum spot-check (sample=${CHECKSUM_SAMPLE_N}; set env CHECKSUM_SAMPLE_N=full for a full pre-freeze run) ---"
set +e
if [ "${CHECKSUM_SAMPLE_N}" = "full" ]; then
  python3 "${CHECKSUM_SCRIPT}" --manifest "${MANIFEST}" --split train --eos-root "${EOS_ROOT}" --full --out "${OUT_DIR}/checksum_train.json"
  CKSUM_TRAIN_RC=$?
  python3 "${CHECKSUM_SCRIPT}" --manifest "${MANIFEST}" --split val   --eos-root "${EOS_ROOT}" --full --out "${OUT_DIR}/checksum_val.json"
  CKSUM_VAL_RC=$?
else
  python3 "${CHECKSUM_SCRIPT}" --manifest "${MANIFEST}" --split train --eos-root "${EOS_ROOT}" --sample-only "${CHECKSUM_SAMPLE_N}" --out "${OUT_DIR}/checksum_train.json"
  CKSUM_TRAIN_RC=$?
  python3 "${CHECKSUM_SCRIPT}" --manifest "${MANIFEST}" --split val   --eos-root "${EOS_ROOT}" --sample-only "${CHECKSUM_SAMPLE_N}" --out "${OUT_DIR}/checksum_val.json"
  CKSUM_VAL_RC=$?
fi
set -e

OVERALL_PASS="true"
[ "${CENSUS_RC}" -ne 0 ] && OVERALL_PASS="false"
[ "${AGG_RC}" -ne 0 ] && OVERALL_PASS="false"
[ "${CROSS_RC}" -ne 0 ] && OVERALL_PASS="false"
[ "${CKSUM_TRAIN_RC}" -ne 0 ] && OVERALL_PASS="false"
[ "${CKSUM_VAL_RC}" -ne 0 ] && OVERALL_PASS="false"
true  # neutralize any nonzero status left by the last `[ ... ]` test above under set -e/pipefail

python3 - "${OUT_DIR}" "${OVERALL_PASS}" "${TS}" "${CENSUS_RC}" "${AGG_RC}" "${CROSS_RC}" "${CKSUM_TRAIN_RC}" "${CKSUM_VAL_RC}" <<'PYEOF'
import json, sys, os
out_dir, overall_pass, ts, census_rc, agg_rc, cross_rc, ck_tr_rc, ck_val_rc = sys.argv[1:9]
receipt = {
    "utc": ts,
    "OVERALL_POSTFLIGHT_PASS": overall_pass == "true",
    "step_exit_codes": {
        "census_2m.py": int(census_rc),
        "aggregate_embedding_audit.py": int(agg_rc),
        "cross_check_manifest_counts.py": int(cross_rc),
        "verify_shard_checksums.py (train)": int(ck_tr_rc),
        "verify_shard_checksums.py (val)": int(ck_val_rc),
    },
    "artifacts": sorted(os.listdir(out_dir)),
}
with open(os.path.join(out_dir, "GATE_RESULT.json"), "w") as f:
    json.dump(receipt, f, indent=2)
print(json.dumps(receipt, indent=2))
PYEOF

echo "OVERALL_POSTFLIGHT_PASS=${OVERALL_PASS}" | tee "${OUT_DIR}/GATE_RESULT.txt" > /dev/null
echo "OVERALL_POSTFLIGHT_PASS=${OVERALL_PASS}"
[ "${OVERALL_PASS}" = "true" ]
