#!/usr/bin/env bash
# Full-population event-level JOIN of native SPA-Net input with frozen ParT
# embeddings, gated behind a passed postflight and using explicit
# provenance identity (JOIN_KEY/{native_hdf5_row_index, jet_slot}) at every
# step -- never positional/sequential guessing across files.
#
# Reuses streaming_native_part_join.py unchanged (project convention: do
# not duplicate an already-tested script). This wrapper adds:
#   1. a hard require that the most recent postflight gate passed
#   2. local shard sync (streaming_native_part_join.py needs a LOCAL
#      --part-shard-dir; the production lives on EOS only)
#   3. per-split run + this package's own verify_join_coverage.py gate
#   4. one combined JOIN_RECEIPT.json for both splits
#
# Usage:
#   run_full_join.sh <postflight_gate_result_dir> [work_dir]
set -uo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FROZEN_FACTS="${PKG_ROOT}/configs/FROZEN_FACTS.json"
JOIN_SCRIPT="/uscms_data/d3/iturkmen/hh4b_delphes/track_b_part_postproduction_tools_20260826_v2/code/streaming_native_part_join.py"
VERIFY_SCRIPT="${PKG_ROOT}/scripts/join/verify_join_coverage.py"

MANIFEST="/uscms_data/d3/iturkmen/hh4b_delphes/track_b_part_full_embedding_production_exact_2M_20260824_v1/manifest/SHARD_MANIFEST.json"
EOS_ROOT="root://cmseos.fnal.gov//store/user/iturkmen/hh4b_delphes/track_b_part_full_embedding_production_exact_2M_20260824_v1/shards"
NATIVE_H5_TRAIN="/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4AF_spanet_partial_events_population_correction_20260819_v1/production_hdf5_build_v1/work/production_2M_train.h5"
NATIVE_H5_VAL="/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4AF_spanet_partial_events_population_correction_20260819_v1/production_hdf5_build_v1/work/production_2M_val.h5"
NATIVE_H5_TRAIN_SHA256_EXPECTED="fe9e4d8fa501e24c9585fa18f59caafe8c7631337f8e0428094fa3229e09fe76"
NATIVE_H5_VAL_SHA256_EXPECTED="3c94bf8300d1dc3324e2cf25f9b6c618dffb94819cb3a297a43ac06320d5c2a2"

GATE_DIR="${1:?usage: run_full_join.sh <postflight_gate_result_dir> [work_dir]}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
WORK_DIR="${2:-${PKG_ROOT}/logs/join_${TS}}"
mkdir -p "${WORK_DIR}"

echo "=== Full join ${TS} -> ${WORK_DIR} ==="

echo "--- Gate check: requiring a PASSED postflight from ${GATE_DIR} ---"
if [ ! -f "${GATE_DIR}/GATE_RESULT.json" ]; then
  echo "FATAL: ${GATE_DIR}/GATE_RESULT.json not found. Run run_postflight_gate.sh first and pass its dir here." >&2
  exit 10
fi
GATE_PASS="$(python3 -c "import json; print(json.load(open('${GATE_DIR}/GATE_RESULT.json'))['OVERALL_POSTFLIGHT_PASS'])")"
if [ "${GATE_PASS}" != "True" ]; then
  echo "FATAL: postflight gate at ${GATE_DIR} did NOT pass (OVERALL_POSTFLIGHT_PASS=${GATE_PASS}). Refusing to join." >&2
  exit 11
fi
echo "Postflight gate PASSED. Proceeding."

echo "--- Native H5 checksum re-verification (both splits) ---"
for pair in "train ${NATIVE_H5_TRAIN} ${NATIVE_H5_TRAIN_SHA256_EXPECTED}" "val ${NATIVE_H5_VAL} ${NATIVE_H5_VAL_SHA256_EXPECTED}"; do
  set -- $pair
  split_name=$1; path=$2; expected=$3
  if [ ! -f "${path}" ]; then
    echo "FATAL: native ${split_name} H5 not found at ${path}" >&2
    exit 12
  fi
  actual="$(sha256sum "${path}" | cut -d' ' -f1)"
  echo "native ${split_name} H5 sha256: ${actual}"
  if [ "${actual}" != "${expected}" ]; then
    echo "FATAL: native ${split_name} H5 sha256 mismatch -- expected ${expected}, got ${actual}." >&2
    exit 13
  fi
done

OVERALL_JOIN_PASS="true"
for SPLIT in train val; do
  echo "--- ${SPLIT}: syncing shards locally (read-only xrdcp from EOS) ---"
  LOCAL_SHARD_DIR="${WORK_DIR}/${SPLIT}_shards/${SPLIT}"
  mkdir -p "${LOCAL_SHARD_DIR}"
  # NOTE: at full production scale this is ~550-580 files per split, a few
  # MB each (~10 GiB total both splits, per the project's own storage
  # estimate) -- run once, not routinely. -f is overwrite-if-stale, never
  # deletes anything under the production's own shards/ directory (source
  # is read-only throughout). Destination given explicitly (no trailing
  # slashes) to avoid xrdcp -r's source-basename-nesting ambiguity.
  xrdcp -f -r "${EOS_ROOT}/${SPLIT}" "${LOCAL_SHARD_DIR}" > "${WORK_DIR}/${SPLIT}_sync.log" 2>&1
  SYNC_RC=$?
  if [ "${SYNC_RC}" -ne 0 ]; then
    echo "FATAL: shard sync for split=${SPLIT} failed (see ${WORK_DIR}/${SPLIT}_sync.log)" >&2
    OVERALL_JOIN_PASS="false"
    continue
  fi

  NATIVE_H5_VAR="NATIVE_H5_${SPLIT^^}"
  NATIVE_H5="${!NATIVE_H5_VAR}"

  echo "--- ${SPLIT}: streaming join (native ${NATIVE_H5} + local ParT shards) ---"
  set +e
  python3 "${JOIN_SCRIPT}" \
    --native-h5 "${NATIVE_H5}" \
    --manifest "${MANIFEST}" \
    --split "${SPLIT}" \
    --part-shard-dir "${LOCAL_SHARD_DIR}" \
    --out "${WORK_DIR}/joined_${SPLIT}.h5"
  JOIN_RC=$?
  set -e
  if [ "${JOIN_RC}" -ne 0 ]; then
    echo "FATAL: join for split=${SPLIT} exited nonzero (fail-closed identity/mask/dimension/finiteness check tripped)." >&2
    OVERALL_JOIN_PASS="false"
    continue
  fi

  echo "--- ${SPLIT}: verifying full coverage + independent re-derivation ---"
  set +e
  python3 "${VERIFY_SCRIPT}" \
    --joined-h5 "${WORK_DIR}/joined_${SPLIT}.h5" \
    --coverage-json "${WORK_DIR}/joined_${SPLIT}.h5.coverage.json" \
    --frozen-facts "${FROZEN_FACTS}" \
    --split "${SPLIT}" \
    --out "${WORK_DIR}/JOIN_RECEIPT_${SPLIT}.json"
  VERIFY_RC=$?
  set -e
  if [ "${VERIFY_RC}" -ne 0 ]; then
    echo "FATAL: join verification FAILED for split=${SPLIT}. See ${WORK_DIR}/JOIN_RECEIPT_${SPLIT}.json" >&2
    OVERALL_JOIN_PASS="false"
  fi
done

python3 - "${WORK_DIR}" "${OVERALL_JOIN_PASS}" "${TS}" <<'PYEOF'
import json, sys, os
work_dir, overall_pass, ts = sys.argv[1:4]
combined = {"utc": ts, "OVERALL_JOIN_PASS": overall_pass == "true"}
for split in ("train", "val"):
    p = os.path.join(work_dir, f"JOIN_RECEIPT_{split}.json")
    combined[split] = json.load(open(p)) if os.path.exists(p) else {"error": "receipt missing -- join for this split did not complete"}
with open(os.path.join(work_dir, "JOIN_RECEIPT_COMBINED.json"), "w") as f:
    json.dump(combined, f, indent=2)
print(json.dumps({k: (v if k in ("utc", "OVERALL_JOIN_PASS") else v.get("OVERALL_JOIN_VERIFICATION_PASS", v.get("error"))) for k, v in combined.items()}, indent=2))
PYEOF

echo "OVERALL_JOIN_PASS=${OVERALL_JOIN_PASS}"
[ "${OVERALL_JOIN_PASS}" = "true" ]
