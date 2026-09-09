#!/bin/bash
# EAF GPU wrapper for launch_spa2m_part.py. Run ONLY on an EAF GPU session,
# NOT from LPC (no GPU there). Requires AUTHORIZED_TO_RUN=True to have been
# set by a human inside launch_spa2m_part.py first -- this wrapper does not
# and cannot flip that itself.
#
# This is a direct generalization of the already-reviewed
# run_2M_seed0_training.sh (same native-launcher wrapper this package's
# launch_spa2m_part.py was itself adapted from): identical pixi-environment
# resolution, identical nvidia-smi -L GPU/MIG discovery-and-validation
# safety checks (never a stale/hardcoded UUID, never a silently-trusted
# placeholder, refuses to guess if zero or more than one device is
# visible). The only new behavior is passing through the extra
# --variant/--hdf5-train/--hdf5-val/--event-yaml/--build-receipt/--run-name
# arguments launch_spa2m_part.py needs that the native launcher did not.
#
# Usage:
#   run_spa2m_part_training.sh <variant: active|all128> <hdf5_train> <hdf5_val> \
#       <event_yaml> <build_receipt_json> <run_name>
set -uo pipefail

if [ "$#" -ne 6 ]; then
  echo "usage: $0 <variant: active|all128> <hdf5_train> <hdf5_val> <event_yaml> <build_receipt_json> <run_name>" >&2
  exit 2
fi
VARIANT="$1"; HDF5_TRAIN="$2"; HDF5_VAL="$3"; EVENT_YAML="$4"; BUILD_RECEIPT="$5"; RUN_NAME="$6"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Same pinned pixi-bootstrapped SPA-Net clone every gate in this project has
# used -- resolved the same way launch_spa2m_part.py resolves its own
# SPANET_REPO constant, so the two files cannot silently diverge.
TRACK_B_ROOT="/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812"
R2_DIR="$TRACK_B_ROOT/phase4AE_spanet_10jet_literature_aligned_training_preflight_20260818_v1/gpu_canary_10jet_trainonlyweighted_v23exact_r2"
if [ ! -d "$R2_DIR" ]; then
  echo "FATAL: cannot resolve R2_DIR=$R2_DIR" >&2
  exit 1
fi
PY="$R2_DIR/.pixi/envs/default/bin/python"
WORK_DIR="$DIR/work"
mkdir -p "$WORK_DIR"

echo "resolved DIR=$DIR"
echo "resolved R2_DIR=$R2_DIR"
echo "resolved PY=$PY"
echo "VARIANT=$VARIANT RUN_NAME=$RUN_NAME"

if [ ! -x "$PY" ]; then
  echo "FATAL: $PY not found or not executable." >&2
  exit 3
fi
echo "test -x \"\$PY\": PASS"

for f in "$HDF5_TRAIN" "$HDF5_VAL" "$EVENT_YAML" "$BUILD_RECEIPT"; do
  if [ ! -f "$f" ]; then
    echo "FATAL: required input not found: $f" >&2
    exit 3
  fi
done

NVSMI_L_LOG="$WORK_DIR/nvidia_smi_L.launch_time.${RUN_NAME}.log"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FATAL: nvidia-smi not found on PATH. This must be run from an actual EAF GPU session, not LPC." >&2
  exit 4
fi
nvidia-smi -L > "$NVSMI_L_LOG" 2>&1
echo "---- nvidia-smi -L (this session, this launch) ----"
cat "$NVSMI_L_LOG"
echo "-----------------------------------------------------"

# Unchanged from run_2M_seed0_training.sh: extract every UUID token this
# session's own nvidia-smi -L just reported, then either validate an
# already-set CUDA_VISIBLE_DEVICES against that list or auto-bind iff
# exactly one device is visible. Never guesses among multiple, never
# trusts an unsubstituted placeholder or a UUID from a different session.
mapfile -t DISCOVERED_UUIDS < <(grep -oE '(GPU|MIG)-[0-9a-fA-F-]+' "$NVSMI_L_LOG")
UUID_SHAPE_RE='^(GPU|MIG)-[0-9a-fA-F]{8}-'

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  if ! [[ "$CUDA_VISIBLE_DEVICES" =~ $UUID_SHAPE_RE ]]; then
    echo "FATAL: CUDA_VISIBLE_DEVICES is set but does not look like a real GPU/MIG UUID: '$CUDA_VISIBLE_DEVICES'" >&2
    exit 6
  fi
  MATCH=0
  for u in "${DISCOVERED_UUIDS[@]:-}"; do
    [ "$u" = "$CUDA_VISIBLE_DEVICES" ] && MATCH=1 && break
  done
  if [ "$MATCH" -ne 1 ]; then
    echo "FATAL: CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' does not match any GPU/MIG UUID this session's own nvidia-smi -L just reported:" >&2
    printf '  %s\n' "${DISCOVERED_UUIDS[@]:-}" >&2
    exit 6
  fi
  echo "CUDA_VISIBLE_DEVICES already set and VERIFIED against this session's own nvidia-smi -L: $CUDA_VISIBLE_DEVICES"
else
  if [ "${#DISCOVERED_UUIDS[@]}" -eq 0 ]; then
    echo "FATAL: CUDA_VISIBLE_DEVICES is unset and nvidia-smi -L reported no GPU/MIG UUIDs in this session." >&2
    exit 5
  elif [ "${#DISCOVERED_UUIDS[@]}" -gt 1 ]; then
    echo "FATAL: CUDA_VISIBLE_DEVICES is unset and nvidia-smi -L reports ${#DISCOVERED_UUIDS[@]} visible GPU/MIG devices -- refusing to guess:" >&2
    printf '  %s\n' "${DISCOVERED_UUIDS[@]}" >&2
    exit 5
  fi
  export CUDA_VISIBLE_DEVICES="${DISCOVERED_UUIDS[0]}"
  echo "Discovered exactly one visible GPU/MIG device this session; binding CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

STDOUT_LOG="$WORK_DIR/training_${RUN_NAME}.stdout.log"
STDERR_LOG="$WORK_DIR/training_${RUN_NAME}.stderr.log"

START_UTC="$(date -u +%FT%TZ)"
"$PY" "$DIR/launch_spa2m_part.py" \
  --variant "$VARIANT" \
  --hdf5-train "$HDF5_TRAIN" \
  --hdf5-val "$HDF5_VAL" \
  --event-yaml "$EVENT_YAML" \
  --build-receipt "$BUILD_RECEIPT" \
  --run-name "$RUN_NAME" \
  --run-dir "$DIR" \
  > "$STDOUT_LOG" 2> "$STDERR_LOG"
STATUS=$?
END_UTC="$(date -u +%FT%TZ)"

echo "EXIT_CODE=$STATUS" > "$WORK_DIR/training_${RUN_NAME}.exit_status"
echo "START_UTC=$START_UTC" >> "$WORK_DIR/training_${RUN_NAME}.exit_status"
echo "END_UTC=$END_UTC" >> "$WORK_DIR/training_${RUN_NAME}.exit_status"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >> "$WORK_DIR/training_${RUN_NAME}.exit_status"

echo "---- warnings (Warning|WARN grep) in stderr log ----"
grep -nE "Warning|WARN" "$STDERR_LOG" | head -100
echo "---- stdout (tail) ----"
tail -c 4000 "$STDOUT_LOG"
echo "---- stderr (tail) ----"
tail -c 4000 "$STDERR_LOG"
echo "---- result JSON (if present) ----"
[ -f "$WORK_DIR/training_${RUN_NAME}_result.json" ] && cat "$WORK_DIR/training_${RUN_NAME}_result.json"
exit "$STATUS"
