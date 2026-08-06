#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -ne 8 ]]; then
    echo "usage: $0 JOB_INDEX OUTER_FOLD CATEGORY_ID STRUCTURE_ID CONFIG INPUT_ROOT OUTPUT_ROOT EXPECTED_HEAD" >&2
    exit 64
fi

JOB_INDEX="$1"
OUTER_FOLD="$2"
CATEGORY_ID="$3"
STRUCTURE_ID="$4"
CONFIG="$5"
INPUT_ROOT="$6"
OUTPUT_ROOT="$7"
EXPECTED_HEAD="$8"

REPO="/uscms_data/d3/iturkmen/repos/hh-bbww-baselines"
PYTHON="/uscms_data/d3/iturkmen/hh4b_delphes/runtime/hh4b_broad_ml_py39_v1/bin/python"
WORKER="$REPO/scripts/analysis/hh4b_multivariate_cut_structure_worker.py"

cd "$REPO"

[[ -x "$PYTHON" ]] || { echo "missing runtime: $PYTHON" >&2; exit 65; }
[[ -f "$WORKER" ]] || { echo "missing worker: $WORKER" >&2; exit 66; }
[[ "$(git rev-parse HEAD)" == "$EXPECTED_HEAD" ]] || {
    echo "repository head mismatch" >&2
    exit 67
}

export PYTHONPATH="$REPO/scripts/analysis${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON" "$WORKER" \
    --config "$CONFIG" \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --outer-fold "$OUTER_FOLD" \
    --category-id "$CATEGORY_ID" \
    --structure-id "$STRUCTURE_ID" \
    --expected-repository-head "$EXPECTED_HEAD"
