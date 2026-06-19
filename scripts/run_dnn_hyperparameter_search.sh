#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hh-spanet-surf/repos/hh-bbww-baselines
source /tmp/hh-bbww-venv/bin/activate

SEARCH_ROOT="outputs/DNN_Hyperparameter_Search/search_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SEARCH_ROOT"

echo "Saving hyperparameter search to: $SEARCH_ROOT"
echo "$SEARCH_ROOT" > outputs/DNN_Hyperparameter_Search/latest_search_root.txt

run_one () {
  NAME="$1"
  shift

  echo
  echo "============================================================"
  echo "RUN: $NAME"
  echo "ARGS: $@"
  echo "============================================================"

  mkdir -p "$SEARCH_ROOT/$NAME"

  python -u scripts/train_hbb_bjet_regression.py \
    --epochs 120 \
    --patience 15 \
    --batch-size 1024 \
    "$@" \
    --outdir "$SEARCH_ROOT/$NAME/outputs" \
    --plot-dir "$SEARCH_ROOT/$NAME/plots" \
    --summary-md "$SEARCH_ROOT/$NAME/RESULTS.md"
}

run_one h256_drop005_huber010 \
  --hidden 256 \
  --dropout 0.05 \
  --huber-beta 0.10

run_one h256_drop000_huber010 \
  --hidden 256 \
  --dropout 0.00 \
  --huber-beta 0.10

run_one h128_drop005_huber005 \
  --hidden 128 \
  --dropout 0.05 \
  --huber-beta 0.05

run_one h128_drop005_huber020 \
  --hidden 128 \
  --dropout 0.05 \
  --huber-beta 0.20

run_one h256_drop005_huber005_calibrated \
  --hidden 256 \
  --dropout 0.05 \
  --huber-beta 0.05 \
  --dnn-post-calibrate

run_one h128_drop010_huber010_no_pf \
  --hidden 128 \
  --dropout 0.10 \
  --huber-beta 0.10 \
  --feature-set no_pf

echo
echo "All hyperparameter-search runs finished."
echo "Results saved in: $SEARCH_ROOT"
