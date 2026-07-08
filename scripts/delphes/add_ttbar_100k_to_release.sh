#!/usr/bin/env bash
set -euo pipefail

: "${HH4B_REPO:?Need HH4B_REPO}"
: "${HH4B_STORE:?Need HH4B_STORE}"

cd "$HH4B_REPO"

RELEASE_DIR=data_release/hh4b_delphes_analysis_v0_2026_07_08

mkdir -p "$RELEASE_DIR/parquet" "$RELEASE_DIR/metadata"

cp -v "$HH4B_STORE/parquet/ttbar_100k_merged_event_summary.parquet" "$RELEASE_DIR/parquet/"
cp -v "$HH4B_STORE/parquet/ttbar_100k_merged_hh4b_candidates.parquet" "$RELEASE_DIR/parquet/"
cp -v "$HH4B_STORE/metadata/ttbar_100k_summary.txt" "$RELEASE_DIR/metadata/"
cp -v "$HH4B_STORE/metadata/ttbar_100k_manifest.csv" "$RELEASE_DIR/metadata/"
cp -v "$HH4B_STORE/logs/ttbar_100k_storage_safe_timing.log" "$RELEASE_DIR/metadata/"

find "$RELEASE_DIR" -type f | sort | xargs sha256sum > "$RELEASE_DIR/checksums.sha256"

python3 scripts/delphes/make_dataset_release_inventory.py

git add "$RELEASE_DIR"
git commit -m "Add ttbar 100k outputs to dataset release v0" || true
