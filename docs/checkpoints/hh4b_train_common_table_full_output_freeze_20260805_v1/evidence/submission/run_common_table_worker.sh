#!/usr/bin/env bash
set -Eeuo pipefail

ROW_INDEX="$1"
INPUT_BASENAME="$2"
INPUT_SHA256="$3"
EXPECTED_HEAD="$4"
RUNTIME_ARCHIVE_BASENAME="$5"
RUNTIME_DIRECTORY_BASENAME="$6"

tar -xzf "$RUNTIME_ARCHIVE_BASENAME"
PYTHON="./$RUNTIME_DIRECTORY_BASENAME/bin/python"

"$PYTHON" hh4b_train_common_table_single_source_worker_v1.py \
    --row-index "$ROW_INDEX" \
    --input-parquet "$INPUT_BASENAME" \
    --input-sha256 "$INPUT_SHA256" \
    --pilot-manifest full_464_source_worker_manifest.tsv \
    --coefficient-registry source_physical_coefficient_registry_canary.tsv \
    --comparison-weight-registry fold_local_source_comparison_weight_registry_canary.tsv \
    --feature-schema common_feature_schema_canary.tsv \
    --policy approved_common_train_table_contract.json \
    --expected-head "$EXPECTED_HEAD" \
    --accounting-output accounting.parquet \
    --resolved-output resolved.parquet \
    --receipt-output receipt.json
