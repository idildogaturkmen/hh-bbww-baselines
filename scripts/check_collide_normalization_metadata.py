'''
Check COLLIDE-1M parquet files for normalization metadata.
This script scans the parquet files in `outputs/collide_selected_backgrounds` for:
- Schema columns with names that look like cross-section, luminosity, generator weight, sum-of-weights, filter efficiency, or k-factor.
- Parquet metadata with keys or values that look like the above.
- Hugging Face cache `.metadata` files for the same keywords.
It also checks the current skim file `outputs/all_background_reco_skim/all_processes_reco_skim.parquet` for the presence of a weight column and whether it is all unit weights.
The results are written to `outputs/collide_normalization_diagnostics`.
'''
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import re
import json

import pandas as pd
import pyarrow.parquet as pq


RAW_ROOT = Path("outputs/collide_selected_backgrounds")
SKIM = Path("outputs/all_background_reco_skim/all_processes_reco_skim.parquet")
OUTDIR = Path("outputs/collide_normalization_diagnostics")

KEYWORD_RE = re.compile(
    r"(xsec|cross|cross[_ -]?section|sigma|lumi|luminosity|"
    r"weight|genweight|gen[_ -]?weight|sumweight|sum[_ -]?weight|"
    r"eventweight|event[_ -]?weight|filter|efficiency|kfactor|k[_ -]?factor)",
    re.I,
)

NEVENT_RE = re.compile(r"NEVENT(\d+)")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(
        p for p in RAW_ROOT.rglob("*.parquet")
        if ".cache" not in p.parts
    )

    schema_rows = []
    metadata_key_counts = defaultdict(int)
    metadata_keyword_rows = []
    filename_rows = []

    for f in parquet_files:
        sample = f.parent.name
        m = NEVENT_RE.search(f.name)
        nevent_from_name = int(m.group(1)) if m else None

        filename_rows.append(
            {
                "sample": sample,
                "file": str(f),
                "nevent_from_filename": nevent_from_name,
            }
        )

        try:
            schema = pq.read_schema(f)
            matching_cols = [c for c in schema.names if KEYWORD_RE.search(c)]

            schema_rows.append(
                {
                    "sample": sample,
                    "file": str(f),
                    "n_columns": len(schema.names),
                    "matching_normalization_like_columns": ";".join(matching_cols),
                }
            )

            meta = pq.ParquetFile(f).metadata.metadata
            if meta:
                for k, v in meta.items():
                    key = k.decode(errors="replace")
                    metadata_key_counts[key] += 1

                    text = ""
                    try:
                        text = v.decode(errors="replace")
                    except Exception:
                        text = repr(v)

                    if KEYWORD_RE.search(key) or KEYWORD_RE.search(text):
                        metadata_keyword_rows.append(
                            {
                                "sample": sample,
                                "file": str(f),
                                "metadata_key": key,
                                "metadata_value_preview": text[:500],
                            }
                        )

        except Exception as e:
            schema_rows.append(
                {
                    "sample": sample,
                    "file": str(f),
                    "n_columns": None,
                    "matching_normalization_like_columns": f"ERROR: {e}",
                }
            )

    schema_df = pd.DataFrame(schema_rows)
    filename_df = pd.DataFrame(filename_rows)

    schema_df.to_csv(OUTDIR / "parquet_schema_normalization_scan.csv", index=False)
    filename_df.to_csv(OUTDIR / "filename_nevent_scan.csv", index=False)

    if metadata_keyword_rows:
        pd.DataFrame(metadata_keyword_rows).to_csv(
            OUTDIR / "parquet_metadata_keyword_hits.csv", index=False
        )
    else:
        pd.DataFrame(
            columns=["sample", "file", "metadata_key", "metadata_value_preview"]
        ).to_csv(OUTDIR / "parquet_metadata_keyword_hits.csv", index=False)

    metadata_keys_df = pd.DataFrame(
        [{"metadata_key": k, "n_files": v} for k, v in sorted(metadata_key_counts.items())]
    )
    metadata_keys_df.to_csv(OUTDIR / "parquet_metadata_keys.csv", index=False)

    # Check Hugging Face cache metadata files.
    hf_meta_files = sorted((RAW_ROOT / ".cache" / "huggingface" / "download").rglob("*.metadata"))
    hf_rows = []

    for f in hf_meta_files:
        try:
            text = f.read_text(errors="replace")
        except Exception as e:
            text = f"ERROR READING FILE: {e}"

        hits = sorted(set(m.group(0) for m in KEYWORD_RE.finditer(text)))
        hf_rows.append(
            {
                "file": str(f),
                "has_normalization_keyword": bool(hits),
                "keyword_hits": ";".join(hits),
                "preview": text[:500].replace("\n", "\\n"),
            }
        )

    pd.DataFrame(hf_rows).to_csv(OUTDIR / "huggingface_metadata_keyword_scan.csv", index=False)

    # Check the current skim's weight column.
    skim_summary = {}
    if SKIM.exists():
        skim = pd.read_parquet(SKIM, columns=["sample", "process_group", "weight"])
        skim_summary["n_skim_events"] = int(len(skim))
        skim_summary["has_weight_column"] = "weight" in skim.columns
        skim_summary["n_unique_weights"] = int(skim["weight"].nunique())
        skim_summary["unique_weights_preview"] = skim["weight"].drop_duplicates().head(20).tolist()

        by_sample = (
            skim.groupby(["process_group", "sample"], dropna=False)
            .agg(
                n_events=("weight", "size"),
                weight_sum=("weight", "sum"),
                weight_min=("weight", "min"),
                weight_max=("weight", "max"),
                n_unique_weights=("weight", "nunique"),
            )
            .reset_index()
        )
        by_sample.to_csv(OUTDIR / "skim_weight_summary_by_sample.csv", index=False)

    with open(OUTDIR / "skim_weight_summary.json", "w") as f:
        json.dump(skim_summary, f, indent=2)

    # Make a readable markdown conclusion.
    any_schema_hits = schema_df["matching_normalization_like_columns"].fillna("").str.len().gt(0).any()
    any_metadata_hits = len(metadata_keyword_rows) > 0
    any_hf_hits = any(row["has_normalization_keyword"] for row in hf_rows)
    skim_all_unit = skim_summary.get("n_unique_weights") == 1 and skim_summary.get("unique_weights_preview") == [1.0]

    with open(OUTDIR / "normalization_diagnostic_summary.md", "w") as f:
        f.write("# COLLIDE-1M normalization metadata diagnostic\n\n")
        f.write(f"- Raw parquet files checked: {len(parquet_files)}\n")
        f.write(f"- Hugging Face `.metadata` files checked: {len(hf_meta_files)}\n")
        f.write(f"- Schema columns with normalization-like names found: {any_schema_hits}\n")
        f.write(f"- Parquet metadata with normalization-like keywords found: {any_metadata_hits}\n")
        f.write(f"- Hugging Face metadata with normalization-like keywords found: {any_hf_hits}\n")
        f.write(f"- Skim has only unit weights: {skim_all_unit}\n\n")

        f.write("## Interpretation\n\n")
        if not any_schema_hits and not any_metadata_hits and skim_all_unit:
            f.write(
                "I did not find cross-section, luminosity, generator-weight, "
                "sum-of-weights, filter-efficiency, or k-factor information in the local "
                "parquet schemas or parquet metadata. The current skim weights appear to be "
                "unit weights. Therefore, the current signal-vs-background plots should be "
                "treated as raw-count/shape diagnostics unless an external normalization table "
                "for COLLIDE-1M is available.\n"
            )
        else:
            f.write(
                "Some normalization-like information may exist. Inspect the CSV outputs in this "
                "directory to decide whether it is genuine physics normalization metadata or just "
                "object names such as GenPart/GenJet.\n"
            )

    print("Wrote diagnostics to:", OUTDIR)
    print("Main summary:", OUTDIR / "normalization_diagnostic_summary.md")


if __name__ == "__main__":
    main()