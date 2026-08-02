#!/usr/bin/env python3
"""Freeze the train-only PN-c7r three-b-to-four-b multijet transfer.

The nominal transfer factor is measured in the frozen CMS-reference annulus
30 <= R_HH(125,120) < 55 GeV.  Direct stitched hard-QCD is used only as
simulation truth for closure and alternative-factor envelopes.  No validation
or test payload is read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import stat
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


C7Q = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/harvest_checkpoints/"
    "pn_c7q_full441_train_physical_tables_20260802_v1"
)
C7Q_SHA256SUMS_SHA256 = "4356554c5cb927f2c1daa1c412f2797ad398f31df5c27525a084748da7eda273"
THREEB_PATH = C7Q / "tables/train_exactly3b_promoted_run2_physical.parquet"
FOURB_PATH = C7Q / "tables/train_at_least4b_run2_physical.parquet"
EXPECTED_TABLE_SHA256 = {
    THREEB_PATH: "9e7ccf6420ddbe57172b07b61f6c2ba74df2ab28785e18665d5833ae321bddbf",
    FOURB_PATH: "9fa0c54b31288a25d1b911aeb8f69b8cb82377aa341a63d5bccc643f80beb6d5",
}

REGIONS = {
    "baseline_all_candidates": lambda d: np.ones(len(d), dtype=bool),
    "cms_reference_sr_rhh125120_lt30": lambda d: d["r_hh_125_120"].to_numpy() < 30.0,
    "cms_reference_cr_rhh125120_ge30_lt55": lambda d: (
        (d["r_hh_125_120"].to_numpy() >= 30.0)
        & (d["r_hh_125_120"].to_numpy() < 55.0)
    ),
    "cms_reference_outside_rhh125120_ge55": lambda d: d["r_hh_125_120"].to_numpy() >= 55.0,
    "optimized_nominal_sr_rhh125120_lt34": lambda d: d["r_hh_125_120"].to_numpy() < 34.0,
    "higher_purity_sr_rhh125120_lt31p5": lambda d: d["r_hh_125_120"].to_numpy() < 31.5,
    "higher_efficiency_sr_rhh125120_lt35p5": lambda d: d["r_hh_125_120"].to_numpy() < 35.5,
}
MHH = {
    "inclusive_mhh": lambda d: np.ones(len(d), dtype=bool),
    "low_mhh": lambda d: d["mhh"].to_numpy() < 450.0,
    "high_mhh": lambda d: d["mhh"].to_numpy() >= 450.0,
}
FACTOR_SCHEMES = {
    "cms_cr_nominal": "cms_reference_cr_rhh125120_ge30_lt55",
    "baseline_global_alternative": "baseline_all_candidates",
    "outside_ge55_alternative": "cms_reference_outside_rhh125120_ge55",
}
WEIGHT = "run2_candidate_physical_weight"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_sha256sums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        digest, rel = line.split(None, 1)
        rel = rel.lstrip("*")
        if rel.startswith("./"):
            rel = rel[2:]
        require(rel not in result, f"duplicate checksum manifest path: {rel}")
        result[rel] = digest
    return result


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"refusing to write empty TSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def stats(frame: pd.DataFrame) -> dict[str, float | int]:
    weights = frame[WEIGHT].to_numpy(dtype=np.float64)
    total = float(weights.sum(dtype=np.float64))
    sum2 = float(np.square(weights).sum(dtype=np.float64))
    return {
        "rows": int(len(frame)),
        "yield": total,
        "sumw2": sum2,
        "effective_events": float(total * total / sum2) if sum2 > 0 else 0.0,
    }


def selected(frame: pd.DataFrame, region: str, mhh: str) -> pd.DataFrame:
    return frame.loc[np.asarray(REGIONS[region](frame)) & np.asarray(MHH[mhh](frame))]


def transfer_factor(threeb: dict[str, float | int], fourb: dict[str, float | int]) -> dict[str, float]:
    y3 = float(threeb["yield"])
    y4 = float(fourb["yield"])
    require(y3 > 0.0 and y4 > 0.0, "transfer-factor normalization yield is nonpositive")
    value = y4 / y3
    relative_variance = float(fourb["sumw2"]) / (y4 * y4) + float(threeb["sumw2"]) / (y3 * y3)
    variance = value * value * relative_variance
    return {
        "value": value,
        "variance": variance,
        "statistical_uncertainty": math.sqrt(variance),
        "relative_statistical_uncertainty": math.sqrt(relative_variance),
    }


def asimov_stat_only(signal: float, background: float) -> float:
    if signal <= 0.0 or background <= 0.0:
        return 0.0
    value = 2.0 * ((signal + background) * math.log1p(signal / background) - signal)
    return math.sqrt(max(0.0, value))


def verify_inputs() -> dict[str, str]:
    require(C7Q.is_dir(), f"missing PN-c7q checkpoint: {C7Q}")
    require(sha256(C7Q / "SHA256SUMS") == C7Q_SHA256SUMS_SHA256, "PN-c7q manifest hash mismatch")
    manifest = parse_sha256sums(C7Q / "SHA256SUMS")
    for rel, expected in manifest.items():
        path = C7Q / rel
        require(path.is_file(), f"missing PN-c7q member: {rel}")
        require(sha256(path) == expected, f"PN-c7q member checksum mismatch: {rel}")
    summary = json.loads((C7Q / "summary.json").read_text())
    require(summary["status"] == "pn_c7q_full441_train_physical_tables_pass", "PN-c7q is not a pass")
    require(summary["split"] == "train", "PN-c7q contains a non-train split")
    require(summary["validation_payload_files_opened"] == 0, "validation payload opened before PN-c7r")
    require(summary["test_or_evaluation_payload_files_opened"] == 0, "test payload opened before PN-c7r")
    observed = {str(C7Q / "SHA256SUMS"): C7Q_SHA256SUMS_SHA256}
    for path, expected in EXPECTED_TABLE_SHA256.items():
        require(sha256(path) == expected, f"PN-c7q table hash mismatch: {path}")
        observed[str(path)] = expected
    return observed


def build_transfer(staging: Path) -> dict[str, Any]:
    input_hashes = verify_inputs()
    threeb = pq.read_table(THREEB_PATH).to_pandas()
    fourb = pq.read_table(FOURB_PATH).to_pandas()
    require(len(threeb) == 108678 and len(fourb) == 30705, "PN-c7q table row-count drift")
    require(set(threeb["final_split"]) == {"train"} and set(fourb["final_split"]) == {"train"}, "non-train row")
    require(set(threeb["physical_category"]) == {"exactly3b_promoted"}, "three-b category drift")
    require(set(fourb["physical_category"]) == {"at_least4b"}, "four-b category drift")

    threeb_background = threeb.loc[threeb["sample_class"] == "background"].copy()
    threeb_ordinary = threeb_background.loc[threeb_background["population_kind"] == "ordinary"].copy()
    threeb_qcd = threeb_background.loc[threeb_background["population_kind"] == "hard_qcd"].copy()
    fourb_qcd = fourb.loc[fourb["population_kind"] == "hard_qcd"].copy()
    fourb_ordinary = fourb.loc[
        (fourb["sample_class"] == "background") & (fourb["population_kind"] == "ordinary")
    ].copy()
    fourb_signal = fourb.loc[fourb["sample_class"] == "signal"].copy()
    require(len(threeb_qcd) == 576 and len(fourb_qcd) == 56, "direct-QCD candidate count drift")
    require(len(threeb_ordinary) + len(threeb_qcd) == len(threeb_background), "pseudodata subtraction partition failed")
    require(
        math.isclose(
            float(threeb_background[WEIGHT].sum()) - float(threeb_ordinary[WEIGHT].sum()),
            float(threeb_qcd[WEIGHT].sum()),
            rel_tol=2e-15,
            abs_tol=1e-8,
        ),
        "simulated-pseudodata ordinary-background subtraction closure failed",
    )

    factor_rows: list[dict[str, Any]] = []
    factors: dict[tuple[str, str], dict[str, float]] = {}
    for mhh in MHH:
        for scheme, region in FACTOR_SCHEMES.items():
            three_stats = stats(selected(threeb_qcd, region, mhh))
            four_stats = stats(selected(fourb_qcd, region, mhh))
            factor = transfer_factor(three_stats, four_stats)
            factors[(mhh, scheme)] = factor
            factor_rows.append({
                "mhh_category": mhh,
                "factor_scheme": scheme,
                "normalization_region": region,
                "threeb_rows": three_stats["rows"],
                "fourb_rows": four_stats["rows"],
                "threeb_yield": three_stats["yield"],
                "fourb_yield": four_stats["yield"],
                "threeb_sumw2": three_stats["sumw2"],
                "fourb_sumw2": four_stats["sumw2"],
                "threeb_effective_events": three_stats["effective_events"],
                "fourb_effective_events": four_stats["effective_events"],
                "transfer_factor": factor["value"],
                "transfer_factor_statistical_uncertainty": factor["statistical_uncertainty"],
                "transfer_factor_relative_statistical_uncertainty": factor["relative_statistical_uncertainty"],
                "role": "nominal" if scheme == "cms_cr_nominal" else "normalization_region_alternative",
            })

    closure_rows: list[dict[str, Any]] = []
    yield_rows: list[dict[str, Any]] = []
    all_envelopes_cover_truth = True
    maximum_nonclosure = 0.0
    maximum_envelope_up = 0.0
    maximum_envelope_down = 0.0
    for mhh in MHH:
        nominal = factors[(mhh, "cms_cr_nominal")]
        alternatives = [factors[(mhh, scheme)]["value"] for scheme in FACTOR_SCHEMES]
        for region in REGIONS:
            three_stats = stats(selected(threeb_qcd, region, mhh))
            truth_stats = stats(selected(fourb_qcd, region, mhh))
            y3 = float(three_stats["yield"])
            truth = float(truth_stats["yield"])
            prediction = nominal["value"] * y3
            prediction_variance = (
                nominal["value"] * nominal["value"] * float(three_stats["sumw2"])
                + y3 * y3 * nominal["variance"]
            )
            alt_predictions = [value * y3 for value in alternatives]
            envelope_low = min(alt_predictions)
            envelope_high = max(alt_predictions)
            covered = envelope_low <= truth <= envelope_high
            all_envelopes_cover_truth &= covered
            residual = prediction - truth
            relative_nonclosure = abs(residual) / abs(truth) if truth else 0.0
            envelope_up = (envelope_high - prediction) / abs(prediction) if prediction else 0.0
            envelope_down = (prediction - envelope_low) / abs(prediction) if prediction else 0.0
            maximum_nonclosure = max(maximum_nonclosure, relative_nonclosure)
            maximum_envelope_up = max(maximum_envelope_up, envelope_up)
            maximum_envelope_down = max(maximum_envelope_down, envelope_down)
            closure_rows.append({
                "mhh_category": mhh,
                "target_region": region,
                "nominal_factor_scheme": "cms_cr_nominal",
                "threeb_qcd_rows": three_stats["rows"],
                "fourb_direct_qcd_rows": truth_stats["rows"],
                "threeb_qcd_yield": y3,
                "transferred_qcd_prediction": prediction,
                "transferred_qcd_prediction_sumw2_statistical": prediction_variance,
                "direct_fourb_qcd_truth": truth,
                "direct_fourb_qcd_truth_sumw2": truth_stats["sumw2"],
                "prediction_over_truth": prediction / truth if truth else 0.0,
                "signed_closure_residual": residual,
                "relative_absolute_nonclosure": relative_nonclosure,
                "alternative_envelope_low": envelope_low,
                "alternative_envelope_high": envelope_high,
                "direct_truth_covered_by_factor_envelope": covered,
                "validation_used": False,
                "test_used": False,
            })

            ordinary = stats(selected(fourb_ordinary, region, mhh))
            signal = stats(selected(fourb_signal, region, mhh))
            background = prediction + float(ordinary["yield"])
            yield_rows.append({
                "mhh_category": mhh,
                "region": region,
                "signal_rows": signal["rows"],
                "ordinary_fourb_background_rows": ordinary["rows"],
                "threeb_qcd_template_rows": three_stats["rows"],
                "signal_yield": signal["yield"],
                "ordinary_fourb_background_yield": ordinary["yield"],
                "transferred_multijet_yield": prediction,
                "primary_background_yield": background,
                "signal_over_background": float(signal["yield"]) / background if background > 0 else 0.0,
                "signal_over_sqrt_background": float(signal["yield"]) / math.sqrt(background) if background > 0 else 0.0,
                "statistical_only_asimov_ZA": asimov_stat_only(float(signal["yield"]), background),
                "systematic_aware_significance_calculated": False,
                "direct_qcd_included_in_primary_background": False,
                "yield_scope": "train_partition_contribution_only",
            })

    require(all_envelopes_cover_truth, "alternative transfer-factor envelope does not cover all direct-QCD closures")
    write_tsv(staging / "transfer_factor_registry.tsv", factor_rows)
    write_tsv(staging / "transfer_closure.tsv", closure_rows)
    write_tsv(staging / "primary_train_yield_projection.tsv", yield_rows)

    # The sidecar carries the primary event-level multijet template.  Only the
    # label-known hard-QCD residual survives exact simulated-pseudodata
    # subtraction; direct four-b QCD is intentionally absent.
    nominal_inclusive = factors[("inclusive_mhh", "cms_cr_nominal")]
    transfer_sidecar = threeb_qcd[[
        "production_row_index", "transport_id", "candidate_row_index", "event",
        "process_or_mode", "campaign", "source_index_or_target_index",
        "generator_nominal_weight", WEIGHT,
    ]].copy()
    transfer_sidecar = transfer_sidecar.rename(columns={WEIGHT: "threeb_run2_candidate_physical_weight"})
    transfer_sidecar["transfer_factor"] = nominal_inclusive["value"]
    transfer_sidecar["primary_multijet_transfer_weight"] = (
        transfer_sidecar["threeb_run2_candidate_physical_weight"] * nominal_inclusive["value"]
    )
    transfer_sidecar["template_role"] = "lower_btag_simulation_pseudodata_after_exact_ordinary_subtraction"
    transfer_path = staging / "threeb_primary_multijet_transfer_sidecar.parquet"
    pq.write_table(pa.Table.from_pandas(transfer_sidecar, preserve_index=False), transfer_path, compression="zstd")

    sr30 = next(
        row for row in closure_rows
        if row["mhh_category"] == "inclusive_mhh"
        and row["target_region"] == "cms_reference_sr_rhh125120_lt30"
    )
    systematic_rows = [
        {
            "nuisance": "nominal_transfer_factor_statistical",
            "scope": "multijet_normalization",
            "relative_down": nominal_inclusive["relative_statistical_uncertainty"],
            "relative_up": nominal_inclusive["relative_statistical_uncertainty"],
            "source": "weighted independent ratio propagation in CMS control annulus",
            "frozen_for_baseline_reporting": True,
        },
        {
            "nuisance": "normalization_region_factor_envelope",
            "scope": "multijet_normalization_and_shape",
            "relative_down": maximum_envelope_down,
            "relative_up": maximum_envelope_up,
            "source": "CMS-annulus nominal versus inclusive/global and R_HH>=55 alternatives by mHH category",
            "frozen_for_baseline_reporting": True,
        },
        {
            "nuisance": "cms_sr_direct_qcd_nonclosure",
            "scope": "cms_reference_signal_region_multijet",
            "relative_down": sr30["relative_absolute_nonclosure"],
            "relative_up": sr30["relative_absolute_nonclosure"],
            "source": "nominal lower-btag prediction versus secondary direct stitched QCD",
            "frozen_for_baseline_reporting": True,
        },
    ]
    write_tsv(staging / "systematic_registry.tsv", systematic_rows)
    source_rows = [
        {"path": path, "sha256": digest, "role": "pinned_PN_c7q_input"}
        for path, digest in sorted(input_hashes.items())
    ]
    write_tsv(staging / "source_evidence_manifest.tsv", source_rows)

    transfer_manifest = {
        "path": transfer_path.name,
        "bytes": transfer_path.stat().st_size,
        "sha256": sha256(transfer_path),
        "rows": pq.read_metadata(transfer_path).num_rows,
        "columns": pq.read_metadata(transfer_path).num_columns,
    }
    summary = {
        "schema_version": 1,
        "status": "pn_c7r_train_multijet_transfer_closure_pass_with_frozen_nonclosure_envelope",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_checkpoint": str(C7Q),
        "split": "train",
        "threeb_simulated_pseudodata_background_rows": len(threeb_background),
        "threeb_ordinary_background_subtraction_rows": len(threeb_ordinary),
        "threeb_primary_multijet_template_rows": len(threeb_qcd),
        "fourb_direct_qcd_closure_rows": len(fourb_qcd),
        "fourb_ordinary_background_rows": len(fourb_ordinary),
        "fourb_signal_rows": len(fourb_signal),
        "nominal_transfer_factor": nominal_inclusive["value"],
        "nominal_transfer_factor_relative_statistical_uncertainty": nominal_inclusive["relative_statistical_uncertainty"],
        "cms_sr_relative_absolute_nonclosure": sr30["relative_absolute_nonclosure"],
        "maximum_relative_absolute_nonclosure_across_frozen_closures": maximum_nonclosure,
        "maximum_normalization_region_envelope_relative_down": maximum_envelope_down,
        "maximum_normalization_region_envelope_relative_up": maximum_envelope_up,
        "all_direct_qcd_closures_covered_by_alternative_factor_envelopes": all_envelopes_cover_truth,
        "direct_stitched_qcd_role": "secondary_closure_only_not_primary_background",
        "primary_multijet_role": "lower_btag_simulation_pseudodata_transfer",
        "systematic_aware_significance_calculated": False,
        "full_run2_prediction": False,
        "validation_payload_files_opened": 0,
        "test_or_evaluation_payload_files_opened": 0,
        "transfer_sidecar": transfer_manifest,
        "next_gate": "freeze immutable normalized train analysis tables and baseline inputs",
        "baseline_development_authorized_after_final_train_table_freeze": True,
    }
    write_json(staging / "summary.json", summary)
    (staging / "RUN_CONTRACT.txt").write_text(
        "The nominal lower-b-tag multijet transfer is the weighted hard-QCD four-b/three-b "
        "ratio in 30 <= R_HH(125,120) < 55 GeV, evaluated independently for the inclusive, "
        "low-mHH, and high-mHH frozen categories. Simulated pseudodata subtracts the exact "
        "ordinary-background component. Direct four-b stitched QCD is secondary closure only. "
        "The large observed nonclosure is preserved and covered by frozen alternative-region "
        "factor envelopes; it is not tuned away. Validation and test remain sealed.\n"
    )
    (staging / "README.md").write_text(
        "# PN-c7r three-b to four-b multijet transfer\n\n"
        "This checkpoint freezes the train-only lower-b-tag simulated-pseudodata transfer, "
        "direct-QCD closure tests, and conservative statistical/nonclosure systematics. The "
        "closure passes because every fixed-region direct-QCD truth yield is contained by the "
        "predefined normalization-region envelope; the numerically large uncertainty is an "
        "explicit result of the small effective QCD sample, not an optimization target.\n"
    )
    return summary


def freeze(staging: Path, output: Path) -> str:
    (staging / "COMPLETE").write_text("pn_c7r_train_multijet_transfer_closure_pass\n")
    files = sorted(path for path in staging.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (staging / "SHA256SUMS").write_text(
        "\n".join(f"{sha256(path)}  ./{path.relative_to(staging).as_posix()}" for path in files) + "\n"
    )
    digest = sha256(staging / "SHA256SUMS")
    os.rename(staging, output)
    for path in sorted(output.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    output.chmod(0o555)
    return digest


def run(output: Path) -> dict[str, Any]:
    require(output.is_absolute(), "output path must be absolute")
    require(output.parent.is_dir(), "output parent does not exist")
    require(not output.exists(), f"refusing to overwrite output: {output}")
    staging = output.with_name(f".{output.name}.staging")
    require(not staging.exists(), f"refusing to overwrite preserved staging: {staging}")
    staging.mkdir()
    try:
        summary = build_transfer(staging)
        digest = freeze(staging, output)
        return {"output": str(output), "sha256sums_sha256": digest, **summary}
    except Exception as exc:
        try:
            write_json(staging / "FAILURE.json", {
                "status": "pn_c7r_failed_preserved_staging",
                "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
        except Exception:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
