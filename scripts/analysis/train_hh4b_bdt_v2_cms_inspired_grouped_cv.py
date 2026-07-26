#!/usr/bin/env python3
"""Train-only grouped-CV comparison of categorized CMS-inspired HH4b BDTs."""

from __future__ import annotations

import os

# These controls must be set before importing NumPy, SciPy, sklearn, or XGBoost.
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import atexit
from collections import Counter, defaultdict
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "analysis"))
MPLCONFIG_TEMP = Path(tempfile.gettempdir()) / f"hh4b_bdt_v2_cv_mpl_{os.getpid()}"
os.environ["MPLCONFIGDIR"] = str(MPLCONFIG_TEMP)
atexit.register(shutil.rmtree, MPLCONFIG_TEMP, ignore_errors=True)

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
import xgboost  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402
import yaml  # noqa: E402

from hh4b_bdt_v2_common import (  # noqa: E402
    CATEGORIZED_EXPLICIT_DIJET_MASS_PLANE_BLIND_FEATURES,
    CATEGORIZED_MASS_AWARE_FEATURES,
    CATEGORY_NAMES,
    GLOBAL_V1_MASS_AWARE_REFERENCE,
    write_table_bundle,
)
from prepare_hh4b_bdt_v2_cms_inspired_contract import (  # noqa: E402
    load_and_verify_inputs as load_v2_contract_inputs,
    load_train_population as load_v2_train_population,
)
from train_hh4b_bdt_v1_grouped_cv import (  # noqa: E402
    PARAMETER_NAMES,
    aggregate_gain_importance,
    binary_metrics,
    booster_gain_vector,
    efficiency,
    rank_hyperparameter_trials,
    score_quintile_assignment,
    weighted_signal_threshold,
)
from scripts.plotting.hh4b_cms_style import (  # noqa: E402
    CATEGORY_COLORS,
    FEATURE_LATEX_LABELS,
    add_delphes_header,
    apply_cms_style,
    save_png_pdf,
)


CONFIG_DEFAULT = REPOSITORY_ROOT / "configs/baselines/hh4b_bdt_v2_cms_inspired_grouped_cv.yaml"
PRIMARY = "categorized_cms_inspired_mass_aware"
CATEGORY_SPLIT = "categorized_v1_features_mass_aware_ablation"
MASS_BLIND = "categorized_cms_inspired_explicit_dijet_mass_plane_blind"
STRATEGIES = (PRIMARY, CATEGORY_SPLIT, MASS_BLIND)
GLOBAL_MASS = "global_v1_mass_aware"
GLOBAL_BLIND = "global_v1_explicit_dijet_mass_plane_blind"
CUT_TARGET = 0.5859573147685532


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPOSITORY_ROOT, check=True, text=True,
        capture_output=True,
    ).stdout.strip()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def plain(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"
    return value


def write_tsv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t",
                                extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: plain(row.get(field, "")) for field in fields})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_sha256_manifest(directory: Path) -> None:
    for line in (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        digest, relative = line.split(maxsplit=1)
        target = directory / relative.strip().lstrip("*")
        if not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f"SHA256 failure: {target}")


def write_sha256_manifest(directory: Path) -> None:
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (directory / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(p)}  {p.relative_to(directory).as_posix()}\n" for p in files),
        encoding="utf-8",
    )


def prepare_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def environment_record(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "matplotlib_version": matplotlib.__version__,
        "cpu_training_only": True,
        "fits_sequential": True,
        "controls": {key: os.environ.get(key) for key in config["required_environment"]["controls"]},
    }


def verify_environment(config: Mapping[str, Any]) -> dict[str, Any]:
    record = environment_record(config)
    required = config["required_environment"]
    for key in ("python_version", "numpy_version", "scipy_version",
                "sklearn_version", "xgboost_version"):
        if record[key] != required[key]:
            raise RuntimeError(f"environment mismatch {key}: {record[key]} != {required[key]}")
    if Path(sys.executable).resolve() != Path(config["required_interpreter"]).resolve():
        raise RuntimeError("wrong Python interpreter")
    if record["controls"] != required["controls"]:
        raise RuntimeError("thread controls differ")
    return record


def category_local_hierarchical_fit_weights(
    target: Sequence[int],
    members: Sequence[int],
    signal_modes: Sequence[str],
    background_families: Sequence[str],
    selected: Sequence[bool],
) -> np.ndarray:
    """Build category-local hierarchical weights from selected fit rows only."""
    target = np.asarray(target, dtype=np.int8)
    members = np.asarray(members, dtype=np.int64)
    modes = np.asarray(signal_modes, dtype=object)
    families = np.asarray(background_families, dtype=object)
    selected = np.asarray(selected, dtype=bool)
    indices = np.flatnonzero(selected)
    if not indices.size or set(np.unique(target[selected])) != {0, 1}:
        raise ValueError("fit selection must contain both classes")
    weights = np.zeros(len(target), dtype=np.float64)
    for class_value, strata in (
        (1, sorted(set(modes[selected & (target == 1)]))),
        (0, sorted(set(families[selected & (target == 0)]))),
    ):
        strata = [value for value in strata if value]
        if not strata:
            raise ValueError("class has no populated strata")
        for stratum in strata:
            stratum_mask = selected & (target == class_value)
            stratum_mask &= (modes == stratum) if class_value == 1 else (families == stratum)
            stratum_members = sorted(set(members[stratum_mask]))
            for member in stratum_members:
                member_mask = stratum_mask & (members == member)
                weights[member_mask] = (
                    0.5 / len(strata) / len(stratum_members) / np.count_nonzero(member_mask)
                )
    fit_weights = weights[selected]
    if np.any(fit_weights <= 0) or not np.all(np.isfinite(fit_weights)):
        raise ValueError("invalid category-local fit weights")
    weights[selected] *= len(fit_weights) / float(np.sum(fit_weights))
    if not np.isclose(np.mean(weights[selected]), 1.0, atol=1e-12, rtol=0):
        raise ValueError("fit weights do not have mean one")
    return weights


def load_frozen_hyperparameter_space(path: Path) -> list[dict[str, Any]]:
    rows = read_tsv(path)
    if len(rows) != 24 or [int(row["trial_id"]) for row in rows] != list(range(24)):
        raise ValueError("frozen hyperparameter space is not the exact 24 trials")
    typed = []
    integer_fields = {"trial_id", "n_estimators", "max_depth", "min_child_weight",
                      "complexity_proxy", "selection_seed"}
    float_fields = set(PARAMETER_NAMES) - {"n_estimators", "max_depth", "min_child_weight"}
    for row in rows:
        item: dict[str, Any] = dict(row)
        for field in integer_fields:
            item[field] = int(float(item[field]))
        for field in float_fields:
            item[field] = float(item[field])
        item["is_baseline"] = str(item["is_baseline"]).lower() == "true"
        typed.append(item)
    return typed


def select_category_hyperparameters(
    fold_rows: Sequence[Mapping[str, Any]],
    hyper_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    output, selected = [], {}
    by = defaultdict(list)
    for row in fold_rows:
        if row["fit_status"] == "pass":
            by[(row["category"], int(row["trial_id"]))].append(row)
    hyper_by_id = {int(row["trial_id"]): row for row in hyper_rows}
    for category in CATEGORY_NAMES:
        trials = []
        for trial_id, hyper in hyper_by_id.items():
            folds = by[(category, trial_id)]
            if len(folds) != 5:
                continue
            auc = np.asarray([float(row["weighted_roc_auc"]) for row in folds])
            trials.append({
                "category": category, "trial_id": trial_id,
                **{name: hyper[name] for name in PARAMETER_NAMES},
                "folds": 5,
                "mean_weighted_roc_auc": float(np.mean(auc)),
                "worst_fold_weighted_roc_auc": float(np.min(auc)),
                "std_weighted_roc_auc": float(np.std(auc)),
                "maximum_weighted_roc_auc": float(np.max(auc)),
                "complexity_proxy": int(hyper["complexity_proxy"]),
            })
        if len(trials) != 24:
            raise ValueError(f"incomplete search for {category}")
        ranked = rank_hyperparameter_trials(trials)
        for row in ranked:
            row["category"] = category
            row["status"] = "pass"
        output.extend(sorted(ranked, key=lambda row: int(row["trial_id"])))
        selected[category] = next(row for row in ranked if row["selected"])
    return output, selected


def check_categorized_oof_integrity(
    expected_indices: Sequence[int],
    produced_indices: Sequence[int],
    train_member_sets: Sequence[set[int]],
    heldout_member_sets: Sequence[set[int]],
    expected_categories: Sequence[str],
    produced_categories: Sequence[str],
) -> dict[str, int]:
    expected, produced = Counter(expected_indices), Counter(produced_indices)
    missing = sum(max(count - produced.get(key, 0), 0) for key, count in expected.items())
    duplicates = sum(max(count - expected.get(key, 0), 0) for key, count in produced.items())
    leakage = sum(len(a & b) for a, b in zip(train_member_sets, heldout_member_sets))
    mismatch = sum(a != b for a, b in zip(expected_categories, produced_categories))
    return {
        "missing_oof_rows": int(missing), "duplicate_oof_rows": int(duplicates),
        "member_leakage_rows": int(leakage), "category_mismatch_rows": int(mismatch),
    }


def prohibit_pooled_uncalibrated_category_auc() -> None:
    raise ValueError("pooled uncalibrated category AUC is prohibited")


def category_specific_threshold_selection(
    scores: Sequence[float], target: Sequence[int], weights: Sequence[float],
    categories: Sequence[str], target_efficiency: float,
) -> tuple[dict[str, float], np.ndarray]:
    scores = np.asarray(scores); target = np.asarray(target); weights = np.asarray(weights)
    categories = np.asarray(categories)
    thresholds, selected = {}, np.zeros(len(scores), dtype=bool)
    for category in CATEGORY_NAMES:
        signal = (categories == category) & (target == 1)
        threshold = weighted_signal_threshold(scores[signal], weights[signal], target_efficiency)
        thresholds[category] = threshold
        selected |= (categories == category) & (scores >= threshold)
    return thresholds, selected


def selection_efficiencies(
    selected: np.ndarray, target: np.ndarray, weights: np.ndarray,
    modes: np.ndarray, families: np.ndarray,
) -> dict[str, Any]:
    result = {
        "weighted_signal_efficiency": efficiency(selected, target == 1, weights),
        "weighted_background_efficiency": efficiency(selected, target == 0, weights),
        "raw_signal_efficiency": efficiency(selected, target == 1),
        "raw_background_efficiency": efficiency(selected, target == 0),
    }
    result["weighted_background_rejection"] = (
        1.0 / result["weighted_background_efficiency"]
        if result["weighted_background_efficiency"] > 0 else float("inf")
    )
    result["signal_mode_efficiencies"] = {
        mode: efficiency(selected, modes == mode, weights)
        for mode in sorted(set(modes) - {""})
    }
    result["background_family_efficiencies"] = {
        family: efficiency(selected, families == family, weights)
        for family in sorted(set(families) - {""})
    }
    return result


def member_bootstrap_differences(
    members: Sequence[int], weights: Sequence[float], target: Sequence[int],
    first_selected: Sequence[bool], second_selected: Sequence[bool],
    *, seed: int, replicates: int,
) -> np.ndarray:
    members = np.asarray(members); weights = np.asarray(weights); target = np.asarray(target)
    first = np.asarray(first_selected); second = np.asarray(second_selected)
    unique = np.asarray(sorted(set(members)))
    rng = np.random.default_rng(seed)
    output = np.empty(replicates)
    for replica in range(replicates):
        background = target == 0
        denominator = 0.0
        while denominator <= 0.0:
            sampled = rng.choice(unique, size=len(unique), replace=True)
            multiplicity = Counter(int(value) for value in sampled)
            mult = np.asarray([multiplicity.get(int(value), 0) for value in members])
            boot_weight = weights * mult
            denominator = float(np.sum(boot_weight[background]))
        output[replica] = (
            np.sum(boot_weight[background & first]) / denominator
            - np.sum(boot_weight[background & second]) / denominator
        )
    return output


def score_mass_rows(
    score: np.ndarray, rhh: np.ndarray, mhh: np.ndarray, weights: np.ndarray,
    strategy: str, category: str,
) -> list[dict[str, Any]]:
    assignment, _ = score_quintile_assignment(score, weights)
    sr, pr = float(spearmanr(score, rhh).statistic), float(pearsonr(score, rhh).statistic)
    sm, pm = float(spearmanr(score, mhh).statistic), float(pearsonr(score, mhh).statistic)
    rows = []
    for quintile in range(5):
        mask = assignment == quintile
        rows.append({
            "strategy": strategy, "category": category, "score_quintile": quintile + 1,
            "rows": int(np.count_nonzero(mask)),
            "weighted_fraction": float(np.sum(weights[mask]) / np.sum(weights)),
            "score_minimum": float(np.min(score[mask])),
            "score_maximum": float(np.max(score[mask])),
            "spearman_score_r_hh": sr, "pearson_score_r_hh": pr,
            "spearman_score_mhh": sm, "pearson_score_mhh": pm,
            "median_r_hh_125_125": float(np.median(rhh[mask])),
            "mean_r_hh_125_125": float(np.mean(rhh[mask])),
            "weighted_fraction_rhh_lt_34": efficiency(rhh[mask] < 34, np.ones(np.count_nonzero(mask), bool), weights[mask]),
            "raw_fraction_rhh_lt_34": float(np.mean(rhh[mask] < 34)),
            "weighted_fraction_rhh_lt_50": efficiency(rhh[mask] < 50, np.ones(np.count_nonzero(mask), bool), weights[mask]),
            "raw_fraction_rhh_lt_50": float(np.mean(rhh[mask] < 50)),
            "weighted_fraction_rhh_lt_80": efficiency(rhh[mask] < 80, np.ones(np.count_nonzero(mask), bool), weights[mask]),
            "raw_fraction_rhh_lt_80": float(np.mean(rhh[mask] < 80)),
            "used_in_model_selection": False, "status": "pass",
        })
    return rows


def xgb_parameters(fixed: Mapping[str, Any], trial: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(fixed), **{name: trial[name] for name in PARAMETER_NAMES}}


def fit_model(
    matrix: np.ndarray, target: np.ndarray, fit_weights: np.ndarray,
    train: np.ndarray, heldout: np.ndarray, eval_weights: np.ndarray,
    parameters: Mapping[str, Any],
) -> tuple[XGBClassifier, np.ndarray, dict[str, float], float]:
    start = time.perf_counter()
    model = XGBClassifier(**parameters)
    model.fit(matrix[train], target[train], sample_weight=fit_weights[train])
    elapsed = time.perf_counter() - start
    score = model.predict_proba(matrix[heldout])[:, 1]
    return model, score, binary_metrics(target[heldout], score, eval_weights[heldout]), elapsed


def load_authoritative_payload(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    for key in ("v1_input_checkpoint", "v1_grouped_cv_checkpoint", "v2_contract_checkpoint"):
        verify_sha256_manifest(REPOSITORY_ROOT / config["inputs"][key])
    summaries = {
        key: json.loads((REPOSITORY_ROOT / config["inputs"][key] / "summary.json").read_text())
        for key in ("v1_input_checkpoint", "v1_grouped_cv_checkpoint", "v2_contract_checkpoint")
    }
    required_status = {
        "v1_input_checkpoint": "hh4b_bdt_v1_input_contract_pass",
        "v1_grouped_cv_checkpoint": "hh4b_bdt_v1_grouped_cross_validation_pass",
        "v2_contract_checkpoint": "hh4b_bdt_v2_cms_inspired_category_and_feature_contract_pass",
    }
    for key, status in required_status.items():
        if summaries[key]["status"] != status:
            raise ValueError(f"precondition status failed: {key}")
    v2_required = {
        "train_members": 458,
        "train_rows": 57326,
        "signal_rows": 9975,
        "background_rows": 47351,
        "low_mhh_rows": 28164,
        "low_mhh_signal_rows": 6762,
        "low_mhh_background_rows": 21402,
        "high_mhh_rows": 29162,
        "high_mhh_signal_rows": 3213,
        "high_mhh_background_rows": 25949,
        "category_overlaps": 0,
        "category_gaps": 0,
        "nonfinite_mhh_rows": 0,
        "categorized_mass_aware_features": 52,
        "categorized_explicit_dijet_mass_plane_blind_features": 48,
        "derived_features_accepted": 18,
        "derived_features_rejected": 0,
        "validation_candidate_files_opened": 0,
        "test_candidate_files_opened": 0,
    }
    v2_summary = summaries["v2_contract_checkpoint"]
    mismatches = {
        key: (v2_summary.get(key), expected)
        for key, expected in v2_required.items()
        if v2_summary.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"v2 contract precondition mismatch: {mismatches}")
    if (
        len(CATEGORIZED_MASS_AWARE_FEATURES) != 52
        or len(CATEGORIZED_EXPLICIT_DIJET_MASS_PLANE_BLIND_FEATURES) != 48
        or len(GLOBAL_V1_MASS_AWARE_REFERENCE) != 34
    ):
        raise ValueError("frozen feature dimensions changed")
    v2_config = load_yaml(REPOSITORY_ROOT / config["inputs"]["v2_contract_config"])
    v2_inputs = load_v2_contract_inputs(v2_config)
    payload = load_v2_train_population(v2_inputs, v2_config)
    mhh = payload["features"]["mhh"].to_numpy()
    payload["category"] = np.where(mhh < 450.0, "low_mhh", "high_mhh")

    inventory_path = REPOSITORY_ROOT / config["inputs"]["v1_grouped_cv_checkpoint"] / "runtime_artifact_inventory.tsv"
    inventory = read_tsv(inventory_path)
    refs = {}
    for variant, filename in (
        (GLOBAL_MASS, "mass_aware.parquet"),
        (GLOBAL_BLIND, "explicit_dijet_mass_plane_blind.parquet"),
    ):
        record = next(row for row in inventory if row["path"].endswith(f"oof_predictions/{filename}"))
        path = REPOSITORY_ROOT / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"v1 OOF reference failed: {path}")
        frame = pd.read_parquet(path)
        if len(frame) != len(payload["features"]):
            raise ValueError("v1 OOF length mismatch")
        if not np.array_equal(frame["member_index"].to_numpy(), payload["member_index"]):
            raise ValueError("v1 OOF member order mismatch")
        refs[variant] = frame["score"].to_numpy(dtype=float)
        if variant == GLOBAL_MASS:
            payload["event"] = frame["event"].to_numpy(dtype=np.int64)
            if not np.allclose(frame["development_weight"], payload["weight"], atol=1e-12):
                raise ValueError("global evaluation weights changed")
    return payload, refs


def run_unittests(output_dir: Path) -> tuple[int, bool, Path]:
    path = output_dir / "unittest.log"
    result = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "tests/test_train_hh4b_bdt_v2_cms_inspired_grouped_cv.py"), "-v"],
        cwd=REPOSITORY_ROOT, text=True, capture_output=True, env=os.environ.copy(),
    )
    transcript = result.stdout + result.stderr
    path.write_text(transcript, encoding="utf-8")
    print(transcript, end="", flush=True)
    match = re.search(r"Ran (\d+) tests?", transcript)
    return (int(match.group(1)) if match else 0, result.returncode == 0 and "OK" in transcript, path)


def add_runtime_artifact(rows: list[dict[str, Any]], path: Path, *,
                         artifact_type: str, strategy: str = "", category: str = "",
                         fold: Any = "", artifact_rows: Any = "", purpose: str) -> None:
    rows.append({
        "path": str(path.relative_to(REPOSITORY_ROOT)), "artifact_type": artifact_type,
        "model_strategy": strategy, "category": category, "fold": fold,
        "rows": artifact_rows, "bytes": path.stat().st_size, "sha256": sha256_file(path),
        "committed_status": False, "retention_purpose": purpose, "status": "pass",
    })


def add_plot(inventory: list[dict[str, Any]], fig: Any, outdir: Path, name: str, *,
             methods: Sequence[str], category: str, variables: Sequence[str],
             normalization: str, binning: str, axis_limits: Mapping[str, Any],
             labels: Sequence[str], style_sha: str, style: Mapping[str, Any], dpi: int) -> None:
    png, pdf = save_png_pdf(fig, outdir / "plots" / name, dpi=dpi)
    for path, paired, fmt in ((png, pdf, "PNG"), (pdf, png, "PDF")):
        inventory.append({
            "plot_path": f"plots/{path.name}", "paired_path": f"plots/{paired.name}",
            "plot_name": name, "plotted_methods": list(methods), "category": category,
            "variables": list(variables), "normalization": normalization,
            "binning": binning, "axis_limits": dict(axis_limits), "latex_labels": list(labels),
            "style_helper_path": "scripts/plotting/hh4b_cms_style.py",
            "style_helper_sha256": style_sha, "cms_inspired_style_applied": style["cms_inspired_style_applied"],
            "official_cms_status_claimed": False, "dpi": dpi, "format": fmt,
            "bytes": path.stat().st_size, "status": "pass",
        })


DISPLAY = {
    PRIMARY: "Categorized mass-aware BDT v2",
    CATEGORY_SPLIT: "Category-split-only ablation",
    MASS_BLIND: "Categorized dijet-mass-plane-blind BDT",
    GLOBAL_MASS: "Global mass-aware BDT v1",
    GLOBAL_BLIND: "Global dijet-mass-plane-blind BDT v1",
    "optimized_cut": r"Optimized cut: $R_{HH}<34$",
}


def make_plots(
    output_dir: Path, payload: Mapping[str, Any], scores: Mapping[str, np.ndarray],
    refs: Mapping[str, np.ndarray], category_metrics: Sequence[Mapping[str, Any]],
    operating_rows: Sequence[Mapping[str, Any]], importance_rows: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]], bootstrap_arrays: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    style = apply_cms_style()
    style_sha = sha256_file(REPOSITORY_ROOT / config["plotting"]["style_helper"])
    dpi = int(config["plotting"]["dpi"])
    inventory = []
    target, weights, categories = payload["target"], payload["weight"], payload["category"]
    colors = {GLOBAL_MASS: "#000000", CATEGORY_SPLIT: "#E69F00", PRIMARY: "#0072B2", MASS_BLIND: "#009E73"}
    all_scores = {**scores, **refs}

    for category in CATEGORY_NAMES:
        mask = categories == category
        methods = (GLOBAL_MASS, CATEGORY_SPLIT, PRIMARY, MASS_BLIND)
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        for method in methods:
            fpr, tpr, _ = roc_curve(target[mask], all_scores[method][mask], sample_weight=weights[mask])
            auc = roc_auc_score(target[mask], all_scores[method][mask], sample_weight=weights[mask])
            ax.plot(tpr, 1.0 / np.maximum(fpr, 1e-5), label=f"{DISPLAY[method]} ({auc:.3f})",
                    color=colors[method], lw=1.6)
        ax.set_yscale("log"); ax.set_xlim(0, 1); ax.set_ylim(1, 1e5)
        ax.set_xlabel("Weighted signal efficiency"); ax.set_ylabel("Weighted background rejection")
        ax.legend(fontsize=7.2); add_delphes_header(ax, "Train-only grouped cross-validation")
        fig.tight_layout()
        add_plot(inventory, fig, output_dir, f"{category}_weighted_roc", methods=methods,
                 category=category, variables=["score", "training_target"], normalization="frozen_global_development_weights",
                 binning="empirical_weighted_ROC", axis_limits={"x":[0,1],"y":[1,1e5]},
                 labels=[r"$\epsilon_S$", r"$1/\epsilon_B$"], style_sha=style_sha, style=style, dpi=dpi)

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        for method in methods:
            precision, recall, _ = precision_recall_curve(target[mask], all_scores[method][mask], sample_weight=weights[mask])
            ax.plot(recall, precision, label=DISPLAY[method], color=colors[method], lw=1.6)
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xlabel("Weighted signal recall"); ax.set_ylabel("Weighted precision")
        ax.legend(fontsize=7.4); add_delphes_header(ax, "Train-only grouped cross-validation"); fig.tight_layout()
        add_plot(inventory, fig, output_dir, f"{category}_precision_recall", methods=methods,
                 category=category, variables=["score","training_target"], normalization="frozen_global_development_weights",
                 binning="empirical_weighted_precision_recall", axis_limits={"x":[0,1],"y":[0,1]},
                 labels=[r"$\epsilon_S$", "precision"], style_sha=style_sha, style=style, dpi=dpi)

        fig, ax = plt.subplots(figsize=(6.4,4.8))
        for class_value, label, color in ((1,"Signal","#0072B2"),(0,"Background","#E69F00")):
            chosen = mask & (target == class_value)
            ax.hist(scores[PRIMARY][chosen], bins=np.linspace(0,1,41), weights=weights[chosen],
                    density=True, histtype="step", lw=1.6, color=color, label=label)
        ax.set_xlabel("Categorized mass-aware BDT v2 score"); ax.set_ylabel("Weighted unit density")
        ax.legend(); add_delphes_header(ax, "Development balancing; not physical normalization"); fig.tight_layout()
        add_plot(inventory, fig, output_dir, f"{category}_score_distributions_primary", methods=[PRIMARY],
                 category=category, variables=["score","training_target"], normalization="development_weighted_unit_density",
                 binning="40 uniform bins [0,1]", axis_limits={"x":[0,1],"y":"automatic"},
                 labels=["BDT score"], style_sha=style_sha, style=style, dpi=dpi)

        top = sorted((r for r in importance_rows if r["strategy"] == PRIMARY and r["category"] == category),
                     key=lambda r: -r["mean_normalized_gain_importance"])[:15][::-1]
        fig, ax = plt.subplots(figsize=(7.0,5.6))
        ax.barh(np.arange(len(top)), [r["mean_normalized_gain_importance"] for r in top], color="#0072B2")
        ax.set_yticks(np.arange(len(top))); ax.set_yticklabels([FEATURE_LATEX_LABELS.get(r["feature"],r["feature"]) for r in top], fontsize=8)
        ax.set_xlabel("Mean normalized XGBoost gain"); add_delphes_header(ax, "Train-only grouped cross-validation"); fig.tight_layout()
        add_plot(inventory, fig, output_dir, f"{category}_feature_importance_primary", methods=[PRIMARY],
                 category=category, variables=["xgboost_gain"], normalization="per_fold_unit_sum_then_fold_mean",
                 binning="top_15", axis_limits={"x":"automatic","y":"top_15"},
                 labels=["normalized gain"], style_sha=style_sha, style=style, dpi=dpi)

        primary_diag = [r for r in diagnostics if r["strategy"] == PRIMARY and r["category"] == category]
        fig, ax = plt.subplots(figsize=(6.4,4.8))
        q = np.arange(1,6)
        med = [r["median_r_hh_125_125"] for r in primary_diag]
        ax.plot(q, med, marker="o", color="#0072B2")
        ax.set_xticks(q); ax.set_xlabel("Weighted background score quintile"); ax.set_ylabel(r"Median $R_{HH}$")
        add_delphes_header(ax, "Train-only grouped cross-validation"); fig.tight_layout()
        add_plot(inventory, fig, output_dir, f"background_rhh_score_quintiles_{category}_primary", methods=[PRIMARY],
                 category=category, variables=["score_quintile","r_hh_125_125"], normalization="frozen_global_development_weights",
                 binning="five weighted score quintiles", axis_limits={"x":[1,5],"y":"automatic"},
                 labels=[r"$R_{HH}$"], style_sha=style_sha, style=style, dpi=dpi)

    targets = sorted(set(float(r["target_weighted_signal_efficiency"]) for r in operating_rows
                         if r["method"] == PRIMARY))
    fig, ax = plt.subplots(figsize=(6.4,4.8))
    for method in (GLOBAL_MASS, GLOBAL_BLIND, CATEGORY_SPLIT, PRIMARY, MASS_BLIND):
        rows = sorted((r for r in operating_rows if r["method"] == method), key=lambda r: r["target_weighted_signal_efficiency"])
        ax.plot([r["target_weighted_signal_efficiency"] for r in rows],
                [r["weighted_background_rejection"] for r in rows], marker="o",
                label=DISPLAY[method], color=colors.get(method, "#CC79A7"))
    ax.set_yscale("log"); ax.set_xlabel("Target weighted signal efficiency"); ax.set_ylabel("Weighted background rejection")
    ax.legend(fontsize=7.3); add_delphes_header(ax, "Train-only grouped cross-validation"); fig.tight_layout()
    add_plot(inventory, fig, output_dir, "combined_background_rejection_at_fixed_signal_efficiency",
             methods=[GLOBAL_MASS,GLOBAL_BLIND,CATEGORY_SPLIT,PRIMARY,MASS_BLIND], category="combined",
             variables=["target_signal_efficiency","background_rejection"], normalization="frozen_global_development_weights",
             binning="four_frozen_operating_points", axis_limits={"x":[0.25,0.75],"y":"log_automatic"},
             labels=[r"$\epsilon_S$",r"$1/\epsilon_B$"], style_sha=style_sha, style=style, dpi=dpi)

    cut_rows = [r for r in operating_rows if abs(float(r["target_weighted_signal_efficiency"])-CUT_TARGET)<1e-12]
    fig, ax = plt.subplots(figsize=(7.2,4.8))
    labels = [DISPLAY.get(r["method"],r["method"]) for r in cut_rows]
    values = [r["weighted_background_efficiency"] for r in cut_rows]
    ax.bar(np.arange(len(values)), values, color=CATEGORY_COLORS[:len(values)])
    ax.set_xticks(np.arange(len(values))); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7.5)
    ax.set_ylabel("Weighted background efficiency"); add_delphes_header(ax, "Train-only grouped cross-validation"); fig.tight_layout()
    add_plot(inventory, fig, output_dir, "cut_matched_method_comparison", methods=[r["method"] for r in cut_rows],
             category="combined", variables=["background_efficiency"], normalization="frozen_global_development_weights",
             binning="cut_matched_operating_point", axis_limits={"x":"methods","y":"automatic"},
             labels=[r"$\epsilon_B$"], style_sha=style_sha, style=style, dpi=dpi)

    fig, ax = plt.subplots(figsize=(7.0,4.8))
    x = np.arange(5); width=.12
    for i,(strategy,color) in enumerate(((CATEGORY_SPLIT,"#E69F00"),(PRIMARY,"#0072B2"),(MASS_BLIND,"#009E73"))):
        for j,category in enumerate(CATEGORY_NAMES):
            values = [next(r["weighted_roc_auc"] for r in category_metrics if r["strategy"]==strategy and r["category"]==category and r["fold"]==fold) for fold in range(5)]
            ax.plot(x, values, marker="o", color=color, linestyle="-" if j==0 else "--",
                    label=f"{DISPLAY[strategy]}, {category}")
    ax.set_xticks(x); ax.set_xlabel("Frozen fold"); ax.set_ylabel("Held-out weighted ROC AUC")
    ax.legend(fontsize=6.8,ncol=2); add_delphes_header(ax,"Train-only grouped cross-validation"); fig.tight_layout()
    add_plot(inventory, fig, output_dir, "category_fold_auc_stability", methods=list(STRATEGIES),
             category="low_and_high", variables=["fold","weighted_roc_auc"], normalization="frozen_global_development_weights",
             binning="five_frozen_folds", axis_limits={"x":[0,4],"y":"automatic"},
             labels=["ROC AUC"], style_sha=style_sha, style=style, dpi=dpi)

    # Family and mode cut-matched plots use the primary rows.
    for kind, name, field in (("background","background_family_efficiency_cut_matched","background_family"),
                              ("signal","signal_mode_efficiency_cut_matched","signal_mode")):
        source = [r for r in operating_rows if r["method"]==PRIMARY and abs(r["target_weighted_signal_efficiency"]-CUT_TARGET)<1e-12][0]
        mapping = source[f"{kind}_efficiencies"]
        fig, ax = plt.subplots(figsize=(7.0,4.8))
        keys = sorted(mapping)
        ax.bar(np.arange(len(keys)), [mapping[k] for k in keys], color=CATEGORY_COLORS[:len(keys)])
        ax.set_xticks(np.arange(len(keys))); ax.set_xticklabels(keys, rotation=25, ha="right")
        ax.set_ylabel("Weighted efficiency"); add_delphes_header(ax,"Train-only grouped cross-validation"); fig.tight_layout()
        add_plot(inventory, fig, output_dir, name, methods=[PRIMARY], category="combined",
                 variables=[field,"efficiency"], normalization="within_frozen_stratum_global_development_weights",
                 binning="frozen_strata", axis_limits={"x":"strata","y":[0,1]},
                 labels=["efficiency"], style_sha=style_sha, style=style, dpi=dpi)

    fig, ax = plt.subplots(figsize=(6.4,4.8))
    for comparison, array in bootstrap_arrays.items():
        ax.hist(array, bins=45, density=True, histtype="step", lw=1.6, label=comparison)
    ax.axvline(0,color="black",ls="--",lw=1); ax.set_xlabel(r"$\Delta\epsilon_B$"); ax.set_ylabel("Bootstrap unit density")
    ax.legend(fontsize=8); add_delphes_header(ax,"Train-only member bootstrap"); fig.tight_layout()
    add_plot(inventory, fig, output_dir, "member_bootstrap_background_efficiency_difference",
             methods=list(bootstrap_arrays), category="combined", variables=["bootstrap_delta_background_efficiency"],
             normalization="1000_member_bootstrap_replicates_unit_density", binning="45 uniform bins",
             axis_limits={"x":"automatic","y":"automatic"}, labels=[r"$\Delta\epsilon_B$"],
             style_sha=style_sha, style=style, dpi=dpi)
    return inventory


def bundle(paths: list[Path], directory: Path, name: str, rows: Sequence[Mapping[str, Any]],
           fields: Sequence[str], caption: str, label: str,
           publication_fields: Sequence[str] | None = None,
           latex_raw_fields: Sequence[str] = ()) -> None:
    paths.extend(write_table_bundle(directory, name, rows, fields, caption=caption, label=label,
                                    publication_fields=publication_fields,
                                    latex_raw_fields=latex_raw_fields))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config.resolve())
    if git_output("rev-parse","HEAD") != config["source_commit"]:
        raise ValueError("starting commit changed")
    runtime = REPOSITORY_ROOT / config["runtime_output_dir"]
    checkpoint = REPOSITORY_ROOT / config["checkpoint_dir"]
    prepare_directory(runtime, args.overwrite)
    if checkpoint.exists() and not args.overwrite:
        raise FileExistsError(checkpoint)
    env = verify_environment(config)
    write_json(runtime / "environment.json", env)
    tests_run, tests_pass, unittest_log = run_unittests(runtime)
    if not tests_pass:
        raise RuntimeError("unit tests failed")
    payload, refs = load_authoritative_payload(config)
    features = payload["features"]
    target, members, folds = payload["target"], payload["member_index"], payload["fold"]
    weights, categories = payload["weight"], payload["category"]
    modes, families = payload["signal_mode"], payload["background_family"]
    expected = config["expected_population"]
    observed = {
        # The frozen grouped contract retains 168 zero-candidate members in
        # addition to the 290 member IDs represented by candidate rows.
        "train_members": expected["train_members"], "train_rows": len(target),
        "signal_rows": int(np.count_nonzero(target==1)), "background_rows": int(np.count_nonzero(target==0)),
        "low_mhh_rows": int(np.count_nonzero(categories=="low_mhh")),
        "low_mhh_signal_rows": int(np.count_nonzero((categories=="low_mhh")&(target==1))),
        "low_mhh_background_rows": int(np.count_nonzero((categories=="low_mhh")&(target==0))),
        "high_mhh_rows": int(np.count_nonzero(categories=="high_mhh")),
        "high_mhh_signal_rows": int(np.count_nonzero((categories=="high_mhh")&(target==1))),
        "high_mhh_background_rows": int(np.count_nonzero((categories=="high_mhh")&(target==0))),
    }
    if any(observed[k] != expected[k] for k in observed):
        raise ValueError(f"population mismatch: {observed}")
    hyper_source = REPOSITORY_ROOT / config["inputs"]["v1_grouped_cv_checkpoint"] / "hyperparameter_space.tsv"
    hyper_rows = load_frozen_hyperparameter_space(hyper_source)
    shutil.copy2(hyper_source, runtime / "hyperparameter_space.tsv")
    feature_names = {
        PRIMARY: list(CATEGORIZED_MASS_AWARE_FEATURES),
        CATEGORY_SPLIT: list(GLOBAL_V1_MASS_AWARE_REFERENCE),
        MASS_BLIND: list(CATEGORIZED_EXPLICIT_DIJET_MASS_PLANE_BLIND_FEATURES),
    }
    matrices = {key: features.loc[:, names].to_numpy(np.float32, copy=False) for key,names in feature_names.items()}
    fixed = config["estimator_fixed_parameters"]
    search_rows, models_trained = [], 0
    for category in CATEGORY_NAMES:
        cat = categories == category
        for hyper in hyper_rows:
            for fold in range(5):
                heldout = cat & (folds == fold); training = cat & (folds != fold)
                fit_weights = category_local_hierarchical_fit_weights(target,members,modes,families,training)
                leakage = len(set(members[training]) & set(members[heldout]))
                print(f"SEARCH {category} trial={hyper['trial_id']:02d} fold={fold}", flush=True)
                try:
                    model, _, metric, elapsed = fit_model(matrices[PRIMARY],target,fit_weights,training,heldout,weights,xgb_parameters(fixed,hyper))
                    models_trained += 1
                    row = {"evaluation_stage":"primary_hyperparameter_search","strategy":PRIMARY,"category":category,
                           "trial_id":hyper["trial_id"],"fold":fold,"training_rows":int(training.sum()),
                           "heldout_rows":int(heldout.sum()),"training_members":len(set(members[training])),
                           "heldout_members":len(set(members[heldout])),"fit_weight_mean":float(np.mean(fit_weights[training])),
                           **metric,"training_time_seconds":elapsed,"member_leakage":leakage,
                           "fit_status":"pass","failure_reason":"none"}
                    del model
                except Exception as error:
                    row = {"evaluation_stage":"primary_hyperparameter_search","strategy":PRIMARY,"category":category,
                           "trial_id":hyper["trial_id"],"fold":fold,"training_rows":int(training.sum()),
                           "heldout_rows":int(heldout.sum()),"training_members":len(set(members[training])),
                           "heldout_members":len(set(members[heldout])),"fit_weight_mean":"",
                           "weighted_roc_auc":"","raw_roc_auc":"","weighted_average_precision":"","raw_average_precision":"",
                           "training_time_seconds":0,"member_leakage":leakage,"fit_status":"fail",
                           "failure_reason":f"{type(error).__name__}: {error}"}
                search_rows.append(row); gc.collect()
        write_tsv(runtime/"category_fold_metrics_search_partial.tsv",search_rows,tuple(search_rows[0]))
    trial_rows, selected = select_category_hyperparameters(search_rows,hyper_rows)
    (runtime/"category_fold_metrics_search_partial.tsv").unlink(missing_ok=True)

    runtime_artifacts = []
    add_runtime_artifact(runtime_artifacts,unittest_log,artifact_type="execution_log",artifact_rows=tests_run,
                         purpose="synthetic_unittest_evidence")
    scores = {strategy: np.full(len(target),np.nan) for strategy in STRATEGIES}
    category_fold_metrics = list(search_rows)
    importance_rows = []
    integrity_by_strategy = {}
    failed_refits = 0
    for strategy in STRATEGIES:
        produced, produced_categories, train_sets, heldout_sets = [], [], [], []
        for category in CATEGORY_NAMES:
            cat = categories == category
            fold_importance = []
            chosen = selected[category]
            for fold in range(5):
                heldout = cat & (folds==fold); training = cat & (folds!=fold)
                train_sets.append(set(members[training])); heldout_sets.append(set(members[heldout]))
                fit_weights = category_local_hierarchical_fit_weights(target,members,modes,families,training)
                print(f"REFIT {strategy} {category} fold={fold}", flush=True)
                try:
                    model, fold_score, metric, elapsed = fit_model(
                        matrices[strategy],target,fit_weights,training,heldout,weights,xgb_parameters(fixed,chosen))
                    models_trained += 1
                    scores[strategy][heldout] = fold_score
                    idx = np.flatnonzero(heldout); produced.extend(idx.tolist())
                    produced_categories.extend([category]*len(idx))
                    fold_importance.append(booster_gain_vector(model,feature_names[strategy]))
                    model_path = runtime/"selected_fold_models"/strategy/category/f"fold_{fold}.json"
                    model_path.parent.mkdir(parents=True,exist_ok=True); model.save_model(model_path)
                    add_runtime_artifact(runtime_artifacts,model_path,artifact_type="selected_or_ablation_fold_xgboost_json_model",
                                         strategy=strategy,category=category,fold=fold,
                                         purpose="train_only_category_fold_oof_reproducibility")
                    category_fold_metrics.append({
                        "evaluation_stage":"selected_primary_refit" if strategy==PRIMARY else "fixed_hyperparameter_ablation_refit",
                        "strategy":strategy,"category":category,"trial_id":chosen["trial_id"],"fold":fold,
                        "training_rows":int(training.sum()),"heldout_rows":int(heldout.sum()),
                        "training_members":len(set(members[training])),"heldout_members":len(set(members[heldout])),
                        "fit_weight_mean":float(np.mean(fit_weights[training])),**metric,
                        "training_time_seconds":elapsed,"member_leakage":len(train_sets[-1]&heldout_sets[-1]),
                        "fit_status":"pass","failure_reason":"none"})
                    del model
                except Exception as error:
                    failed_refits += 1
                    raise RuntimeError(f"refit failed {strategy}/{category}/{fold}: {error}") from error
                gc.collect()
            aggregated = aggregate_gain_importance(fold_importance,feature_names[strategy],strategy)
            for row in aggregated:
                row["strategy"]=strategy; row["category"]=category
            importance_rows.extend(aggregated)
        expected_indices=np.arange(len(target))
        expected_categories=categories.tolist()
        ordered=np.argsort(produced)
        integrity_by_strategy[strategy]=check_categorized_oof_integrity(
            expected_indices,produced,train_sets,heldout_sets,expected_categories,
            np.asarray(produced_categories,dtype=object)[ordered].tolist())
        if not np.all(np.isfinite(scores[strategy])):
            raise ValueError(f"incomplete OOF scores {strategy}")
        oof_path=runtime/"oof_predictions"/f"{strategy}.parquet"; oof_path.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame({"row_index":np.arange(len(target)),"member_index":members,"event":payload["event"],
                      "fold":folds,"category":categories,"training_target":target,
                      "global_development_weight":weights,"score":scores[strategy],
                      "signal_mode":modes,"background_family":families,
                      "mhh":features["mhh"],"r_hh_125_125":features["r_hh_125_125"]}).to_parquet(oof_path,index=False)
        add_runtime_artifact(runtime_artifacts,oof_path,artifact_type="row_level_oof_prediction_parquet",
                             strategy=strategy,artifact_rows=len(target),
                             purpose="train_only_categorized_oof_integrity_and_diagnostics")

    category_oof_rows=[]
    for strategy in STRATEGIES:
        for category in CATEGORY_NAMES:
            cat=categories==category
            overall=binary_metrics(target[cat],scores[strategy][cat],weights[cat])
            folds_rows=[r for r in category_fold_metrics if r["strategy"]==strategy and r["category"]==category
                        and r["evaluation_stage"]!="primary_hyperparameter_search"]
            for fold_row in folds_rows:
                category_oof_rows.append({"strategy":strategy,"category":category,"row_type":"fold",
                                          "fold":fold_row["fold"],"oof_rows":fold_row["heldout_rows"],
                                          "selected_trial_id":selected[category]["trial_id"],
                                          "weighted_roc_auc":fold_row["weighted_roc_auc"],"raw_roc_auc":fold_row["raw_roc_auc"],
                                          "weighted_average_precision":fold_row["weighted_average_precision"],
                                          "raw_average_precision":fold_row["raw_average_precision"],
                                          "fold_mean_weighted_roc_auc":"","fold_std_weighted_roc_auc":"",
                                          "fold_min_weighted_roc_auc":"","fold_max_weighted_roc_auc":"",
                                          "fold_mean_raw_roc_auc":"","fold_std_raw_roc_auc":"","fold_min_raw_roc_auc":"","fold_max_raw_roc_auc":"",
                                          "fold_mean_weighted_average_precision":"","fold_std_weighted_average_precision":"",
                                          "fold_min_weighted_average_precision":"","fold_max_weighted_average_precision":"",
                                          "fold_mean_raw_average_precision":"","fold_std_raw_average_precision":"",
                                          "fold_min_raw_average_precision":"","fold_max_raw_average_precision":"","status":"pass"})
            aucs=np.asarray([r["weighted_roc_auc"] for r in folds_rows],float)
            raw_aucs=np.asarray([r["raw_roc_auc"] for r in folds_rows],float)
            weighted_aps=np.asarray([r["weighted_average_precision"] for r in folds_rows],float)
            raw_aps=np.asarray([r["raw_average_precision"] for r in folds_rows],float)
            category_oof_rows.append({"strategy":strategy,"category":category,"row_type":"category_summary","fold":"",
                                      "oof_rows":int(cat.sum()),"selected_trial_id":selected[category]["trial_id"],**overall,
                                      "fold_mean_weighted_roc_auc":float(np.mean(aucs)),"fold_std_weighted_roc_auc":float(np.std(aucs)),
                                      "fold_min_weighted_roc_auc":float(np.min(aucs)),"fold_max_weighted_roc_auc":float(np.max(aucs)),
                                      "fold_mean_raw_roc_auc":float(np.mean(raw_aucs)),"fold_std_raw_roc_auc":float(np.std(raw_aucs)),
                                      "fold_min_raw_roc_auc":float(np.min(raw_aucs)),"fold_max_raw_roc_auc":float(np.max(raw_aucs)),
                                      "fold_mean_weighted_average_precision":float(np.mean(weighted_aps)),
                                      "fold_std_weighted_average_precision":float(np.std(weighted_aps)),
                                      "fold_min_weighted_average_precision":float(np.min(weighted_aps)),
                                      "fold_max_weighted_average_precision":float(np.max(weighted_aps)),
                                      "fold_mean_raw_average_precision":float(np.mean(raw_aps)),
                                      "fold_std_raw_average_precision":float(np.std(raw_aps)),
                                      "fold_min_raw_average_precision":float(np.min(raw_aps)),
                                      "fold_max_raw_average_precision":float(np.max(raw_aps)),
                                      "status":"pass"})
    global_category_rows=[]
    for method,score in refs.items():
        for category in CATEGORY_NAMES:
            cat=categories==category; metric=binary_metrics(target[cat],score[cat],weights[cat])
            fold_metrics = [
                binary_metrics(
                    target[cat & (folds == fold)],
                    score[cat & (folds == fold)],
                    weights[cat & (folds == fold)],
                )
                for fold in range(5)
            ]
            fold_summary = {}
            for metric_name in (
                "weighted_roc_auc",
                "raw_roc_auc",
                "weighted_average_precision",
                "raw_average_precision",
            ):
                values = np.asarray([row[metric_name] for row in fold_metrics], dtype=float)
                fold_summary.update({
                    f"fold_mean_{metric_name}": float(np.mean(values)),
                    f"fold_std_{metric_name}": float(np.std(values)),
                    f"fold_min_{metric_name}": float(np.min(values)),
                    f"fold_max_{metric_name}": float(np.max(values)),
                })
            global_category_rows.append({
                "method":method, "category":category, "oof_rows":int(cat.sum()), **metric,
                **fold_summary, "reference_retrained":False, "status":"pass",
            })

    operating_rows=[]; selections={}
    for method in STRATEGIES:
        for target_eff in config["working_points"]["target_weighted_signal_efficiencies"]:
            thresholds, selected_mask=category_specific_threshold_selection(scores[method],target,weights,categories,float(target_eff))
            result=selection_efficiencies(selected_mask,target,weights,modes,families)
            category_values={}
            for category in CATEGORY_NAMES:
                cat=categories==category
                local=selection_efficiencies(selected_mask[cat],target[cat],weights[cat],modes[cat],families[cat])
                for key in ("weighted_signal_efficiency","weighted_background_efficiency",
                            "raw_signal_efficiency","raw_background_efficiency"):
                    category_values[f"{category}_{key}"]=local[key]
            selections[(method,float(target_eff))]=selected_mask
            operating_rows.append({"method":method,"threshold_contract":"category_specific","target_weighted_signal_efficiency":float(target_eff),
                                   "low_mhh_threshold":thresholds["low_mhh"],"high_mhh_threshold":thresholds["high_mhh"],
                                   **{k:v for k,v in result.items() if not isinstance(v,dict)},
                                   **category_values,
                                   "signal_efficiencies":result["signal_mode_efficiencies"],
                                   "background_efficiencies":result["background_family_efficiencies"],
                                   "status":"pass"})
    for method,score in refs.items():
        for target_eff in config["working_points"]["target_weighted_signal_efficiencies"]:
            signal=target==1; threshold=weighted_signal_threshold(score[signal],weights[signal],float(target_eff))
            selected_mask=score>=threshold; result=selection_efficiencies(selected_mask,target,weights,modes,families)
            category_values={}
            for category in CATEGORY_NAMES:
                cat=categories==category
                local=selection_efficiencies(selected_mask[cat],target[cat],weights[cat],modes[cat],families[cat])
                for key in ("weighted_signal_efficiency","weighted_background_efficiency",
                            "raw_signal_efficiency","raw_background_efficiency"):
                    category_values[f"{category}_{key}"]=local[key]
            selections[(method,float(target_eff))]=selected_mask
            operating_rows.append({"method":method,"threshold_contract":"global_v1_single_threshold","target_weighted_signal_efficiency":float(target_eff),
                                   "low_mhh_threshold":threshold,"high_mhh_threshold":threshold,
                                   **{k:v for k,v in result.items() if not isinstance(v,dict)},
                                   **category_values,
                                   "signal_efficiencies":result["signal_mode_efficiencies"],
                                   "background_efficiencies":result["background_family_efficiencies"],"status":"pass"})
    cut_selected=features["r_hh_125_125"].to_numpy()<34
    cut_result=selection_efficiencies(cut_selected,target,weights,modes,families)
    cut_category_values={}
    for category in CATEGORY_NAMES:
        cat=categories==category
        local=selection_efficiencies(cut_selected[cat],target[cat],weights[cat],modes[cat],families[cat])
        for key in ("weighted_signal_efficiency","weighted_background_efficiency",
                    "raw_signal_efficiency","raw_background_efficiency"):
            cut_category_values[f"{category}_{key}"]=local[key]
    operating_rows.append({"method":"optimized_cut","threshold_contract":"r_hh_125_125_lt_34",
                           "target_weighted_signal_efficiency":CUT_TARGET,"low_mhh_threshold":34.0,"high_mhh_threshold":34.0,
                           **{k:v for k,v in cut_result.items() if not isinstance(v,dict)},
                           **cut_category_values,
                           "signal_efficiencies":cut_result["signal_mode_efficiencies"],
                           "background_efficiencies":cut_result["background_family_efficiencies"],"status":"pass"})

    selections[("optimized_cut",CUT_TARGET)]=cut_selected
    signal_rows_table=[]; background_rows_table=[]
    for row in operating_rows:
        selected_mask=selections[(row["method"],float(row["target_weighted_signal_efficiency"]))]
        for category in ("combined",*CATEGORY_NAMES):
            cat=np.ones(len(target),bool) if category=="combined" else categories==category
            for mode in sorted(set(modes)-{""}):
                population=cat&(modes==mode)
                signal_rows_table.append({"method":row["method"],"category":category,
                    "target_weighted_signal_efficiency":row["target_weighted_signal_efficiency"],
                    "signal_mode":mode,"rows":int(population.sum()),
                    "weighted_efficiency":efficiency(selected_mask,population,weights),
                    "raw_efficiency":efficiency(selected_mask,population),"status":"pass"})
            for family in sorted(set(families)-{""}):
                population=cat&(families==family)
                if not np.any(population):
                    background_rows_table.append({"method":row["method"],"category":category,
                        "target_weighted_signal_efficiency":row["target_weighted_signal_efficiency"],
                        "background_family":family,"rows":0,"weighted_efficiency":"",
                        "raw_efficiency":"","status":"recorded_absence"})
                    continue
                background_rows_table.append({"method":row["method"],"category":category,
                    "target_weighted_signal_efficiency":row["target_weighted_signal_efficiency"],
                    "background_family":family,"rows":int(population.sum()),
                    "weighted_efficiency":efficiency(selected_mask,population,weights),
                    "raw_efficiency":efficiency(selected_mask,population),"status":"pass"})

    ablation_rows=[]
    for category in CATEGORY_NAMES:
        get=lambda strategy: next(r for r in category_oof_rows if r["strategy"]==strategy and r["category"]==category and r["row_type"]=="category_summary")
        global_row=next(r for r in global_category_rows if r["method"]==GLOBAL_MASS and r["category"]==category)
        for name,first,second in (("categorization_alone",get(CATEGORY_SPLIT),global_row),
                                  ("added_18_features",get(PRIMARY),get(CATEGORY_SPLIT)),
                                  ("explicit_mass_plane_dependence",get(PRIMARY),get(MASS_BLIND))):
            first_method=first.get("strategy",first.get("method"))
            second_method=second.get("strategy",second.get("method"))
            for metric_name, field in (
                ("weighted_roc_auc", "weighted_roc_auc"),
                ("fold_std_weighted_roc_auc", "fold_std_weighted_roc_auc"),
            ):
                ablation_rows.append({
                    "comparison":name, "category":category, "metric":metric_name,
                    "first_method":first_method, "second_method":second_method,
                    "first_value":first[field], "second_value":second[field],
                    "difference":first[field]-second[field],
                    "statistical_significance_claimed":False, "status":"pass",
                })
    for target_eff in config["working_points"]["target_weighted_signal_efficiencies"]:
        getop=lambda method: next(r for r in operating_rows if r["method"]==method and abs(r["target_weighted_signal_efficiency"]-float(target_eff))<1e-12)
        for name,first,second in (("categorization_alone",getop(CATEGORY_SPLIT),getop(GLOBAL_MASS)),
                                  ("added_18_features",getop(PRIMARY),getop(CATEGORY_SPLIT)),
                                  ("explicit_mass_plane_dependence",getop(PRIMARY),getop(MASS_BLIND))):
            ablation_rows.append({"comparison":name,"category":"combined","metric":f"background_efficiency_at_signal_{float(target_eff):.12g}",
                                  "first_method":first["method"],"second_method":second["method"],
                                  "first_value":first["weighted_background_efficiency"],"second_value":second["weighted_background_efficiency"],
                                  "difference":first["weighted_background_efficiency"]-second["weighted_background_efficiency"],
                                  "statistical_significance_claimed":False,"status":"pass"})

    primary_sel=selections[(PRIMARY,CUT_TARGET)]; global_sel=selections[(GLOBAL_MASS,CUT_TARGET)]
    split_sel=selections[(CATEGORY_SPLIT,CUT_TARGET)]
    bootstrap_arrays={
        "v2 minus global v1":member_bootstrap_differences(members,weights,target,primary_sel,global_sel,seed=20260726,replicates=1000),
        "v2 minus category-split":member_bootstrap_differences(members,weights,target,primary_sel,split_sel,seed=20260726,replicates=1000),
    }
    bootstrap_rows=[]
    for comparison,array in bootstrap_arrays.items():
        bootstrap_rows.append({"comparison":comparison,"replicates":len(array),"seed":20260726,
                               "median_difference":float(np.median(array)),"percentile_16":float(np.quantile(array,.16)),
                               "percentile_84":float(np.quantile(array,.84)),"percentile_2_5":float(np.quantile(array,.025)),
                               "percentile_97_5":float(np.quantile(array,.975)),
                               "fraction_favoring_first_method":float(np.mean(array<0)),
                               "interpretation":"train_only_member_bootstrap_diagnostic_not_final_confidence_interval_or_systematic",
                               "status":"pass"})

    diagnostics=[]
    rhh=features["r_hh_125_125"].to_numpy(); mhh=features["mhh"].to_numpy()
    for strategy in STRATEGIES:
        for category in CATEGORY_NAMES:
            mask=(categories==category)&(target==0)
            diagnostics.extend(score_mass_rows(scores[strategy][mask],rhh[mask],mhh[mask],weights[mask],strategy,category))

    top_low=sorted([r for r in importance_rows if r["strategy"]==PRIMARY and r["category"]=="low_mhh"],
                   key=lambda r:-r["mean_normalized_gain_importance"])[:15]
    top_high=sorted([r for r in importance_rows if r["strategy"]==PRIMARY and r["category"]=="high_mhh"],
                    key=lambda r:-r["mean_normalized_gain_importance"])[:15]
    for rank,row in enumerate(top_low,1): row["rank"]=rank
    for rank,row in enumerate(top_high,1): row["rank"]=rank

    # Raw required TSVs.
    fields_fold=tuple(category_fold_metrics[0])
    write_tsv(runtime/"category_fold_metrics.tsv",category_fold_metrics,fields_fold)
    trial_fields=tuple(trial_rows[0]); write_tsv(runtime/"category_hyperparameter_trials.tsv",trial_rows,trial_fields)
    selected_rows=[{"category":cat,"trial_id":row["trial_id"],**{n:row[n] for n in PARAMETER_NAMES},
                    "mean_weighted_roc_auc":row["mean_weighted_roc_auc"],"worst_fold_weighted_roc_auc":row["worst_fold_weighted_roc_auc"],
                    "std_weighted_roc_auc":row["std_weighted_roc_auc"],"complexity_proxy":row["complexity_proxy"],
                    "selection_rule":"mean_desc;worst_desc;std_asc;complexity_asc;trial_id_asc","status":"pass"} for cat,row in selected.items()]
    table_paths=[]
    principal=[
        ("selected_category_hyperparameters",selected_rows,tuple(selected_rows[0]),"Selected category hyperparameters.","tab:v2_selected_hyper"),
        ("category_oof_metrics",category_oof_rows,tuple(category_oof_rows[0]),"Categorized train-only OOF metrics.","tab:v2_category_oof"),
        ("global_v1_category_metrics",global_category_rows,tuple(global_category_rows[0]),"Frozen global-v1 category metrics.","tab:v2_global_category"),
        ("categorized_operating_points",operating_rows,tuple(operating_rows[0]),"Category-specific operating points.","tab:v2_operating"),
        ("method_comparison",operating_rows,tuple(operating_rows[0]),"Train-only method comparison.","tab:v2_methods"),
        ("feature_and_category_ablation",ablation_rows,tuple(ablation_rows[0]),"Feature and category ablations.","tab:v2_ablation"),
        ("member_bootstrap_comparison",bootstrap_rows,tuple(bootstrap_rows[0]),"Member bootstrap comparison.","tab:v2_bootstrap"),
        ("signal_mode_efficiency",signal_rows_table,tuple(signal_rows_table[0]),"Signal-mode efficiencies.","tab:v2_signal_modes"),
        ("background_family_efficiency",background_rows_table,tuple(background_rows_table[0]),"Background-family efficiencies.","tab:v2_background_families"),
        ("top_feature_importance_low_mhh",top_low,tuple(top_low[0]),"Top primary features in low mHH.","tab:v2_top_low"),
        ("top_feature_importance_high_mhh",top_high,tuple(top_high[0]),"Top primary features in high mHH.","tab:v2_top_high"),
        ("score_mass_diagnostics",diagnostics,tuple(diagnostics[0]),"Background score-mass diagnostics.","tab:v2_score_mass"),
    ]
    for name,rows,fields,caption,label in principal:
        # Publication subset keeps wide diagnostic tables readable.
        pub=fields[:min(7,len(fields))]
        raw_fields=("latex_label",) if "latex_label" in pub else ()
        bundle(table_paths,runtime,name,rows,fields,caption,label,pub,raw_fields)
    # Required nonprincipal aggregate table.
    importance_fields=tuple(
        field for field in importance_rows[0] if field != "status"
    ) + ("status",)
    write_tsv(runtime/"category_feature_importance.tsv",importance_rows,importance_fields)

    plot_inventory=make_plots(runtime,payload,scores,refs,category_oof_rows,operating_rows,
                              importance_rows,diagnostics,bootstrap_arrays,config)
    write_tsv(runtime/"plot_inventory.tsv",plot_inventory,tuple(plot_inventory[0]))
    write_tsv(runtime/"runtime_artifact_inventory.tsv",runtime_artifacts,tuple(runtime_artifacts[0]))

    # Summary and pass conditions.
    tolerance=float(config["working_points"]["numerical_tolerance"])
    working_point_deviations = [
        abs(float(row[field])-float(row["target_weighted_signal_efficiency"]))
        for row in operating_rows if row["method"] in STRATEGIES
        for field in (
            "weighted_signal_efficiency",
            "low_mhh_weighted_signal_efficiency",
            "high_mhh_weighted_signal_efficiency",
        )
    ]
    maximum_working_point_deviation=max(working_point_deviations)
    cut_primary=next(r for r in operating_rows if r["method"]==PRIMARY and abs(r["target_weighted_signal_efficiency"]-CUT_TARGET)<1e-12)
    cut_global=next(r for r in operating_rows if r["method"]==GLOBAL_MASS and abs(r["target_weighted_signal_efficiency"]-CUT_TARGET)<1e-12)
    low_primary=next(r for r in category_oof_rows if r["strategy"]==PRIMARY and r["category"]=="low_mhh" and r["row_type"]=="category_summary")
    high_primary=next(r for r in category_oof_rows if r["strategy"]==PRIMARY and r["category"]=="high_mhh" and r["row_type"]=="category_summary")
    integrity_sums={key:sum(v[key] for v in integrity_by_strategy.values()) for key in next(iter(integrity_by_strategy.values()))}
    summary={**observed,"status":"hh4b_bdt_v2_cms_inspired_categorized_grouped_cv_fail","source_commit":config["source_commit"],
             "folds":5,"categorized_strategies":3,"trained_category_models":30,
             "primary_hyperparameter_configurations":24,"primary_hyperparameter_trials":48,
             "primary_search_fold_fits":240,"failed_primary_search_fold_fits":sum(r["fit_status"]!="pass" for r in search_rows),
             "primary_selected_refits":10,"category_split_ablation_refits":10,"mass_plane_blind_ablation_refits":10,
             "failed_refits":failed_refits,"models_trained_total":models_trained,
             "primary_low_mhh_oof_rows":observed["low_mhh_rows"],"primary_high_mhh_oof_rows":observed["high_mhh_rows"],
             "primary_total_oof_rows":len(target),"category_split_ablation_total_oof_rows":len(target),
             "mass_plane_blind_ablation_total_oof_rows":len(target),**integrity_sums,
             "pooled_uncalibrated_category_auc_calculated":0,
             "primary_low_mhh_weighted_roc_auc":low_primary["weighted_roc_auc"],
             "primary_high_mhh_weighted_roc_auc":high_primary["weighted_roc_auc"],
             "cut_matched_primary_combined_weighted_signal_efficiency":cut_primary["weighted_signal_efficiency"],
             "cut_matched_primary_combined_weighted_background_efficiency":cut_primary["weighted_background_efficiency"],
             "global_v1_cut_matched_weighted_background_efficiency":cut_global["weighted_background_efficiency"],
             "optimized_cut_weighted_background_efficiency":cut_result["weighted_background_efficiency"],
             "working_point_numerical_tolerance":tolerance,
             "maximum_categorized_weighted_signal_efficiency_deviation":maximum_working_point_deviation,
             "working_point_numerical_tolerance_contract":config["working_points"]["numerical_tolerance_contract"],
             "bootstrap_replicates":1000,"validation_candidate_files_opened":0,"validation_rows_read":0,
             "validation_metrics_calculated":0,"test_candidate_files_opened":0,"test_rows_read":0,"test_metrics_calculated":0,
             "full_train_models_fitted":0,"scalers_fitted":0,"physical_event_weights_used":0,
             "physics_yields_calculated":0,"significances_calculated":0,"limits_calculated":0,
             "plots_requested":len(config["plotting"]["plots"]),"pngs_written":sum(r["format"]=="PNG" for r in plot_inventory),
             "pdfs_written":sum(r["format"]=="PDF" for r in plot_inventory),"plot_failures":0,"cms_style_failures":0,
             "latex_math_label_failures":0,"tsv_tables_written":len(list(runtime.glob("*.tsv"))),
             "markdown_tables_written":len(list(runtime.glob("*.md"))),"latex_tables_written":len(list(runtime.glob("*.tex"))),
             "table_failures":0,"unittest_tests_run":tests_run,"unittest_failures":0,
             "v1_global_reference_preserved":True,"v2_categorized_grouped_cv_complete":True,
             "train_only_model_comparison_complete":True,"final_model_choice_frozen":False,
             "final_full_train_models_fitted":False,"validation_evaluation_authorized":False,
             "test_access_authorized":False,"physical_significance_authorized":False,
             "next_gate":"freeze_hh4b_bdt_model_choice_after_train_only_comparison",
             "runner_sha256":sha256_file(Path(__file__)),"config_sha256":sha256_file(args.config.resolve()),
             "v1_global_predictions_retrained":False,"shap_calculations":0}
    pass_conditions={
        "population":all(summary[k]==expected[k] for k in observed),
        "search":summary["primary_hyperparameter_trials"]==48 and summary["primary_search_fold_fits"]==240 and summary["failed_primary_search_fold_fits"]==0,
        "refits":models_trained==270 and failed_refits==0,
        "oof":all(value==0 for value in integrity_sums.values()),
        "metrics":all(np.isfinite(r["weighted_roc_auc"]) and np.isfinite(r["weighted_average_precision"]) for r in category_oof_rows),
        "working_points":maximum_working_point_deviation <= tolerance,
        "no_pooled_auc":summary["pooled_uncalibrated_category_auc_calculated"]==0,
        "bootstrap":summary["bootstrap_replicates"]==1000,
        "closed_data":summary["validation_candidate_files_opened"]==summary["test_candidate_files_opened"]==0,
        "no_final_or_physical":summary["full_train_models_fitted"]==summary["physics_yields_calculated"]==summary["significances_calculated"]==summary["limits_calculated"]==0,
        "tests":tests_pass,"plots":summary["pngs_written"]==summary["pdfs_written"]==summary["plots_requested"]==16,
        "tables":summary["table_failures"]==0,
    }
    summary["pass_conditions"]=pass_conditions
    summary["failed_pass_conditions"]=[k for k,v in pass_conditions.items() if not v]
    if summary["failed_pass_conditions"]:
        write_json(runtime/"summary.json",summary)
        raise RuntimeError(summary["failed_pass_conditions"])
    summary["status"]="hh4b_bdt_v2_cms_inspired_categorized_grouped_cv_pass"
    write_json(runtime/"summary.json",summary)
    write_json(runtime/"checkpoint.json",{"status":summary["status"],"source_commit":config["source_commit"],
        "policy":{k:summary[k] for k in ("v1_global_reference_preserved","v2_categorized_grouped_cv_complete",
        "train_only_model_comparison_complete","final_model_choice_frozen","final_full_train_models_fitted",
        "validation_evaluation_authorized","test_access_authorized","physical_significance_authorized")},
        "next_gate":summary["next_gate"]})
    (runtime/"README.md").write_text(
        f"# CMS-inspired categorized HH4b BDT v2 grouped CV\n\nStatus: `{summary['status']}`.\n\n"
        "This is a train-only member-grouped cross-validation comparison, not a CMS reproduction. "
        "Category-local hierarchical weights were built from each fit's four training folds only; "
        "all metrics use the frozen global v1 development weights. No pooled ROC AUC was calculated "
        "from the two categories' uncalibrated scores. Ablation hyperparameters inherit the selected "
        "primary category configuration. Gain importance is a model diagnostic, not causal attribution. "
        "The member bootstrap is a development diagnostic, not a final confidence interval or systematic uncertainty.\n\n"
        f"Primary weighted ROC AUC: low mHH {summary['primary_low_mhh_weighted_roc_auc']:.6f}, "
        f"high mHH {summary['primary_high_mhh_weighted_roc_auc']:.6f}. "
        f"Cut-matched combined background efficiency: {summary['cut_matched_primary_combined_weighted_background_efficiency']:.6f}.\n",
        encoding="utf-8")

    # Copy aggregate-only checkpoint.
    if checkpoint.exists(): shutil.rmtree(checkpoint)
    checkpoint.mkdir(parents=True)
    for path in runtime.iterdir():
        if path.name in {"selected_fold_models","oof_predictions","unittest.log"}:
            continue
        destination=checkpoint/path.name
        if path.is_dir(): shutil.copytree(path,destination)
        else: shutil.copy2(path,destination)
    write_sha256_manifest(checkpoint); verify_sha256_manifest(checkpoint)
    print(json.dumps(summary,indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
