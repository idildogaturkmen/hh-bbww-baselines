#!/usr/bin/env python3
"""Shared fail-closed helpers for PN-c7 publication and ML checkpoints."""

from __future__ import annotations

import csv
from decimal import Decimal, localcontext
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


CHECKPOINT_ROOT = Path("/uscms_data/d3/iturkmen/hh4b_delphes/harvest_checkpoints")
CHECKPOINTS = {
    "c7q": CHECKPOINT_ROOT / "pn_c7q_full441_train_physical_tables_20260802_v1",
    "c7r": CHECKPOINT_ROOT / "pn_c7r_threeb_fourb_multijet_transfer_20260802_v1",
    "c7s": CHECKPOINT_ROOT / "pn_c7s_final_train_baseline_inputs_20260802_v1",
    "c7t": CHECKPOINT_ROOT / "pn_c7t_reestablished_train_baselines_20260802_v1",
    "c7u": CHECKPOINT_ROOT / "pn_c7u_reestablished_baseline_changes_20260802_v1",
}
CHECKPOINT_MANIFEST_SHA256 = {
    "c7q": "4356554c5cb927f2c1daa1c412f2797ad398f31df5c27525a084748da7eda273",
    "c7r": "a663fe73d23db0e2ee856dd041916a22008c020c9b11a7eb59e3dd14acbd19c6",
    "c7s": "9885748add80bfd44cb01947e5ae21b58e08f28396c55cf1fdc7934063309ae1",
    "c7t": "2de28666cdd9e055747b7e2d3df70968bad3a8d8d0613bf3d5c6e05a1d6fc016",
    "c7u": "31bad707918b9b485292e8e70a82962ffba5a2528548f3c258394c4a55adb19f",
}


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
        require(rel not in result, f"duplicate SHA256SUMS member: {rel}")
        result[rel] = digest
    return result


def verify_checkpoint(tag: str) -> list[dict[str, Any]]:
    require(tag in CHECKPOINTS, f"unknown checkpoint tag: {tag}")
    root = CHECKPOINTS[tag]
    manifest = root / "SHA256SUMS"
    require(manifest.is_file(), f"missing checkpoint manifest: {manifest}")
    require(
        sha256(manifest) == CHECKPOINT_MANIFEST_SHA256[tag],
        f"checkpoint manifest identity drift: {tag}",
    )
    evidence = []
    for rel, expected in parse_sha256sums(manifest).items():
        member = root / rel
        require(member.is_file(), f"missing checkpoint member: {member}")
        observed = sha256(member)
        require(observed == expected, f"checkpoint member checksum drift: {member}")
        evidence.append({
            "checkpoint": tag,
            "checkpoint_root": str(root),
            "relative_path": rel,
            "bytes": member.stat().st_size,
            "sha256": observed,
        })
    return evidence


def verify_all_checkpoints() -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for tag in ("c7q", "c7r", "c7s", "c7t", "c7u"):
        evidence.extend(verify_checkpoint(tag))
    return evidence


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    require(not missing, f"{label} missing required columns: {missing}")


def require_train_only(frame: pd.DataFrame, split_column: str, label: str) -> None:
    require_columns(frame, [split_column], label)
    values = set(frame[split_column].dropna().astype(str))
    require(values == {"train"}, f"{label} contains non-train split values: {sorted(values)}")


def require_group_fold_integrity(
    frame: pd.DataFrame,
    *,
    group_column: str = "registry_group_id",
    fold_column: str = "registry_oof_fold",
    folds: int = 5,
) -> None:
    require_columns(frame, [group_column, fold_column], "group-fold table")
    require(not frame[group_column].isna().any(), "null source group")
    require(not frame[fold_column].isna().any(), "null OOF fold")
    per_group = frame.groupby(group_column, sort=False)[fold_column].nunique()
    require(bool((per_group == 1).all()), "source group assigned to multiple OOF folds")
    observed = set(frame[fold_column].astype(int))
    require(observed == set(range(folds)), f"OOF fold coverage drift: {sorted(observed)}")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Sequence[dict[str, Any]], *, allow_empty: bool = False) -> None:
    require(bool(rows) or allow_empty, f"refusing empty TSV: {path}")
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare_staging(output: Path) -> Path:
    require(output.is_absolute(), f"output must be absolute: {output}")
    require(output.parent.is_dir(), f"output parent missing: {output.parent}")
    require(not output.exists(), f"refusing to overwrite completed output: {output}")
    staging = output.with_name(f".{output.name}.staging")
    require(not staging.exists(), f"refusing to overwrite preserved staging: {staging}")
    staging.mkdir()
    return staging


def artifact_rows(root: Path, *, exclude: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = exclude or set()
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in excluded:
            rows.append({
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    return rows


def seal_checkpoint(staging: Path, output: Path, marker: str) -> str:
    required = {
        "README.md", "RUN_CONTRACT.txt", "summary.json", "artifact_manifest.tsv",
        "source_evidence_manifest.tsv",
    }
    require(required.issubset({path.name for path in staging.iterdir()}), "checkpoint contract incomplete")
    (staging / "COMPLETE").write_text(marker + "\n")
    files = sorted(path for path in staging.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (staging / "SHA256SUMS").write_text(
        "\n".join(f"{sha256(path)}  ./{path.relative_to(staging).as_posix()}" for path in files) + "\n"
    )
    manifest_sha = sha256(staging / "SHA256SUMS")
    os.rename(staging, output)
    for path in sorted(output.rglob("*"), reverse=True):
        path.chmod(0o444 if path.is_file() else 0o555)
    output.chmod(0o555)
    return manifest_sha


def effective_events(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=np.float64)
    total = float(values.sum())
    sum2 = float(np.square(values).sum())
    return total * total / sum2 if sum2 > 0.0 else 0.0


def asimov_za(signal: float, background: float) -> float:
    if signal <= 0.0 or background <= 0.0:
        return 0.0
    return math.sqrt(max(0.0, 2.0 * ((signal + background) * math.log1p(signal / background) - signal)))


def asimov_za_with_uncertainty(signal: float, background: float, sigma_background: float) -> float:
    """Cowan Asimov significance with stable extended-precision arithmetic.

    The frozen multijet uncertainty can be orders of magnitude larger than the
    signal.  Evaluating the two logarithmic terms as binary64 scalars then
    subtracting them produced spurious exact zeros in the c7v scan.  NumPy's
    long-double evaluation retains the same formula and nuisance contract while
    avoiding that cancellation on the supported LPC platform.
    """

    if signal <= 0.0 or background <= 0.0:
        return 0.0
    if sigma_background <= 0.0:
        return asimov_za(signal, background)
    s = np.longdouble(signal)
    b = np.longdouble(background)
    variance = np.longdouble(sigma_background) ** 2
    first = (s + b) * np.log(((s + b) * (b + variance)) / (b * b + (s + b) * variance))
    second = (b * b / variance) * np.log1p(variance * s / (b * (b + variance)))
    q0 = np.longdouble(2.0) * (first - second)
    require(bool(np.isfinite(q0)), "nonfinite Asimov background-uncertainty calculation")
    # Extremely small q0 values need more than the platform's 80-bit mantissa.
    # Decimal is used only in this cancellation-dominated tail.
    if q0 < np.longdouble("1e-11"):
        with localcontext() as context:
            context.prec = 60
            ds = Decimal(str(signal))
            db = Decimal(str(background))
            dv = Decimal(str(sigma_background)) ** 2
            dfirst = (ds + db) * (((ds + db) * (db + dv)) / (db * db + (ds + db) * dv)).ln()
            dsecond = (db * db / dv) * (Decimal(1) + dv * ds / (db * (db + dv))).ln()
            dq0 = Decimal(2) * (dfirst - dsecond)
            require(dq0 >= 0, f"negative high-precision Asimov test statistic: {dq0}")
            return float(dq0.sqrt())
    return float(np.sqrt(max(np.longdouble(0.0), q0)))


def source_member_bootstrap_draws(
    members: pd.DataFrame,
    *,
    replicates: int = 1000,
    seed: int = 20260802,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build the exact c7t stratified source-member bootstrap draw stream.

    Columns in the returned matrix follow the input member-table row order.
    Each entry is a source-member multiplicity.  Event rows are never sampled
    independently.  The RNG loop deliberately matches c7t: replica outer loop,
    insertion-ordered ``(sample_class, stratum)`` inner loop, and one continuous
    NumPy generator seeded once.
    """

    require(replicates > 0, "bootstrap replicate count must be positive")
    required = ["member_index", "sample_class", "stratum", "transport_id"]
    require_columns(members, required, "source-member bootstrap registry")
    require(not members[required].isna().any().any(), "null source-member bootstrap field")
    require(not members["member_index"].duplicated().any(), "duplicate bootstrap member index")
    require(not members["transport_id"].duplicated().any(), "duplicate bootstrap transport ID")

    strata: dict[tuple[str, str], list[int]] = {}
    for position, member in enumerate(members.itertuples(index=False)):
        key = (str(member.sample_class), str(member.stratum))
        strata.setdefault(key, []).append(position)
    require({key[0] for key in strata} == {"signal", "background"}, "bootstrap class contract drift")

    rng = np.random.default_rng(seed)
    draws = np.zeros((replicates, len(members)), dtype=np.int16)
    registry_rows: list[dict[str, Any]] = []
    sample_class = members["sample_class"].astype(str).to_numpy()
    member_ids = members["member_index"].astype(int).to_numpy()
    for replica in range(replicates):
        for positions in strata.values():
            selected_local = rng.integers(0, len(positions), size=len(positions))
            selected_positions = np.asarray(positions, dtype=int)[selected_local]
            np.add.at(draws[replica], selected_positions, 1)
        multiplicities = draws[replica]
        signal_counts = multiplicities[sample_class == "signal"].astype(np.float64)
        background_counts = multiplicities[sample_class == "background"].astype(np.float64)
        signal_unique = int(np.count_nonzero(signal_counts))
        background_unique = int(np.count_nonzero(background_counts))
        stratum_counts = {
            f"{klass}:{stratum}": {
                "drawn": int(multiplicities[positions].sum()),
                "unique": int(np.count_nonzero(multiplicities[positions])),
            }
            for (klass, stratum), positions in strata.items()
        }
        nonzero = np.flatnonzero(multiplicities)
        selected = {str(member_ids[pos]): int(multiplicities[pos]) for pos in nonzero}

        def source_neff(counts: np.ndarray) -> float:
            sum2 = float(np.square(counts).sum())
            return float(counts.sum() ** 2 / sum2) if sum2 > 0.0 else 0.0

        valid = signal_unique > 0 and background_unique > 0
        registry_rows.append({
            "replica_id": replica,
            "random_seed": seed,
            "rng_contract": "numpy_default_rng_continuous_stream_replica_outer_stratum_inner",
            "selected_source_member_multiplicities_json": json.dumps(selected, sort_keys=True, separators=(",", ":")),
            "selected_source_members": len(selected),
            "number_signal_sources": signal_unique,
            "number_background_sources": background_unique,
            "process_stratum_counts_json": json.dumps(stratum_counts, sort_keys=True, separators=(",", ":")),
            "effective_signal_source_statistics": source_neff(signal_counts),
            "effective_background_source_statistics": source_neff(background_counts),
            "draw_sha256": hashlib.sha256(multiplicities.tobytes()).hexdigest(),
            "validity_status": "valid" if valid else "invalid",
            "failure_reason": "" if valid else "missing resampled signal or background source",
        })
    validate_source_member_bootstrap_draws(members, draws)
    return draws, registry_rows


def validate_source_member_bootstrap_draws(members: pd.DataFrame, draws: np.ndarray) -> None:
    """Fail closed if a draw matrix violates the frozen stratified contract."""

    require(draws.ndim == 2 and draws.shape[1] == len(members), "bootstrap draw shape mismatch")
    require(np.issubdtype(draws.dtype, np.integer), "bootstrap multiplicities must be integers")
    require(bool((draws >= 0).all()), "negative bootstrap multiplicity")
    strata: dict[tuple[str, str], list[int]] = {}
    for position, member in enumerate(members.itertuples(index=False)):
        strata.setdefault((str(member.sample_class), str(member.stratum)), []).append(position)
    for key, positions in strata.items():
        observed = draws[:, positions].sum(axis=1)
        require(bool((observed == len(positions)).all()), f"bootstrap stratum-size drift: {key}")


def bootstrap_quantile_summary(
    values: Sequence[float] | np.ndarray,
    *,
    invalid_reasons: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return the frozen asymmetric percentile summary without filling invalids."""

    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    valid = array[finite]
    require(len(valid) > 0, "no valid bootstrap replicas")
    reasons: dict[str, int] = {}
    if invalid_reasons is not None:
        require(len(invalid_reasons) == len(array), "bootstrap invalid-reason alignment drift")
        for is_valid, reason in zip(finite, invalid_reasons):
            if not is_valid:
                key = str(reason) or "nonfinite metric"
                reasons[key] = reasons.get(key, 0) + 1
    return {
        "bootstrap_mean": float(np.mean(valid)),
        "bootstrap_median": float(np.median(valid)),
        "bootstrap_standard_deviation": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
        "bootstrap_p16": float(np.quantile(valid, 0.16)),
        "bootstrap_p84": float(np.quantile(valid, 0.84)),
        "bootstrap_p2p5": float(np.quantile(valid, 0.025)),
        "bootstrap_p97p5": float(np.quantile(valid, 0.975)),
        "valid_replicas": int(finite.sum()),
        "invalid_replicas": int((~finite).sum()),
        "invalid_reason_counts_json": json.dumps(reasons, sort_keys=True, separators=(",", ":")),
    }


def correlated_multijet_profile_za(
    signal: np.ndarray,
    ordinary_background: np.ndarray,
    transferred_qcd: np.ndarray,
    qcd_relative_uncertainty: np.ndarray,
) -> float:
    """Profile a single shared Gaussian QCD nuisance over exclusive categories."""

    s = np.asarray(signal, dtype=np.float64)
    o = np.asarray(ordinary_background, dtype=np.float64)
    q = np.asarray(transferred_qcd, dtype=np.float64)
    u = np.asarray(qcd_relative_uncertainty, dtype=np.float64)
    require(s.shape == o.shape == q.shape == u.shape, "category array shape mismatch")
    require(bool((s >= 0).all() and (o >= 0).all() and (q >= 0).all() and (u >= 0).all()),
            "negative category input")
    observed = s + o + q

    def objective(theta: float) -> float:
        expected = o + q * np.maximum(1.0 + u * theta, 1.0e-12)
        expected = np.maximum(expected, 1.0e-12)
        poisson = np.where(
            observed > 0.0,
            observed * np.log(observed / expected) + expected - observed,
            expected,
        )
        return float(2.0 * poisson.sum() + theta * theta)

    result = minimize_scalar(objective, bounds=(-8.0, 8.0), method="bounded", options={"xatol": 1.0e-12})
    require(result.success and math.isfinite(result.fun), "correlated nuisance profiling failed")
    return math.sqrt(max(0.0, float(result.fun)))


def weighted_efficiency(mask: np.ndarray, population: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(np.asarray(weights)[population].sum())
    require(denominator > 0.0, "nonpositive weighted-efficiency denominator")
    return float(np.asarray(weights)[mask & population].sum() / denominator)


def threshold_at_efficiency(
    scores: np.ndarray,
    signal_mask: np.ndarray,
    weights: np.ndarray,
    target: float,
) -> float:
    require(0.0 < target < 1.0, "target efficiency outside (0,1)")
    signal_scores = np.asarray(scores, dtype=np.float64)[signal_mask]
    signal_weights = np.asarray(weights, dtype=np.float64)[signal_mask]
    require(len(signal_scores) > 0 and bool((signal_weights > 0).all()), "invalid signal threshold population")
    order = np.argsort(-signal_scores, kind="mergesort")
    ordered_scores = signal_scores[order]
    ordered_weights = signal_weights[order]
    boundaries = np.r_[np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]), len(ordered_scores) - 1]
    efficiencies = np.cumsum(ordered_weights)[boundaries] / ordered_weights.sum()
    return float(ordered_scores[boundaries[int(np.argmin(np.abs(efficiencies - target)))]])
