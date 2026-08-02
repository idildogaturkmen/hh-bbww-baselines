#!/usr/bin/env python3
"""Run the frozen train-only cut, BDT, dense-DNN, and LBN-DNN baselines.

Models are evaluated with deterministic five-fold source-group out-of-fold
predictions.  Operating thresholds match the frozen R_HH(125,120)<34 train
signal efficiency.  Validation and test payloads are never read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import stat
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
import torch
import torch.nn as nn
import xgboost
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


REPO = Path("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines")
C7S = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/harvest_checkpoints/"
    "pn_c7s_final_train_baseline_inputs_20260802_v1"
)
C7R = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/harvest_checkpoints/"
    "pn_c7r_threeb_fourb_multijet_transfer_20260802_v1"
)
C7S_MANIFEST_SHA = "9885748add80bfd44cb01947e5ae21b58e08f28396c55cf1fdc7934063309ae1"
C7R_MANIFEST_SHA = "a663fe73d23db0e2ee856dd041916a22008c020c9b11a7eb59e3dd14acbd19c6"
CONFIG = REPO / "configs/baselines/hh4b_expanded_cut_bdt_protocol_v2.json"
CONFIG_SHA = "cc56783a4a5a19619c9dd69c1c82878a47cdf6ee6b3db24c43ca11f3a60fa636"
TRAIN_PATH = C7S / "tables/train_fourb_model_development.parquet"
PROJECTION_PATH = C7S / "tables/train_primary_physical_projection.parquet"
DIRECT_PATH = C7S / "tables/train_direct_qcd_secondary_projection.parquet"

MODEL_ORDER = ("cut", "bdt", "dense_dnn", "lbn_dnn")
FOLD_COUNT = 5
DENSE_SEED = 12345
LBN_SEED = 12345
BOOTSTRAP_SEED = 20260802
BOOTSTRAP_REPLICATES = 1000

SNAPSHOT = {
    "cut": {"asimov_ZA": 0.0334, "signal_over_background": 5.8e-5, "background_effective_events": 17.25},
    "bdt": {"weighted_auc": 0.665, "asimov_ZA": 0.0386, "bootstrap_median_ZA": 0.0391,
            "signal_over_background": 1.0e-4, "background_effective_events": 6.51},
    "dense_dnn": {"weighted_auc": 0.648, "asimov_ZA": 0.0369, "bootstrap_median_ZA": 0.0377},
    "lbn_dnn": {"asimov_ZA": 0.0372, "bootstrap_median_ZA": 0.0382},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_sha_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        digest, rel = line.split(None, 1)
        rel = rel.lstrip("*")
        if rel.startswith("./"):
            rel = rel[2:]
        require(rel not in result, f"duplicate manifest member: {rel}")
        result[rel] = digest
    return result


def verify_checkpoint(path: Path, expected_manifest_sha: str) -> None:
    require(sha256(path / "SHA256SUMS") == expected_manifest_sha, f"checkpoint manifest mismatch: {path}")
    for rel, expected in parse_sha_manifest(path / "SHA256SUMS").items():
        member = path / rel
        require(member.is_file() and sha256(member) == expected, f"checkpoint member mismatch: {member}")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"empty TSV output: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def weighted_efficiency(mask: np.ndarray, population: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(weights[population].sum())
    require(denominator > 0.0, "nonpositive efficiency denominator")
    return float(weights[mask & population].sum() / denominator)


def threshold_at_efficiency(scores: np.ndarray, signal: np.ndarray, weights: np.ndarray, target: float) -> float:
    signal_scores = scores[signal]
    signal_weights = weights[signal]
    order = np.argsort(-signal_scores, kind="mergesort")
    ordered_scores = signal_scores[order]
    ordered_weights = signal_weights[order]
    boundaries = np.r_[np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]), len(ordered_scores) - 1]
    efficiencies = np.cumsum(ordered_weights)[boundaries] / ordered_weights.sum()
    position = int(np.argmin(np.abs(efficiencies - target)))
    return float(ordered_scores[boundaries[position]])


def neff(weights: np.ndarray) -> float:
    total = float(weights.sum())
    sum2 = float(np.square(weights).sum())
    return total * total / sum2 if sum2 > 0.0 else 0.0


def asimov(signal: float, background: float) -> float:
    if signal <= 0.0 or background <= 0.0:
        return 0.0
    return math.sqrt(max(0.0, 2.0 * ((signal + background) * math.log1p(signal / background) - signal)))


def asimov_with_background_uncertainty(signal: float, background: float, sigma_background: float) -> float:
    if signal <= 0.0 or background <= 0.0:
        return 0.0
    if sigma_background <= 0.0:
        return asimov(signal, background)
    variance = sigma_background * sigma_background
    first = (signal + background) * math.log(
        ((signal + background) * (background + variance))
        / (background * background + (signal + background) * variance)
    )
    second = (background * background / variance) * math.log(
        1.0 + variance * signal / (background * (background + variance))
    )
    return math.sqrt(max(0.0, 2.0 * (first - second)))


def pt_eta_phi_m_to_p4(frame: pd.DataFrame) -> np.ndarray:
    jets = []
    for index in range(1, 5):
        pt = frame[f"j{index}_pt"].to_numpy(dtype=np.float64)
        eta = frame[f"j{index}_eta"].to_numpy(dtype=np.float64)
        phi = frame[f"j{index}_phi"].to_numpy(dtype=np.float64)
        mass = np.maximum(frame[f"j{index}_mass"].to_numpy(dtype=np.float64), 0.0)
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        energy = np.sqrt(np.maximum(px * px + py * py + pz * pz + mass * mass, 0.0))
        jets.append(np.stack([energy, px, py, pz], axis=1))
    result = np.stack(jets, axis=1).astype(np.float32) / 100.0
    require(np.isfinite(result).all(), "nonfinite four-vector input")
    return result


class SmallDNN(nn.Module):
    """Frozen dense DNN-v3 architecture."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values).squeeze(-1)


class LorentzCombinationLayer(nn.Module):
    """Frozen lightweight LBN-v3 learned positive combinations."""

    def __init__(self, n_input: int = 4, n_combos: int = 8) -> None:
        super().__init__()
        initial = torch.full((n_combos, n_input), -5.0)
        for index in range(min(n_input, n_combos)):
            initial[index, index] = 0.54
        for combo, pair in enumerate(((0, 1), (2, 3), (0, 2), (1, 3)), start=n_input):
            if combo < n_combos:
                for jet in pair:
                    initial[combo, jet] = 0.54
        self.raw_weights = nn.Parameter(initial)

    def forward(self, p4: torch.Tensor) -> torch.Tensor:
        weights = torch.nn.functional.softplus(self.raw_weights)
        combos = torch.einsum("ki,bic->bkc", weights, p4)
        energy, px, py, pz = (combos[..., i] for i in range(4))
        pt = torch.sqrt(torch.clamp(px * px + py * py, min=1e-8))
        momentum = torch.sqrt(torch.clamp(px * px + py * py + pz * pz, min=1e-8))
        mass = torch.sqrt(torch.clamp(energy * energy - momentum * momentum, min=0.0) + 1e-8)
        eta = 0.5 * torch.log(
            torch.clamp(momentum + pz, min=1e-6) / torch.clamp(momentum - pz, min=1e-6)
        )
        eta = torch.clamp(eta, min=-8.0, max=8.0)
        return torch.stack([energy, px, py, pz, pt, mass, eta], dim=-1).flatten(start_dim=1)


class LBNDNN(nn.Module):
    """Frozen best-sensitivity LBN p4-plus-topology architecture."""

    def __init__(self, aux_dim: int, n_combos: int = 8) -> None:
        super().__init__()
        self.lbn = LorentzCombinationLayer(4, n_combos)
        self.net = nn.Sequential(
            nn.Linear(n_combos * 7 + aux_dim, 128), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, p4: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([self.lbn(p4), aux], dim=1)).squeeze(-1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_torch_binary(
    model: nn.Module,
    tensors: tuple[np.ndarray, ...],
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    epochs: int,
    seed: int,
) -> nn.Module:
    seed_everything(seed)
    normalized_weights = weights.astype(np.float32) / float(weights.sum())
    dataset = TensorDataset(
        *(torch.tensor(values, dtype=torch.float32) for values in tensors),
        torch.tensor(labels.astype(np.float32), dtype=torch.float32),
        torch.tensor(normalized_weights, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, generator=generator, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    model.train()
    for _ in range(epochs):
        for batch in loader:
            *features, batch_labels, batch_weights = batch
            optimizer.zero_grad(set_to_none=True)
            logits = model(*features)
            loss = (loss_function(logits, batch_labels) * batch_weights).sum()
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def torch_predict(model: nn.Module, tensors: tuple[np.ndarray, ...]) -> np.ndarray:
    model.eval()
    dataset = TensorDataset(*(torch.tensor(values, dtype=torch.float32) for values in tensors))
    loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=0)
    outputs: list[np.ndarray] = []
    for batch in loader:
        outputs.append(torch.sigmoid(model(*batch)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def run_bdt(
    training: pd.DataFrame, projection: pd.DataFrame, features: list[str], config: dict[str, Any], models: Path,
    *, smoke: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    train_scores = np.full(len(training), np.nan, dtype=np.float64)
    projection_scores = np.full(len(projection), np.nan, dtype=np.float64)
    records = []
    folds = [0] if smoke else list(range(FOLD_COUNT))
    params = dict(config["models"]["global_mass_aware"]["parameters"])
    params.update(config["estimator_fixed_parameters"])
    params["random_state"] = config["models"]["global_mass_aware"]["random_state"]
    if smoke:
        params["n_estimators"] = 5
    for fold in folds:
        print(f"BDT fold {fold}/{FOLD_COUNT - 1}", flush=True)
        fit = training["registry_oof_fold"].to_numpy() != fold
        held = training["registry_oof_fold"].to_numpy() == fold
        projected = projection["registry_oof_fold"].to_numpy() == fold
        model = XGBClassifier(**params)
        model.fit(
            training.loc[fit, features].to_numpy(dtype=np.float32),
            training.loc[fit, "registry_training_target"].to_numpy(dtype=np.int8),
            sample_weight=training.loc[fit, "development_hierarchical_weight"].to_numpy(dtype=np.float64),
        )
        train_scores[held] = model.predict_proba(training.loc[held, features].to_numpy(dtype=np.float32))[:, 1]
        projection_scores[projected] = model.predict_proba(projection.loc[projected, features].to_numpy(dtype=np.float32))[:, 1]
        model_path = models / f"bdt_fold{fold}.json"
        model.save_model(model_path)
        records.append({"model": "bdt", "fold": fold, "path": model_path.name, "sha256": sha256(model_path)})
    return train_scores, projection_scores, records


def run_dense(
    training: pd.DataFrame, projection: pd.DataFrame, features: list[str], models: Path, *, smoke: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    train_scores = np.full(len(training), np.nan, dtype=np.float64)
    projection_scores = np.full(len(projection), np.nan, dtype=np.float64)
    records = []
    folds = [0] if smoke else list(range(FOLD_COUNT))
    epochs = 2 if smoke else 120
    for fold in folds:
        print(f"dense DNN fold {fold}/{FOLD_COUNT - 1}", flush=True)
        fit = training["registry_oof_fold"].to_numpy() != fold
        held = training["registry_oof_fold"].to_numpy() == fold
        projected = projection["registry_oof_fold"].to_numpy() == fold
        scaler = StandardScaler().fit(training.loc[fit, features].to_numpy(dtype=np.float64))
        x_fit = scaler.transform(training.loc[fit, features]).astype(np.float32)
        x_held = scaler.transform(training.loc[held, features]).astype(np.float32)
        x_projected = scaler.transform(projection.loc[projected, features]).astype(np.float32)
        seed = DENSE_SEED + fold
        seed_everything(seed)
        model = SmallDNN(len(features))
        model = train_torch_binary(
            model, (x_fit,),
            training.loc[fit, "registry_training_target"].to_numpy(),
            training.loc[fit, "development_hierarchical_weight"].to_numpy(),
            epochs=epochs, seed=seed,
        )
        train_scores[held] = torch_predict(model, (x_held,))
        projection_scores[projected] = torch_predict(model, (x_projected,))
        model_path = models / f"dense_dnn_fold{fold}.pt"
        torch.save({
            "model_state_dict": model.state_dict(), "features": features,
            "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
            "fold": fold, "seed": seed, "epochs": epochs,
        }, model_path)
        records.append({"model": "dense_dnn", "fold": fold, "path": model_path.name, "sha256": sha256(model_path)})
    return train_scores, projection_scores, records


def run_lbn(
    training: pd.DataFrame, projection: pd.DataFrame, topology_features: list[str], models: Path, *, smoke: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    train_scores = np.full(len(training), np.nan, dtype=np.float64)
    projection_scores = np.full(len(projection), np.nan, dtype=np.float64)
    records = []
    train_p4 = pt_eta_phi_m_to_p4(training)
    projection_p4 = pt_eta_phi_m_to_p4(projection)
    folds = [0] if smoke else list(range(FOLD_COUNT))
    epochs = 2 if smoke else 140
    for fold in folds:
        print(f"LBN-DNN fold {fold}/{FOLD_COUNT - 1}", flush=True)
        fit = training["registry_oof_fold"].to_numpy() != fold
        held = training["registry_oof_fold"].to_numpy() == fold
        projected = projection["registry_oof_fold"].to_numpy() == fold
        scaler = StandardScaler().fit(training.loc[fit, topology_features].to_numpy(dtype=np.float64))
        aux_fit = scaler.transform(training.loc[fit, topology_features]).astype(np.float32)
        aux_held = scaler.transform(training.loc[held, topology_features]).astype(np.float32)
        aux_projected = scaler.transform(projection.loc[projected, topology_features]).astype(np.float32)
        seed = LBN_SEED + fold
        seed_everything(seed)
        model = LBNDNN(len(topology_features), n_combos=8)
        model = train_torch_binary(
            model, (train_p4[fit], aux_fit),
            training.loc[fit, "registry_training_target"].to_numpy(),
            training.loc[fit, "development_hierarchical_weight"].to_numpy(),
            epochs=epochs, seed=seed,
        )
        train_scores[held] = torch_predict(model, (train_p4[held], aux_held))
        projection_scores[projected] = torch_predict(model, (projection_p4[projected], aux_projected))
        model_path = models / f"lbn_dnn_fold{fold}.pt"
        torch.save({
            "model_state_dict": model.state_dict(), "topology_features": topology_features,
            "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
            "fold": fold, "seed": seed, "epochs": epochs, "n_combos": 8,
            "mode": "lbn_p4_plus_topology",
        }, model_path)
        records.append({"model": "lbn_dnn", "fold": fold, "path": model_path.name, "sha256": sha256(model_path)})
    return train_scores, projection_scores, records


def bootstrap_metrics(
    projection: pd.DataFrame, selected: np.ndarray, members: list[dict[str, str]], *, replicates: int,
) -> dict[str, float]:
    selected_frame = projection.loc[selected].copy()
    weight = "primary_projection_physical_weight_inclusive"
    group_yields = selected_frame.groupby("registry_group_id", sort=False)[weight].sum().to_dict()
    strata: dict[tuple[str, str], list[float]] = {}
    for member in members:
        key = (member["sample_class"], member["stratum"])
        strata.setdefault(key, []).append(float(group_yields.get(member["transport_id"], 0.0)))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    za_values = np.empty(replicates, dtype=np.float64)
    for replica in range(replicates):
        signal = 0.0
        background = 0.0
        for (sample_class, _), values_list in strata.items():
            values = np.asarray(values_list, dtype=np.float64)
            total = float(values[rng.integers(0, len(values), size=len(values))].sum())
            if sample_class == "signal":
                signal += total
            else:
                background += total
        za_values[replica] = asimov(signal, background)
    return {
        "bootstrap_replicates": replicates,
        "bootstrap_mean_ZA": float(za_values.mean()),
        "bootstrap_standard_deviation_ZA": float(za_values.std(ddof=1)),
        "bootstrap_median_ZA": float(np.median(za_values)),
        "bootstrap_p16_ZA": float(np.quantile(za_values, 0.16)),
        "bootstrap_p84_ZA": float(np.quantile(za_values, 0.84)),
    }


def baseline_metrics(
    name: str,
    training: pd.DataFrame,
    projection: pd.DataFrame,
    direct: pd.DataFrame,
    train_scores: np.ndarray,
    projection_scores: np.ndarray,
    selection: np.ndarray,
    direct_selection: np.ndarray,
    threshold: float | str,
    factor_values: list[float],
    factor_stat_relative: float,
    members: list[dict[str, str]],
    *,
    smoke: bool,
) -> dict[str, Any]:
    labels = training["registry_training_target"].to_numpy(dtype=np.int8)
    dev_weights = training["development_hierarchical_weight"].to_numpy(dtype=np.float64)
    auc = float(roc_auc_score(labels, train_scores, sample_weight=dev_weights))
    unweighted_auc = float(roc_auc_score(labels, train_scores))
    phys = projection["primary_projection_physical_weight_inclusive"].to_numpy(dtype=np.float64)
    signal_mask = projection["registry_training_target"].to_numpy(dtype=np.int8) == 1
    background_mask = ~signal_mask
    selected_signal = phys[selection & signal_mask]
    selected_background = phys[selection & background_mask]
    signal_yield = float(selected_signal.sum())
    background_yield = float(selected_background.sum())

    transferred_mask = projection["analysis_population_role"].to_numpy() == "primary_transferred_multijet_template"
    selected_threeb_base = projection.loc[selection & transferred_mask, "run2_candidate_physical_weight"].to_numpy(dtype=np.float64)
    qcd_prediction = float(selected_threeb_base.sum() * factor_values[0])
    alternative_predictions = [float(selected_threeb_base.sum() * factor) for factor in factor_values]
    direct_qcd_mask = direct["population_kind"].to_numpy() == "hard_qcd"
    direct_qcd_weights = direct.loc[direct_selection & direct_qcd_mask, "direct_projection_physical_weight"].to_numpy(dtype=np.float64)
    direct_qcd_truth = float(direct_qcd_weights.sum())
    qcd_nonclosure_relative = (
        abs(qcd_prediction - direct_qcd_truth) / abs(direct_qcd_truth) if direct_qcd_truth else 0.0
    )
    envelope_low = min(alternative_predictions)
    envelope_high = max(alternative_predictions)
    envelope_relative = (
        max(abs(envelope_low - qcd_prediction), abs(envelope_high - qcd_prediction)) / abs(qcd_prediction)
        if qcd_prediction else 0.0
    )
    qcd_systematic_absolute = abs(qcd_prediction) * math.sqrt(
        factor_stat_relative * factor_stat_relative
        + envelope_relative * envelope_relative
        + qcd_nonclosure_relative * qcd_nonclosure_relative
    )
    bootstrap = bootstrap_metrics(
        projection, selection, members,
        replicates=10 if smoke else BOOTSTRAP_REPLICATES,
    )
    return {
        "baseline": name,
        "operating_threshold": threshold,
        "weighted_auc": auc,
        "unweighted_auc": unweighted_auc,
        "selected_signal_rows": int(np.count_nonzero(selection & signal_mask)),
        "selected_background_rows": int(np.count_nonzero(selection & background_mask)),
        "selected_transferred_qcd_rows": int(np.count_nonzero(selection & transferred_mask)),
        "selected_direct_qcd_closure_rows": int(np.count_nonzero(direct_selection & direct_qcd_mask)),
        "signal_yield": signal_yield,
        "background_yield": background_yield,
        "signal_over_background": signal_yield / background_yield if background_yield > 0 else 0.0,
        "signal_over_sqrt_background": signal_yield / math.sqrt(background_yield) if background_yield > 0 else 0.0,
        "asimov_ZA": asimov(signal_yield, background_yield),
        "background_effective_events": neff(selected_background),
        "signal_effective_events": neff(selected_signal),
        "transferred_qcd_prediction": qcd_prediction,
        "direct_qcd_secondary_truth": direct_qcd_truth,
        "score_domain_qcd_relative_nonclosure": qcd_nonclosure_relative,
        "score_domain_factor_envelope_low": envelope_low,
        "score_domain_factor_envelope_high": envelope_high,
        "score_domain_direct_truth_covered_by_factor_envelope": envelope_low <= direct_qcd_truth <= envelope_high,
        "multijet_systematic_absolute": qcd_systematic_absolute,
        "multijet_systematic_relative_to_total_background": qcd_systematic_absolute / background_yield if background_yield else 0.0,
        "systematic_aware_asimov_ZA": asimov_with_background_uncertainty(
            signal_yield, background_yield, qcd_systematic_absolute
        ),
        **bootstrap,
    }


def comparison_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_name = {row["baseline"]: row for row in metrics}
    for baseline, references in SNAPSHOT.items():
        current = by_name[baseline]
        for metric, previous in references.items():
            observed = float(current[metric])
            rows.append({
                "baseline": baseline,
                "metric": metric,
                "earlier_snapshot": previous,
                "current_reestablished": observed,
                "absolute_change": observed - previous,
                "relative_change": (observed - previous) / previous if previous else 0.0,
                "material_absolute_or_relative_change": abs(observed - previous) > 0.002 or abs((observed - previous) / previous) > 0.10,
            })
    return rows


def freeze(staging: Path, output: Path, marker: str) -> str:
    (staging / "COMPLETE").write_text(marker + "\n")
    files = sorted(path for path in staging.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (staging / "SHA256SUMS").write_text(
        "\n".join(f"{sha256(path)}  ./{path.relative_to(staging).as_posix()}" for path in files) + "\n"
    )
    digest = sha256(staging / "SHA256SUMS")
    os.rename(staging, output)
    for path in sorted(output.rglob("*"), reverse=True):
        path.chmod(0o444 if path.is_file() else 0o555)
    output.chmod(0o555)
    return digest


def build(staging: Path, *, smoke: bool) -> dict[str, Any]:
    verify_checkpoint(C7S, C7S_MANIFEST_SHA)
    verify_checkpoint(C7R, C7R_MANIFEST_SHA)
    require(sha256(CONFIG) == CONFIG_SHA, "frozen BDT protocol checksum mismatch")
    config = json.loads(CONFIG.read_text())
    c7s_summary = json.loads((C7S / "summary.json").read_text())
    require(c7s_summary["baseline_development_authorized"] is True, "baseline gate not authorized")
    require(c7s_summary["validation_payload_files_opened"] == 0, "validation was opened")
    require(c7s_summary["test_or_evaluation_payload_files_opened"] == 0, "test was opened")

    training = pq.read_table(TRAIN_PATH).to_pandas()
    projection = pq.read_table(PROJECTION_PATH).to_pandas()
    direct = pq.read_table(DIRECT_PATH).to_pandas()
    require(len(training) == 30705 and len(projection) == 31225 and len(direct) == 30705, "baseline input count drift")
    require(set(training["registry_final_split"]) == {"train"}, "non-train model input")
    require(set(projection["registry_final_split"]) == {"train"}, "non-train projection input")
    features = list(config["models"]["global_mass_aware"]["features"])
    topology_features = list(config["models"]["global_mass_plane_blind"]["features"])
    require(features == c7s_summary["mass_aware_features"], "mass-aware feature-order drift")
    require(topology_features == c7s_summary["mass_plane_blind_features"], "topology feature-order drift")

    models_dir = staging / "models"
    models_dir.mkdir()
    predictions_dir = staging / "predictions"
    predictions_dir.mkdir()
    model_records: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    training_predictions = training[[
        "registry_group_id", "registry_member_index", "registry_oof_fold", "registry_training_target",
        "registry_sample_class", "registry_process_or_mode", "development_hierarchical_weight",
        "run2_candidate_physical_weight", "population_kind", "transport_id", "event",
    ]].copy()
    projection_predictions = projection[[
        "registry_group_id", "registry_member_index", "registry_oof_fold", "registry_training_target",
        "registry_sample_class", "registry_process_or_mode", "analysis_population_role",
        "primary_projection_physical_weight_inclusive", "run2_candidate_physical_weight",
        "population_kind", "transport_id", "event",
    ]].copy()

    label = training["registry_training_target"].to_numpy(dtype=np.int8)
    dev_weight = training["development_hierarchical_weight"].to_numpy(dtype=np.float64)
    signal = label == 1
    cut_train_selection = training["r_hh_125_120"].to_numpy() < 34.0
    cut_projection_selection = projection["r_hh_125_120"].to_numpy() < 34.0
    cut_direct_selection = direct["r_hh_125_120"].to_numpy() < 34.0
    target_efficiency = weighted_efficiency(cut_train_selection, signal, dev_weight)
    factor_rows = read_tsv(C7R / "transfer_factor_registry.tsv")
    factor_values = [
        float(row["transfer_factor"]) for row in factor_rows if row["mhh_category"] == "inclusive_mhh"
    ]
    require(len(factor_values) == 3, "inclusive transfer-factor scheme count drift")
    # Keep nominal first regardless of TSV ordering.
    nominal_row = next(
        row for row in factor_rows
        if row["mhh_category"] == "inclusive_mhh" and row["factor_scheme"] == "cms_cr_nominal"
    )
    nominal_factor = float(nominal_row["transfer_factor"])
    factor_values = [nominal_factor] + [value for value in factor_values if value != nominal_factor]
    factor_stat_relative = float(nominal_row["transfer_factor_relative_statistical_uncertainty"])
    members = read_tsv(C7S / "member_fold_registry.tsv")

    print("cut baseline", flush=True)
    cut_train_score = -training["r_hh_125_120"].to_numpy(dtype=np.float64)
    cut_projection_score = -projection["r_hh_125_120"].to_numpy(dtype=np.float64)
    training_predictions["cut_score"] = cut_train_score
    projection_predictions["cut_score"] = cut_projection_score
    all_metrics.append(baseline_metrics(
        "cut", training, projection, direct, cut_train_score, cut_projection_score,
        cut_projection_selection, cut_direct_selection, "r_hh_125_120 < 34 GeV",
        factor_values, factor_stat_relative, members, smoke=smoke,
    ))

    runners: list[tuple[str, Callable[..., tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]]]] = [
        ("bdt", lambda: run_bdt(training, projection, features, config, models_dir, smoke=smoke)),
        ("dense_dnn", lambda: run_dense(training, projection, features, models_dir, smoke=smoke)),
        ("lbn_dnn", lambda: run_lbn(training, projection, topology_features, models_dir, smoke=smoke)),
    ]
    for name, runner in runners:
        print(f"starting {name}", flush=True)
        train_score, projection_score, records = runner()
        model_records.extend(records)
        if smoke:
            completed = training["registry_oof_fold"].to_numpy() == 0
            completed_projection = projection["registry_oof_fold"].to_numpy() == 0
            require(np.isfinite(train_score[completed]).all(), f"{name} smoke train prediction failure")
            require(np.isfinite(projection_score[completed_projection]).all(), f"{name} smoke projection failure")
            train_score[~completed] = 0.5
            projection_score[~completed_projection] = 0.5
        else:
            require(np.isfinite(train_score).all() and np.isfinite(projection_score).all(), f"{name} OOF prediction failure")
        threshold = threshold_at_efficiency(train_score, signal, dev_weight, target_efficiency)
        selection = projection_score >= threshold
        direct_selection = train_score >= threshold
        training_predictions[f"{name}_score"] = train_score
        projection_predictions[f"{name}_score"] = projection_score
        all_metrics.append(baseline_metrics(
            name, training, projection, direct, train_score, projection_score,
            selection, direct_selection, threshold,
            factor_values, factor_stat_relative, members, smoke=smoke,
        ))
        print(f"completed {name}", flush=True)

    write_tsv(staging / "baseline_metrics.tsv", all_metrics)
    comparisons = comparison_rows(all_metrics)
    write_tsv(staging / "earlier_snapshot_comparison.tsv", comparisons)
    write_tsv(staging / "model_artifact_manifest.tsv", model_records)
    pq.write_table(
        pa.Table.from_pandas(training_predictions, preserve_index=False),
        predictions_dir / "train_fourb_oof_scores.parquet", compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pandas(projection_predictions, preserve_index=False),
        predictions_dir / "train_primary_projection_oof_scores.parquet", compression="zstd",
    )
    artifacts = []
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path.name not in {"SHA256SUMS"}:
            artifacts.append({
                "path": path.relative_to(staging).as_posix(), "bytes": path.stat().st_size,
                "sha256": sha256(path), "role": "baseline_output",
            })
    write_tsv(staging / "artifact_manifest.tsv", artifacts)

    material_changes = [row for row in comparisons if row["material_absolute_or_relative_change"]]
    summary = {
        "schema_version": 1,
        "status": "pn_c7t_smoke_pass" if smoke else "pn_c7t_reestablished_train_baselines_pass",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_checkpoint": str(C7S),
        "model_order": list(MODEL_ORDER),
        "split": "train",
        "evaluation_method": "five_fold_source_group_out_of_fold" if not smoke else "single_fold_two_epoch_smoke",
        "target_weighted_signal_efficiency": target_efficiency,
        "metrics": {row["baseline"]: row for row in all_metrics},
        "earlier_snapshot_material_change_count": len(material_changes),
        "material_change_drivers": [
            "complete_441_source_train_composition_replaces_earlier_population",
            "exact_generator_weight_transport_and_Run2_coefficients_are_now_applied",
            "primary_multijet_yield_uses_threeb_transfer_instead_of_direct_QCD",
            "small_effective_QCD_statistics_produce_large_transfer_nonclosure_systematics",
            "all_model_results_are_source_group_OOF_train_estimates_not_validation_results",
        ],
        "runtime": {
            "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "pyarrow": pa.__version__, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
            "torch": torch.__version__, "device": "cpu", "torch_threads": torch.get_num_threads(),
        },
        "validation_payload_files_opened": 0,
        "test_or_evaluation_payload_files_opened": 0,
        "validation_metrics_reported": False,
        "test_metrics_reported": False,
        "full_run2_prediction": False,
        "next_gate": "freeze train baseline comparison; validation may be opened only by a separately authorized single-pass protocol",
    }
    write_json(staging / "summary.json", summary)
    (staging / "RUN_CONTRACT.txt").write_text(
        "Baselines execute in the frozen order cut, BDT, dense DNN, LBN DNN. The 34-feature "
        "mass-aware set, 30-feature topology set, model architectures, seeds, BDT hyperparameters, "
        "and source-group folds are frozen inputs. Thresholds use only OOF train scores and match "
        "the frozen R_HH<34 weighted train signal efficiency. Primary physical metrics use the "
        "three-b multijet transfer; direct QCD is score-domain closure only. Validation and test "
        "are not opened.\n"
    )
    (staging / "README.md").write_text(
        "# PN-c7t re-established HH4b train baselines\n\n"
        "Train-only group-OOF cut, XGBoost BDT, dense-DNN, and LBN-DNN results on the complete "
        "441-source production. Physical metrics use the PN-c7r lower-b-tag multijet transfer.\n"
    )
    return summary


def run(output: Path, *, smoke: bool) -> dict[str, Any]:
    require(output.is_absolute() and output.parent.is_dir(), "invalid absolute output path")
    require(not output.exists(), f"refusing to overwrite output: {output}")
    staging = output.with_name(f".{output.name}.staging")
    require(not staging.exists(), f"refusing to overwrite preserved staging: {staging}")
    staging.mkdir()
    try:
        summary = build(staging, smoke=smoke)
        marker = "pn_c7t_smoke_pass" if smoke else "pn_c7t_reestablished_train_baselines_pass"
        digest = freeze(staging, output, marker)
        return {"output": str(output), "sha256sums_sha256": digest, **summary}
    except Exception as exc:
        try:
            write_json(staging / "FAILURE.json", {
                "status": "pn_c7t_failed_preserved_staging", "error": str(exc),
                "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "traceback": traceback.format_exc(), "smoke": smoke,
            })
        except Exception:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    print(json.dumps(run(args.output, smoke=args.smoke), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
