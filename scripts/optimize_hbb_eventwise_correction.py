"""
Optimize H→bb event-by-event b-jet correction strategies.

This is a rigorous follow-up to the simple cap scan:
  - no fixed 1.25 assumption
  - continuous smooth-cap optimization
  - optional gain parameter
  - pT-dependent event-level mass-scale correction
  - calibration/evaluation split
  - CMS-like AK4 baseline pT/eta philosophy
  - exact and scaled-to-125 metrics

Input expected from existing SPA-Net/Hbb HDF5:
  INPUTS/Jets/log_pt
  INPUTS/Jets/log_mass
  INPUTS/Jets/log_corrected_pt
  INPUTS/Jets/log_corrected_mass
  INPUTS/Jets/response_corr
  INPUTS/Jets/eta
  INPUTS/Jets/sin_phi
  INPUTS/Jets/cos_phi
  TARGETS/h/b1
  TARGETS/h/b2

Outputs:
  outputs/hbb_eventwise_correction_optimization/optimized_strategy_summary.csv
  outputs/hbb_eventwise_correction_optimization/eval_event_mbb_by_strategy.csv
  outputs/hbb_eventwise_correction_optimization/pt_scale_curve_*.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

try:
    from scipy.optimize import differential_evolution, minimize_scalar
    from scipy.interpolate import PchipInterpolator
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


M_HIGGS = 125.0


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def smooth_min(x: np.ndarray, cap: float, beta: float = 40.0) -> np.ndarray:
    """
    Differentiable approximation to min(x, cap).
    For x << cap, returns approximately x.
    For x >> cap, returns approximately cap.
    """
    z = beta * (cap - x)
    return cap - np.logaddexp(0.0, z) / beta


def width68(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    q16, q84 = np.percentile(values, [16, 84])
    return 0.5 * (q84 - q16)


def truncated_mean(values: np.ndarray, center_mode: str = "median") -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan

    if center_mode == "mean":
        c0 = np.mean(values)
    else:
        c0 = np.median(values)

    w = width68(values)
    if not np.isfinite(w) or w <= 0:
        return float(np.mean(values))

    keep = np.abs(values - c0) <= w
    if np.sum(keep) < max(10, 0.2 * len(values)):
        return float(np.mean(values))
    return float(np.mean(values[keep]))


def summarize(values: np.ndarray, center: str = "median") -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "n": 0,
            "median": np.nan,
            "mean": np.nan,
            "truncated_mean": np.nan,
            "center_value": np.nan,
            "width68_gev": np.nan,
            "width68_percent_of_center": np.nan,
            "scale_to_125": np.nan,
            "scaled_width68_gev": np.nan,
            "raw_frac_90_140": np.nan,
            "raw_tail_low_90": np.nan,
            "raw_tail_high_140": np.nan,
            "scaled_frac_90_140": np.nan,
            "scaled_tail_low_90": np.nan,
            "scaled_tail_high_140": np.nan,
        }

    med = float(np.median(values))
    mean = float(np.mean(values))
    tmean = truncated_mean(values)

    if center == "mean":
        c = mean
    elif center == "truncated_mean":
        c = tmean
    else:
        c = med

    w68 = float(width68(values))
    scale = M_HIGGS / c if c > 0 else np.nan
    scaled = values * scale if np.isfinite(scale) else values * np.nan

    return {
        "n": int(len(values)),
        "median": med,
        "mean": mean,
        "truncated_mean": tmean,
        "center_value": c,
        "width68_gev": w68,
        "width68_percent_of_center": 100.0 * w68 / c if c > 0 else np.nan,
        "scale_to_125": scale,
        "scaled_width68_gev": w68 * scale if np.isfinite(scale) else np.nan,
        "raw_frac_90_140": float(np.mean((values >= 90.0) & (values <= 140.0))),
        "raw_tail_low_90": float(np.mean(values < 90.0)),
        "raw_tail_high_140": float(np.mean(values > 140.0)),
        "scaled_frac_90_140": float(np.mean((scaled >= 90.0) & (scaled <= 140.0))),
        "scaled_tail_low_90": float(np.mean(scaled < 90.0)),
        "scaled_tail_high_140": float(np.mean(scaled > 140.0)),
    }


def pair_mass(pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2) -> np.ndarray:
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    e1 = np.sqrt(np.maximum((pt1 * np.cosh(eta1)) ** 2 + mass1**2, 0.0))

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    e2 = np.sqrt(np.maximum((pt2 * np.cosh(eta2)) ** 2 + mass2**2, 0.0))

    e = e1 + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    return np.sqrt(np.maximum(e**2 - px**2 - py**2 - pz**2, 0.0))


@dataclass
class HbbArrays:
    pt: np.ndarray
    mass: np.ndarray
    eta: np.ndarray
    phi: np.ndarray
    response_corr: np.ndarray
    original_corr_pt: np.ndarray
    original_corr_mass: np.ndarray
    b1: np.ndarray
    b2: np.ndarray
    idx: np.ndarray
    min_bjet_pt: np.ndarray


def load_hbb_arrays(path: Path) -> HbbArrays:
    with h5py.File(path, "r") as f:
        log_pt = f["INPUTS/Jets/log_pt"][:]
        log_mass = f["INPUTS/Jets/log_mass"][:]
        log_corr_pt = f["INPUTS/Jets/log_corrected_pt"][:]
        log_corr_mass = f["INPUTS/Jets/log_corrected_mass"][:]
        response_corr = f["INPUTS/Jets/response_corr"][:]
        eta = f["INPUTS/Jets/eta"][:]
        sin_phi = f["INPUTS/Jets/sin_phi"][:]
        cos_phi = f["INPUTS/Jets/cos_phi"][:]
        b1 = f["TARGETS/h/b1"][:].astype(int)
        b2 = f["TARGETS/h/b2"][:].astype(int)

    pt = np.exp(log_pt)
    mass = np.exp(log_mass)
    phi = np.arctan2(sin_phi, cos_phi)
    original_corr_pt = np.exp(log_corr_pt)
    original_corr_mass = np.exp(log_corr_mass)
    idx = np.arange(len(b1))
    min_bjet_pt = np.minimum(pt[idx, b1], pt[idx, b2])

    return HbbArrays(
        pt=pt,
        mass=mass,
        eta=eta,
        phi=phi,
        response_corr=response_corr,
        original_corr_pt=original_corr_pt,
        original_corr_mass=original_corr_mass,
        b1=b1,
        b2=b2,
        idx=idx,
        min_bjet_pt=min_bjet_pt,
    )


def mass_uncorrected(a: HbbArrays) -> np.ndarray:
    i = a.idx
    return pair_mass(
        a.pt[i, a.b1], a.eta[i, a.b1], a.phi[i, a.b1], a.mass[i, a.b1],
        a.pt[i, a.b2], a.eta[i, a.b2], a.phi[i, a.b2], a.mass[i, a.b2],
    )


def mass_original_dnn(a: HbbArrays) -> np.ndarray:
    i = a.idx
    return pair_mass(
        a.original_corr_pt[i, a.b1], a.eta[i, a.b1], a.phi[i, a.b1], a.original_corr_mass[i, a.b1],
        a.original_corr_pt[i, a.b2], a.eta[i, a.b2], a.phi[i, a.b2], a.original_corr_mass[i, a.b2],
    )


def mass_smooth_capped_dnn(
    a: HbbArrays,
    cap: float,
    gain: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Event-by-event correction:
      raw DNN response_corr is smoothly capped at cap,
      then the correction displacement from 1 is scaled by gain.

    corr_final = 1 + gain * (smooth_min(response_corr, cap) - 1)

    Returns:
      mbb, event_mean_correction
    """
    r = smooth_min(a.response_corr, cap=cap, beta=beta)
    corr = 1.0 + gain * (r - 1.0)
    corr = np.clip(corr, 0.2, 3.0)

    pt_corr = a.pt * corr
    mass_corr = a.mass * corr

    i = a.idx
    mbb = pair_mass(
        pt_corr[i, a.b1], a.eta[i, a.b1], a.phi[i, a.b1], mass_corr[i, a.b1],
        pt_corr[i, a.b2], a.eta[i, a.b2], a.phi[i, a.b2], mass_corr[i, a.b2],
    )

    event_corr = 0.5 * (corr[i, a.b1] + corr[i, a.b2])
    return mbb, event_corr


def make_pt_scale_curve(
    min_pt: np.ndarray,
    mbb: np.ndarray,
    n_bins: int = 8,
    shrink: float = 0.8,
    min_scale: float = 0.75,
    max_scale: float = 1.35,
) -> tuple[callable, pd.DataFrame]:
    """
    Build a smooth event-by-event mass scale factor versus softer-b-jet pT.

    scale(min_pt) = 125 / median(mbb in pT bin), smoothed/interpolated.

    shrink < 1 pulls the correction toward 1 to avoid overfitting:
      scale_final = 1 + shrink * (scale_raw - 1)
    """
    min_pt = np.asarray(min_pt, dtype=float)
    mbb = np.asarray(mbb, dtype=float)
    mask = np.isfinite(min_pt) & np.isfinite(mbb) & (mbb > 0)

    x = min_pt[mask]
    y = mbb[mask]

    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(x, qs))
    if len(edges) < 4:
        edges = np.linspace(np.nanmin(x), np.nanmax(x), min(n_bins + 1, 5))

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        keep = (x >= lo) & (x <= hi if hi == edges[-1] else x < hi)
        if np.sum(keep) < 20:
            continue
        med = float(np.median(y[keep]))
        raw_scale = M_HIGGS / med if med > 0 else 1.0
        scale = 1.0 + shrink * (raw_scale - 1.0)
        scale = float(np.clip(scale, min_scale, max_scale))
        rows.append(
            {
                "pt_low": float(lo),
                "pt_high": float(hi),
                "pt_center": float(np.median(x[keep])),
                "n": int(np.sum(keep)),
                "median_mbb": med,
                "raw_scale": raw_scale,
                "shrunk_clipped_scale": scale,
            }
        )

    curve_df = pd.DataFrame(rows)
    if len(curve_df) < 2:
        def f_const(z):
            return np.ones_like(np.asarray(z, dtype=float))
        return f_const, curve_df

    xp = curve_df["pt_center"].to_numpy()
    fp = curve_df["shrunk_clipped_scale"].to_numpy()

    if HAS_SCIPY and len(xp) >= 3:
        interp = PchipInterpolator(xp, fp, extrapolate=True)

        def f(z):
            out = interp(np.asarray(z, dtype=float))
            return np.clip(out, min_scale, max_scale)

    else:
        def f(z):
            return np.interp(np.asarray(z, dtype=float), xp, fp, left=fp[0], right=fp[-1])

    return f, curve_df


def smooth_objective(values: np.ndarray, tau: float = 4.0) -> float:
    """
    Smooth objective for optimizer.
    Lower is better.

    It uses scaled-to-125 masses, relative width, soft high-tail penalty,
    and soft 90-140 window penalty.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 50:
        return 1e9

    med = np.median(values)
    scaled = values * (M_HIGGS / med)

    rel_width = width68(values) / med
    soft_window = sigmoid((scaled - 90.0) / tau) * sigmoid((140.0 - scaled) / tau)
    soft_high_tail = sigmoid((scaled - 140.0) / tau)

    loss = rel_width + 0.40 * np.mean(soft_high_tail) + 0.25 * (1.0 - np.mean(soft_window))
    return float(loss)


def rank_proxy(metrics: dict, efficiency: float) -> float:
    """
    Signal-only proxy ranking. Higher is better.
    This is not the final significance; it is only a signal-shape proxy
    until background samples are processed.
    """
    rel_width = metrics["width68_percent_of_center"] / 100.0
    useful_eff = efficiency * metrics["scaled_frac_90_140"]
    tail_penalty = 1.0 + metrics["scaled_tail_high_140"] + 0.5 * metrics["scaled_tail_low_90"]
    return float(useful_eff / max(rel_width * tail_penalty, 1e-9))


def optimize_smooth_cap(a: HbbArrays, calib_idx: np.ndarray, beta: float, optimize_gain: bool) -> dict:
    def objective_1d(cap):
        mbb, _ = mass_smooth_capped_dnn(a, cap=float(cap), gain=1.0, beta=beta)
        return smooth_objective(mbb[calib_idx])

    def objective_2d(params):
        cap, gain = params
        mbb, _ = mass_smooth_capped_dnn(a, cap=float(cap), gain=float(gain), beta=beta)
        return smooth_objective(mbb[calib_idx])

    if optimize_gain:
        if HAS_SCIPY:
            res = differential_evolution(
                objective_2d,
                bounds=[(1.00, 2.50), (0.40, 1.30)],
                tol=1e-4,
                polish=True,
                seed=123,
            )
            cap, gain = map(float, res.x)
            loss = float(res.fun)
        else:
            rng = np.random.default_rng(123)
            best = (1e99, 1.25, 1.0)
            for _ in range(400):
                cap = rng.uniform(1.00, 2.50)
                gain = rng.uniform(0.40, 1.30)
                loss = objective_2d((cap, gain))
                if loss < best[0]:
                    best = (loss, cap, gain)
            loss, cap, gain = best
    else:
        if HAS_SCIPY:
            res = minimize_scalar(objective_1d, bounds=(1.00, 2.50), method="bounded", options={"xatol": 1e-4})
            cap, gain, loss = float(res.x), 1.0, float(res.fun)
        else:
            caps = np.linspace(1.00, 2.50, 301)
            losses = np.array([objective_1d(c) for c in caps])
            j = int(np.nanargmin(losses))
            cap, gain, loss = float(caps[j]), 1.0, float(losses[j])

    return {"cap": cap, "gain": gain, "calib_loss": loss}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/spanet_hbb/spanet_hbb_test.h5")
    parser.add_argument("--outdir", default="outputs/hbb_eventwise_correction_optimization")
    parser.add_argument("--center", choices=["median", "mean", "truncated_mean"], default="median")
    parser.add_argument("--calib-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--smooth-beta", type=float, default=40.0)
    parser.add_argument("--pt-scale-bins", type=int, default=8)
    parser.add_argument("--pt-scale-shrink", type=float, default=0.8)
    parser.add_argument("--min-cut", type=float, default=0.0)
    parser.add_argument("--max-cut", type=float, default=60.0)
    parser.add_argument("--cut-step", type=float, default=1.0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    a = load_hbb_arrays(Path(args.file))
    n = len(a.idx)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_calib = int(args.calib_fraction * n)
    calib_idx = np.sort(perm[:n_calib])
    eval_idx = np.sort(perm[n_calib:])

    print(f"[opt] events: {n}")
    print(f"[opt] calibration events: {len(calib_idx)}")
    print(f"[opt] evaluation events: {len(eval_idx)}")
    print(f"[opt] scipy available: {HAS_SCIPY}")

    strategies = {}

    # Baselines.
    strategies["uncorrected"] = {
        "mbb": mass_uncorrected(a),
        "description": "No DNN correction",
        "cap": np.nan,
        "gain": np.nan,
        "calib_loss": np.nan,
    }

    strategies["dnn_original"] = {
        "mbb": mass_original_dnn(a),
        "description": "Original stored DNN correction",
        "cap": np.nan,
        "gain": np.nan,
        "calib_loss": smooth_objective(mass_original_dnn(a)[calib_idx]),
    }

    # Smooth cap only.
    opt_cap = optimize_smooth_cap(a, calib_idx, beta=args.smooth_beta, optimize_gain=False)
    mbb_cap, event_corr_cap = mass_smooth_capped_dnn(
        a, cap=opt_cap["cap"], gain=opt_cap["gain"], beta=args.smooth_beta
    )
    strategies["dnn_smooth_cap_opt"] = {
        "mbb": mbb_cap,
        "event_corr": event_corr_cap,
        "description": "Smooth cap optimized on calibration split",
        **opt_cap,
    }

    # Smooth cap + gain.
    opt_cap_gain = optimize_smooth_cap(a, calib_idx, beta=args.smooth_beta, optimize_gain=True)
    mbb_cap_gain, event_corr_cap_gain = mass_smooth_capped_dnn(
        a, cap=opt_cap_gain["cap"], gain=opt_cap_gain["gain"], beta=args.smooth_beta
    )
    strategies["dnn_smooth_cap_gain_opt"] = {
        "mbb": mbb_cap_gain,
        "event_corr": event_corr_cap_gain,
        "description": "Smooth cap and gain optimized on calibration split",
        **opt_cap_gain,
    }

    # pT-dependent event scale corrections.
    for base_name in ["dnn_original", "dnn_smooth_cap_opt", "dnn_smooth_cap_gain_opt"]:
        base_mbb = strategies[base_name]["mbb"]
        scale_fun, curve_df = make_pt_scale_curve(
            a.min_bjet_pt[calib_idx],
            base_mbb[calib_idx],
            n_bins=args.pt_scale_bins,
            shrink=args.pt_scale_shrink,
        )
        curve_path = outdir / f"pt_scale_curve_{base_name}.csv"
        curve_df.to_csv(curve_path, index=False)

        event_scale = scale_fun(a.min_bjet_pt)
        corrected_mbb = base_mbb * event_scale

        strategies[f"{base_name}_ptscale"] = {
            "mbb": corrected_mbb,
            "event_scale": event_scale,
            "description": f"{base_name} plus smooth event-by-event scale vs softer-bjet pT",
            "cap": strategies[base_name].get("cap", np.nan),
            "gain": strategies[base_name].get("gain", np.nan),
            "calib_loss": smooth_objective(corrected_mbb[calib_idx]),
        }

    # Save event-level evaluation table.
    event_df = pd.DataFrame(
        {
            "event_local_index": a.idx,
            "split": np.where(np.isin(a.idx, calib_idx), "calib", "eval"),
            "min_bjet_pt": a.min_bjet_pt,
        }
    )

    for name, d in strategies.items():
        event_df[f"mbb_{name}"] = d["mbb"]
        if "event_corr" in d:
            event_df[f"event_corr_{name}"] = d["event_corr"]
        if "event_scale" in d:
            event_df[f"event_scale_{name}"] = d["event_scale"]

    event_df.to_csv(outdir / "eval_event_mbb_by_strategy.csv", index=False)

    # Summaries for dense min-bjet-pT cuts.
    cut_values = np.arange(args.min_cut, args.max_cut + 0.5 * args.cut_step, args.cut_step)

    rows = []
    for name, d in strategies.items():
        values_all = d["mbb"]

        for split_name, split_idx in [("calib", calib_idx), ("eval", eval_idx)]:
            for cut in cut_values:
                keep = split_idx[a.min_bjet_pt[split_idx] >= cut]
                efficiency = len(keep) / max(len(split_idx), 1)
                metrics = summarize(values_all[keep], center=args.center)
                proxy = rank_proxy(metrics, efficiency) if metrics["n"] > 0 else np.nan

                rows.append(
                    {
                        "strategy": name,
                        "description": d["description"],
                        "split": split_name,
                        "min_bjet_pt_cut": float(cut),
                        "efficiency_within_split": efficiency,
                        "optimization_cap": d.get("cap", np.nan),
                        "optimization_gain": d.get("gain", np.nan),
                        "calib_loss": d.get("calib_loss", np.nan),
                        "signal_proxy_score_higher_is_better": proxy,
                        **metrics,
                    }
                )

    summary = pd.DataFrame(rows)
    summary_path = outdir / "optimized_strategy_summary.csv"
    summary.to_csv(summary_path, index=False)

    eval_summary = summary[summary["split"] == "eval"].copy()
    best = eval_summary.sort_values("signal_proxy_score_higher_is_better", ascending=False).head(20)

    print("\n=== Top 20 evaluation strategies by signal-only proxy ===")
    cols = [
        "strategy",
        "min_bjet_pt_cut",
        "efficiency_within_split",
        "optimization_cap",
        "optimization_gain",
        "center_value",
        "width68_percent_of_center",
        "scaled_width68_gev",
        "scaled_frac_90_140",
        "scaled_tail_high_140",
        "signal_proxy_score_higher_is_better",
    ]
    print(best[cols].to_string(index=False))

    print("\nWrote:")
    print(f"  {summary_path}")
    print(f"  {outdir / 'eval_event_mbb_by_strategy.csv'}")
    print(f"  {outdir / 'pt_scale_curve_*.csv'}")


if __name__ == "__main__":
    main()