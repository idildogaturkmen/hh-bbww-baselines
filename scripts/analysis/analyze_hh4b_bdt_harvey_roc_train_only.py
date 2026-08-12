#!/usr/bin/env python3
"""Train-only tail/support audit of immutable HH4b nested outer-OOF scores."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.plotting.hh4b_cms_style import add_delphes_header, apply_cms_style, save_png_pdf

REPO = Path(__file__).resolve().parents[2]
SOURCE = Path("/uscms_data/d3/iturkmen/hh4b_delphes/bdt_apples_to_apples/production_audit/cluster_3799191")
OUT = REPO / "artifacts/hh4b_bdt_apples_to_apples/harvey_roc_train_only_v1"
VARIANTS = ("global_mass_plane_blind", "global_mass_aware")
LABELS = {"global_mass_plane_blind": "Global mass-plane-blind BDT", "global_mass_aware": "Global mass-aware BDT"}
COLORS = {"global_mass_plane_blind": "#0072B2", "global_mass_aware": "#D55E00"}
EXPECTED = {"global_mass_plane_blind": "dfee50e0a83ae3f0f3382f2e9e8205b39c47749c23b3e694c0fdce49e490c14a", "global_mass_aware": "1c17362c5cae14ab3ac980033bb3052803bc81f9b121b183390594dab0988725"}
TARGETS = (0.60, 0.585957, 0.50, 0.40, 0.25, 0.10)
TAIL_TARGETS = (1e-1, 1e-2, 1e-3, 1e-4)


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def neff(sumw: float, sumw2: float) -> float:
    return float(sumw * sumw / sumw2) if sumw2 > 0 else 0.0


def za(s: float, b: float) -> float:
    if s <= 0 or b <= 0:
        return float("nan")
    return math.sqrt(max(0.0, 2.0 * ((s + b) * math.log1p(s / b) - s)))


def threshold_for_signal(frame: pd.DataFrame, target: float) -> float:
    scores = np.sort(frame.loc[frame.class_label.eq(1), "score"].to_numpy(float))[::-1]
    index = min(len(scores) - 1, max(0, int(round(target * len(scores))) - 1))
    return float(scores[index])


def threshold_for_background(frame: pd.DataFrame, target: float) -> float:
    scores = np.sort(frame.loc[frame.class_label.eq(0), "score"].to_numpy(float))[::-1]
    index = min(len(scores) - 1, max(0, int(round(target * len(scores))) - 1))
    return float(scores[index])


def pthat(source: str) -> str:
    match = re.search(r"pthat(\d+to\d+)", source.lower())
    return match.group(1) if match else "unavailable"


def metrics(frame: pd.DataFrame, threshold: float) -> tuple[dict, pd.DataFrame]:
    selected = frame.score.to_numpy(float) >= threshold
    signal = frame.class_label.to_numpy(int) == 1
    background = ~signal
    qcd = frame.process_or_mode.astype(str).to_numpy() == "qcd_hardqcd"
    weight = frame.resolved_selection_contribution_weight.to_numpy(float)
    broad_b = float(weight[background].sum())
    masks = {"signal": selected & signal, "background": selected & background, "qcd": selected & qcd}
    sums = {name: float(weight[mask].sum()) for name, mask in masks.items()}
    sums2 = {name: float(np.square(weight[mask]).sum()) for name, mask in masks.items()}
    nonqcd = sums["background"] - sums["qcd"]
    qmask = masks["qcd"]
    qweights = weight[qmask]
    qsource = frame.loc[qmask, "source_uid"].astype(str)
    source_sums = pd.DataFrame({"source_uid": qsource.to_numpy(), "weight": qweights}).groupby("source_uid", as_index=False).weight.sum() if qmask.any() else pd.DataFrame(columns=["source_uid", "weight"])
    max_source = float(source_sums.weight.abs().max()) if len(source_sums) else 0.0
    qden = abs(sums["qcd"])
    row = {
        "score_threshold": threshold,
        "epsilon_S": float(masks["signal"].sum() / signal.sum()),
        "epsilon_B": float(masks["background"].sum() / background.sum()),
        "rejection": float(background.sum() / masks["background"].sum()) if masks["background"].sum() else float("inf"),
        "raw_signal_rows": int(masks["signal"].sum()), "raw_background_rows": int(masks["background"].sum()),
        "physical_S": sums["signal"], "physical_B": sums["background"], "physical_QCD_B": sums["qcd"], "physical_nonQCD_B": nonqcd,
        "physical_background_efficiency": sums["background"] / broad_b,
        "S_over_B": sums["signal"] / sums["background"] if sums["background"] != 0 else float("nan"),
        "S_over_sqrt_B_diagnostic": sums["signal"] / math.sqrt(sums["background"]) if sums["background"] > 0 else float("nan"),
        "ZA_stat_only": za(sums["signal"], sums["background"]),
        "signal_sumw": sums["signal"], "signal_sumw2": sums2["signal"], "signal_Neff": neff(sums["signal"], sums2["signal"]),
        "background_sumw": sums["background"], "background_sumw2": sums2["background"], "background_Neff": neff(sums["background"], sums2["background"]),
        "qcd_raw_rows": int(qmask.sum()), "qcd_sumw": sums["qcd"], "qcd_sumw2": sums2["qcd"], "qcd_Neff": neff(sums["qcd"], sums2["qcd"]),
        "qcd_largest_abs_event_weight": float(np.abs(qweights).max()) if len(qweights) else 0.0,
        "qcd_largest_event_fraction_of_yield": float(np.abs(qweights).max() / qden) if len(qweights) and qden else float("nan"),
        "qcd_largest_source_fraction_of_yield": max_source / qden if qden else float("nan"),
        "QCD_Neff_below_100": neff(sums["qcd"], sums2["qcd"]) < 100,
        "QCD_raw_O_1_to_10": 0 < qmask.sum() <= 10,
        "zero_QCD_rows_not_zero_physical_QCD": not qmask.any(),
    }
    if len(source_sums):
        qevents = pd.DataFrame({"source_uid": qsource.to_numpy(), "weight": qweights})
        qevents["pthat"] = qevents.source_uid.map(pthat)
        composition = qevents.groupby("pthat", as_index=False).agg(
            qcd_rows=("source_uid", "size"),
            qcd_sources=("source_uid", "nunique"),
            qcd_yield=("weight", "sum"),
            qcd_sumw2=("weight", lambda x: float(np.square(x).sum())),
        )
    else:
        composition = pd.DataFrame([{"pthat": "no_surviving_MC_rows", "qcd_rows": 0, "qcd_sources": 0, "qcd_yield": 0.0, "qcd_sumw2": 0.0}])
    return row, composition


def save(fig, name: str) -> None:
    fig.tight_layout()
    png, pdf = save_png_pdf(fig, OUT / "figures/png" / name)
    pdf.replace(OUT / "figures/pdf" / pdf.name)
    plt.close(fig)
    require(png.is_file() and (OUT / "figures/pdf" / pdf.name).is_file(), f"missing {name}")


def make_figures(curves: pd.DataFrame, folds: pd.DataFrame) -> None:
    apply_cms_style()
    specs = [
        ("01_conventional_roc", "epsilon_B", "epsilon_S", "Background efficiency", "Signal efficiency", False),
        ("02_log_background_efficiency", "epsilon_S", "epsilon_B", "Signal efficiency", "Background efficiency", True),
        ("03_background_rejection", "epsilon_S", "rejection", "Signal efficiency", "Background rejection", True),
        ("04_s_over_sqrt_b_efficiency", "epsilon_S", "epsilon_S_over_sqrt_epsilon_B", "Signal efficiency", r"$\epsilon_S/\sqrt{\epsilon_B}$", False),
        ("05_physical_background", "epsilon_S", "physical_B", "Signal efficiency", "Physical background yield", True),
        ("06_physical_signal_over_background", "epsilon_S", "S_over_B", "Signal efficiency", "Physical S/B", True),
        ("07_physical_signal_over_sqrt_background", "epsilon_S", "S_over_sqrt_B_diagnostic", "Signal efficiency", r"Physical $S/\sqrt{B}$ (diagnostic)", False),
        ("08_qcd_neff", "epsilon_S", "qcd_Neff", "Signal efficiency", r"QCD $N_{\rm eff}$", True),
    ]
    for name, x, y, xlabel, ylabel, logy in specs:
        fig, ax = plt.subplots(figsize=(6.6, 5.1))
        for variant in VARIANTS:
            d = curves[curves.variant.eq(variant)].sort_values(x)
            ax.plot(d[x].to_numpy(), d[y].to_numpy(), color=COLORS[variant], label=LABELS[variant])
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend()
        if name == "02_log_background_efficiency":
            ax.scatter([.40, .10], [1e-2, 1e-3], marker="*", s=90, color="#6A3D9A", zorder=5)
            ax.annotate("CMS HIG-24-010 resolved SR4b-conditioned literature reference;\nnot apples-to-apples with this broad global BDT.", (.40, 1e-2), xytext=(.26, 2.5e-4), fontsize=8, arrowprops={"arrowstyle": "->"})
        add_delphes_header(ax, "Immutable five-fold nested outer-OOF TRAIN")
        save(fig, name)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for variant in VARIANTS:
        for target, style in ((.40, "-"), (.10, "--")):
            d = folds[folds.variant.eq(variant) & np.isclose(folds.target_epsilon_S, target)].sort_values("outer_fold")
            ax.plot(d.outer_fold.to_numpy(), d.rejection.to_numpy(), marker="o", linestyle=style, color=COLORS[variant], label=f"{LABELS[variant]}, $\\epsilon_S$={target:.2f}")
    ax.set(xlabel="Outer fold", ylabel="Background rejection", xticks=range(5), yscale="log")
    ax.grid(alpha=.25); ax.legend(fontsize=8); add_delphes_header(ax, "Fold stability, TRAIN outer-OOF")
    save(fig, "09_fold_rejection_stability")


def shape_ladder(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    # Five broad score regions; merge downward until every retained bin has >=10 QCD rows.
    edges = np.unique(np.quantile(frame.score.to_numpy(float), np.linspace(0, 1, 6)))
    bins = pd.cut(frame.score, edges, include_lowest=True, duplicates="drop")
    rows = []
    for interval in bins.cat.categories:
        mask = bins.eq(interval).to_numpy()
        sig = mask & frame.class_label.eq(1).to_numpy(); bg = mask & frame.class_label.eq(0).to_numpy(); qcd = mask & frame.process_or_mode.eq("qcd_hardqcd").to_numpy()
        w = frame.resolved_selection_contribution_weight.to_numpy(float)
        s, b, qw, qw2 = w[sig].sum(), w[bg].sum(), w[qcd].sum(), np.square(w[qcd]).sum()
        rows.append({"variant": variant, "score_bin": str(interval), "S": s, "B": b, "qcd_raw_rows": int(qcd.sum()), "qcd_Neff": neff(qw, qw2), "ZA_bin": za(s, b), "supported": qcd.sum() >= 10 and neff(qw, qw2) >= 10})
    out = pd.DataFrame(rows)
    supported = out.supported.to_numpy(bool)
    zshape = math.sqrt(float(np.nansum(np.square(out.loc[supported, "ZA_bin"])))) if supported.any() else float("nan")
    out["supported_shape_ZA"] = zshape
    out["supported_shape_mu95_diagnostic"] = 1.96 / zshape if zshape > 0 else float("nan")
    return out


def main() -> None:
    for name in ("figures/png", "figures/pdf", "figure_data", "tables", "manifests"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    frames = {}
    for variant in VARIANTS:
        path = SOURCE / f"pooled_oof_{variant}.parquet"
        require(sha256(path) == EXPECTED[variant], f"immutable OOF hash changed: {variant}")
        frame = pd.read_parquet(path)
        require(len(frame) == 1_042_397 and frame.event_uid.nunique() == len(frame), "OOF population changed")
        frames[variant] = frame

    wp_rows, fold_rows, tail_rows, composition_rows, curve_rows, shape_parts = [], [], [], [], [], []
    for variant, frame in frames.items():
        for target in TARGETS:
            threshold = threshold_for_signal(frame, target)
            row, comp = metrics(frame, threshold)
            row.update(variant=variant, target_epsilon_S=target)
            wp_rows.append(row)
            comp = comp.assign(variant=variant, target_epsilon_S=target, score_threshold=threshold)
            composition_rows.append(comp)
            for fold in range(5):
                part = frame[frame.source_fold.eq(fold)].reset_index(drop=True)
                frow, _ = metrics(part, threshold_for_signal(part, target))
                fold_rows.append({"variant": variant, "outer_fold": fold, "target_epsilon_S": target, **frow})
        for target in TAIL_TARGETS:
            threshold = threshold_for_background(frame, target)
            row, _ = metrics(frame, threshold)
            supportable = row["raw_background_rows"] >= 100 and row["qcd_raw_rows"] >= 10 and row["qcd_Neff"] >= 10
            row.update(variant=variant, target_epsilon_B=target, scientifically_supportable=supportable,
                       support_reason="background>=100, QCD rows>=10, QCD Neff>=10" if supportable else "insufficient empirical background and/or QCD support")
            tail_rows.append(row)
        for target in np.linspace(.02, .98, 97):
            row, _ = metrics(frame, threshold_for_signal(frame, float(target)))
            row.update(variant=variant, target_epsilon_S=float(target), epsilon_S_over_sqrt_epsilon_B=row["epsilon_S"] / math.sqrt(row["epsilon_B"]) if row["epsilon_B"] > 0 else float("nan"))
            curve_rows.append(row)
        shape_parts.append(shape_ladder(frame, variant))

    wp = pd.DataFrame(wp_rows); folds = pd.DataFrame(fold_rows); tails = pd.DataFrame(tail_rows); curves = pd.DataFrame(curve_rows)
    wp.to_csv(OUT / "tables/working_points.tsv", sep="\t", index=False)
    folds.to_csv(OUT / "tables/fold_stability.tsv", sep="\t", index=False)
    tails.to_csv(OUT / "tables/deep_tail.tsv", sep="\t", index=False)
    pd.concat(composition_rows, ignore_index=True).to_csv(OUT / "tables/qcd_pthat_composition.tsv", sep="\t", index=False)
    curves.to_csv(OUT / "figure_data/roc_yield_support_curves.tsv", sep="\t", index=False)
    shape = pd.concat(shape_parts, ignore_index=True); shape.to_csv(OUT / "tables/supported_score_shape_ladder.tsv", sep="\t", index=False)
    spread = folds.groupby(["variant", "target_epsilon_S"]).agg(rejection_min=("rejection", "min"), rejection_max=("rejection", "max"), rejection_mean=("rejection", "mean"), epsilon_B_min=("epsilon_B", "min"), epsilon_B_max=("epsilon_B", "max"), qcd_Neff_min=("qcd_Neff", "min"), qcd_raw_min=("qcd_raw_rows", "min")).reset_index()
    spread.to_csv(OUT / "tables/fold_spread.tsv", sep="\t", index=False)

    feature_gap = pd.DataFrame([
        ("continuous per-jet b-tag discriminator", False, False, False, True, "authoritative extractor reads Delphes Jet.BTag, which is binary 0/1 in frozen TRAIN sources"),
        ("fifth b-tag-ranked jet", False, False, True, False, "variable-length selected jets exist upstream; fifth scalar jet not in common table"),
        ("five leading b-tagged jet four-vectors", False, False, True, False, "only chosen candidate j1-j4 scalar four-vectors stored"),
        ("alternative pair masses among stored four candidate jets", False, True, False, False, "derivable from j1-j4 pt/eta/phi/mass"),
        ("pair delta_eta", False, True, False, False, "derivable for alternative j1-j4 pairings"),
        ("pair delta_phi", False, True, False, False, "derivable for alternative j1-j4 pairings"),
        ("pair delta_R", False, True, False, False, "derivable for alternative j1-j4 pairings"),
        ("continuous candidate b-tag information", False, False, False, True, "j1_btag...j4_btag and upstream Jet.BTag contain only 0/1"),
        ("candidate mass-resolution information", False, False, True, False, "no per-candidate mass-resolution columns"),
        ("truth pairing information (diagnostic only)", False, False, True, False, "pairing columns are reconstructed choices, not truth matching"),
    ], columns=["CMS_information", "available_in_current_tables", "derivable_now", "requires_TRAIN_re_extraction", "requires_new_Delphes_generation", "evidence"])
    feature_gap.to_csv(OUT / "tables/cms_feature_schema_gap.tsv", sep="\t", index=False)
    make_figures(curves, folds)

    # Diagnosis uses the mass-aware model as the stronger current representation.
    aware40 = wp[wp.variant.eq("global_mass_aware") & np.isclose(wp.target_epsilon_S, .40)].iloc[0]
    aware10 = wp[wp.variant.eq("global_mass_aware") & np.isclose(wp.target_epsilon_S, .10)].iloc[0]
    reaches = aware40.rejection >= 50 and aware10.rejection >= 500
    qcd_starved = aware40.qcd_Neff < 100 or aware10.qcd_Neff < 100 or aware10.qcd_raw_rows <= 10
    diagnosis = "CLASSIFIER_DISCRIMINATION_PROMISING_QCD_MODEL_LIMITING" if reaches and qcd_starved else ("CLASSIFIER_AND_QCD_LIMITING" if qcd_starved else "CLASSIFIER_REPRESENTATION_LIMITING")
    diag = {"diagnosis": diagnosis, "reaches_contextual_scales": bool(reaches), "physical_tail_qcd_starved": bool(qcd_starved), "validation_payloads_opened": 0, "test_payloads_opened": 0,
            "PRIMARY_PERFORMANCE_SOURCE": "five-fold nested outer-OOF", "cms_reference_role": "context only; resolved SR4b-conditioned and not apples-to-apples"}
    (OUT / "diagnosis.json").write_text(json.dumps(diag, indent=2, sort_keys=True) + "\n")

    def fmtrow(row) -> str:
        return f"{row.target_epsilon_S:.6g} | {row.epsilon_B:.6g} | {row.rejection:.3g} | {row.physical_S:.4g} | {row.physical_B:.4g} | {int(row.qcd_raw_rows)} | {row.qcd_Neff:.3g}"
    lines = ["# Harvey-facing global BDT background-suppression brief", "", "## MEASURED FACTS", "", "All performance values below use the immutable five-fold nested outer-OOF TRAIN predictions. ROC quantities are unweighted event counts; physical yields use signed Run-2 evaluation weights.", ""]
    for variant in VARIANTS:
        lines += [f"### {LABELS[variant]}", "", "target epsilon_S | achieved epsilon_S | epsilon_B | rejection | physical S | physical B | QCD raw rows | QCD Neff", "---:|---:|---:|---:|---:|---:|---:|---:"]
        for row in wp[wp.variant.eq(variant)].itertuples():
            lines.append(f"{row.target_epsilon_S:.6g} | " + fmtrow(row))
        lines.append("")
    lines += ["Zero surviving QCD simulation rows, where encountered, must not be interpreted as zero physical QCD.", "", "## CMS LITERATURE REFERENCE", "", "The contextual HIG-24-010 resolved SR4b-conditioned guide points are approximately rejection 100 at epsilon_S~0.40 and 1000 at epsilon_S~0.10. They are not apples-to-apples with this broad global classifier and were not used for tuning.", "", "## DIAGNOSIS", "", f"`{diagnosis}`", "", "The current representation lacks continuous b-tag scores, the fifth b-tag-ranked jet, its four-vector, and five-jet alternative combinations. High-score physical interpretation is additionally limited wherever QCD Neff falls below 100 or only O(1-10) QCD rows survive.", "", "## UNRESOLVED HYPOTHESES", "", "A richer global representation may recover additional rejection, but this cannot be established from the current OOF products. Sparse weighted QCD tails may also obscure the true physical rejection. Broad-preselection differences remain entangled with the non-apples-to-apples CMS comparison.", "", "## NEXT EXPERIMENT", "", "Perform a bounded TRAIN-only re-extraction canary that materializes the five leading b-tag-ranked jet four-vectors, alternative pair kinematics, and any genuinely continuous tag discriminator available upstream. Preserve five-fold source-group nested CV and sealed validation/test. Do not submit a large extraction or new scientific training campaign until schema closure and QCD-tail support are demonstrated.", ""]
    (OUT / "harvey_brief.md").write_text("\n".join(lines))

    protocol = {"status": "prepared_not_started", "scope": "enriched global BDT only", "requires_train_re_extraction": True, "new_delphes_generation_required_for_schema": False,
                "candidate_inputs": ["five leading b-tag-ranked jet pt/eta/phi/mass", "all pair masses/delta_eta/delta_phi/delta_R among those jets", "continuous tag discriminator only if upstream branch is genuinely continuous"],
                "forbidden_inputs": ["truth pairing"], "cv": "same immutable five-fold source-group nested CV", "validation_payloads_allowed": False, "test_payloads_allowed": False,
                "worker_plan": "bounded 1 signal + 1 QCD + 1 ttbar source schema/provenance canary; stop before scale-out", "production_submission_authorized": False}
    (OUT / "enriched_global_bdt_protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    old_za = 0.0265
    gap = (74.0 / 5.9) ** 2
    sensitivity = {"old_simple_one_bin": {"S": 184.7, "B": 4.865e7, "ZA": old_za, "mu95_diagnostic": 1.96 / old_za},
                   "cms_expected_mu95_literature_benchmark": 5.9, "required_ZA_for_mu95_5p9": 1.96 / 5.9,
                   "ZA_improvement_factor": (1.96 / 5.9) / old_za, "background_reduction_factor_if_S_fixed_counting_only": gap,
                   "interpretation": "Equivalent fixed-S one-bin background reduction is only a scaling diagnostic; genuine gain may come from rejection, categories, and supported shape information."}
    (OUT / "sensitivity_gap.json").write_text(json.dumps(sensitivity, indent=2, sort_keys=True) + "\n")
    ladder_rows = [{"method": "old_simple_R_HH_one_bin", "variant": "historical", "signal_efficiency": float("nan"), "S": 184.7, "B": 4.865e7, "ZA_stat_only": old_za, "mu95_diagnostic": 1.96 / old_za, "support_rule": "provided frozen historical count"}]
    for row in wp.itertuples():
        ladder_rows.append({"method": "existing_BDT_working_point", "variant": row.variant, "signal_efficiency": row.epsilon_S, "S": row.physical_S, "B": row.physical_B, "ZA_stat_only": row.ZA_stat_only, "mu95_diagnostic": 1.96 / row.ZA_stat_only if row.ZA_stat_only > 0 else float("nan"), "support_rule": "immutable pooled outer-OOF count"})
    for variant in VARIANTS:
        first = shape[shape.variant.eq(variant)].iloc[0]
        ladder_rows.append({"method": "supported_five_bin_BDT_score_shape", "variant": variant, "signal_efficiency": float("nan"), "S": float("nan"), "B": float("nan"), "ZA_stat_only": first.supported_shape_ZA, "mu95_diagnostic": first.supported_shape_mu95_diagnostic, "support_rule": "only bins with QCD raw>=10 and QCD Neff>=10 included; unsupported bins omitted"})
    pd.DataFrame(ladder_rows).to_csv(OUT / "tables/sensitivity_diagnostic_ladder.tsv", sep="\t", index=False)

    manifest_rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.tsv":
            manifest_rows.append({"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest_rows).to_csv(OUT / "manifests/artifact_manifest.tsv", sep="\t", index=False)
    print("HARVEY_TRAIN_ONLY_ROC_AUDIT=PASS")


if __name__ == "__main__":
    main()
