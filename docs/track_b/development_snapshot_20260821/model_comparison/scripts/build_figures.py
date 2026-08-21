"""Paper-ready figures (PDF + SVG + PNG), clean HEP-publication aesthetic
(CMS-inspired conventions, but NO CMS/CMS-Preliminary/CMS-Simulation
logo -- this is not an official CMS result).

Reads ONLY:
  - MASTER_DEVELOPMENT_RESULTS.json (working points, AUC -- for panels A/B/C)
  - the frozen SPA-Net 2M raw score arrays (val_signal_probs_400k.npy,
    val_process_labels_400k.npy) -- the only model with frozen exact
    per-event score arrays available -- for panel D (conventional ROC)
  - the frozen 2M and 10M training result JSONs' epoch_history -- for
    panel E (training-scaling curves)

Panel F (normalized score distributions for SPA-Net 10M and BDT-KF) is
explicitly SKIPPED: neither model has a frozen raw per-event score
array available (10M's evaluation script only saved summary statistics
+ working points, not the 400,000 raw scores; BDT-KF's matched-400k
scoring likewise only saved summary/working-point JSON) -- per
instruction, this snapshot does not regenerate data merely to produce
an optional plot.

Uses system python3 + matplotlib (not the pinned SPA-Net pixi env,
which lacks matplotlib) -- plotting has no scientific dependency on
the frozen training environment. No data is regenerated; every number
plotted is read from an already-frozen, hash-verified artifact.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER = os.path.join(OUT_DIR, "master_results", "MASTER_DEVELOPMENT_RESULTS.json")
FIG_DIR = os.path.join(OUT_DIR, "figures")

HH4B = "/uscms_data/d3/iturkmen/hh4b_delphes"
SPANET_2M_EVAL_DIR = os.path.join(HH4B, "track_b_phase4_preflight_20260812",
                                   "phase4AF_spanet_partial_events_population_correction_20260819_v1",
                                   "spanet_2M_seed0_classification_evaluation_v1")
SIGNAL_PROBS_2M = os.path.join(SPANET_2M_EVAL_DIR, "work", "val_signal_probs_400k.npy")
PROCESS_LABELS_2M = os.path.join(SPANET_2M_EVAL_DIR, "work", "val_process_labels_400k.npy")

RUN_2M_RESULT = os.path.join(HH4B, "track_b_phase4_preflight_20260812",
                              "phase4AF_spanet_partial_events_population_correction_20260819_v1",
                              "spanet_2M_seed0_production_training_v1", "work", "training_2M_seed0_result.json")
RUN_10M_RESULT = os.path.join(HH4B, "track_b_phase4_preflight_20260812",
                               "phase4AI_spanet_10M_scaling_seed0_20260821_v1",
                               "full_run_attempt2_eaf_local_staging", "work", "training_10M_seed0_full_attempt2_result.json")

DEV_WATERMARK = "DEVELOPMENT ONLY -- not an official CMS result"

MODEL_STYLE = {
    "BDT_K": dict(color="#6c757d", marker="s", label="BDT-K (kinematics-only)"),
    "BDT_KF": dict(color="#1f77b4", marker="D", label="BDT-KF (+flavor tag)"),
    "SPANET_2M": dict(color="#d62728", marker="o", label="SPA-Net 2M (primary)"),
    "SPANET_10M": dict(color="#2ca02c", marker="^", label="SPA-Net 10M (primary)"),
}
MODEL_ORDER = ["BDT_K", "BDT_KF", "SPANET_2M", "SPANET_10M"]


def hep_style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.linewidth": 1.1, "axes.labelsize": 12, "axes.titlesize": 12,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.minor.size": 3, "ytick.minor.size": 3,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "legend.frameon": False, "legend.fontsize": 9.5,
        "figure.dpi": 150, "savefig.bbox": "tight",
    })


def watermark(ax):
    ax.text(0.02, 0.02, DEV_WATERMARK, transform=ax.transAxes, fontsize=7.5,
             color="#888888", style="italic", ha="left", va="bottom")


def save_all(fig, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.close(fig)


def rejection_curve_plot(models, bkg_key, title, name, annotate_finite_support=False):
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    n_omitted_zero = 0
    for mid in MODEL_ORDER:
        st = MODEL_STYLE[mid]
        wps = models[mid]["working_points"]
        xs, ys, warn_xy = [], [], []
        for wp in wps:
            r = wp[f"R_{bkg_key}"]
            if r is None:
                n_omitted_zero += 1
                continue
            xs.append(wp["achieved_epsS"])
            ys.append(r)
            if wp["finite_support_warnings"]:
                warn_xy.append((wp["achieved_epsS"], r))
        order = np.argsort(xs)
        xs = np.array(xs)[order]
        ys = np.array(ys)[order]
        ax.plot(xs, ys, color=st["color"], marker=st["marker"], markersize=5.5,
                 linewidth=1.6, label=st["label"])
        if annotate_finite_support and warn_xy:
            wx, wy = zip(*warn_xy)
            ax.scatter(wx, wy, s=110, facecolors="none", edgecolors=st["color"],
                       linewidths=1.6, zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel(r"Signal efficiency $\epsilon_S$")
    ax.set_ylabel(r"Background rejection $R_B = 1/\epsilon_B$")
    ax.set_title(title, fontsize=12)
    if annotate_finite_support:
        handles, labels = ax.get_legend_handles_labels()
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], marker="o", markersize=8, markerfacecolor="none",
                               markeredgecolor="black", linestyle="none",
                               label="finite-support / Poisson-limited"))
        ax.legend(handles=handles, loc="upper right", fontsize=8.5)
        if n_omitted_zero:
            fig.text(0.5, -0.01,
                      f"{n_omitted_zero} working point(s) omitted: zero surviving background events (rejection undefined).",
                      ha="center", fontsize=8, style="italic", color="#555555")
    else:
        ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.25)
    watermark(ax)
    save_all(fig, name)


def main():
    m = json.load(open(MASTER))
    models = m["models"]
    os.makedirs(FIG_DIR, exist_ok=True)
    hep_style()

    # ---- Panel A: signal efficiency vs all-background rejection ----
    rejection_curve_plot(models, "all_background",
                          "DEVELOPMENT: signal efficiency vs. all-background rejection\n(matched 400k development cohort)",
                          "fig_A_signal_eff_vs_all_background_rejection")

    # ---- Panel B: signal efficiency vs QCD rejection ----
    rejection_curve_plot(models, "qcd",
                          "DEVELOPMENT: signal efficiency vs. QCD rejection\n(matched 400k development cohort)",
                          "fig_B_signal_eff_vs_qcd_rejection")

    # ---- Panel C: signal efficiency vs ttbar rejection, finite-support annotated ----
    rejection_curve_plot(models, "ttbar",
                          "DEVELOPMENT: signal efficiency vs. ttbar rejection\n(matched 400k development cohort -- finite-support limited at tight working points)",
                          "fig_C_signal_eff_vs_ttbar_rejection", annotate_finite_support=True)

    # ---- Panel D: conventional ROC curve, SPA-Net 2M only (only model with frozen raw score arrays) ----
    probs = np.load(SIGNAL_PROBS_2M)
    labels = np.load(PROCESS_LABELS_2M, allow_pickle=True)
    is_signal = (labels == "signal")
    is_bg = ~is_signal
    assert probs.shape[0] == 400000 and is_signal.sum() == 193358

    def roc(scores, y_is_pos):
        order = np.argsort(-scores)
        y_sorted = y_is_pos[order]
        tps = np.cumsum(y_sorted)
        fps = np.cumsum(~y_sorted)
        n_pos = y_is_pos.sum()
        n_neg = (~y_is_pos).sum()
        tpr = tps / n_pos
        fpr = fps / n_neg
        return np.concatenate([[0.0], fpr]), np.concatenate([[0.0], tpr])

    y_pos = is_signal
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    for bkg_name, bkg_mask, color in [
        ("all-background", is_bg, "#d62728"),
        ("QCD", labels == "qcd", "#ff7f0e"),
        ("ttbar", labels == "ttbar", "#9467bd"),
    ]:
        sel = is_signal | bkg_mask
        fpr, tpr = roc(probs[sel], is_signal[sel])
        auc = np.trapz(tpr, fpr)
        ax.plot(fpr, tpr, color=color, linewidth=1.8, label=f"vs {bkg_name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1.0, label="random")
    ax.set_xlabel(r"False positive rate $\epsilon_B$")
    ax.set_ylabel(r"True positive rate $\epsilon_S$")
    ax.set_title("DEVELOPMENT: conventional ROC curve, SPA-Net 2M primary\n(only model with frozen exact per-event score arrays)", fontsize=11.5)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    watermark(ax)
    save_all(fig, "fig_D_roc_curve_spanet_2M")

    # ---- Panel E: 2M vs 10M training curves ----
    r2m = json.load(open(RUN_2M_RESULT))
    r10m = json.load(open(RUN_10M_RESULT))
    eh2, eh10 = r2m["epoch_history"], r10m["epoch_history"]

    def series(eh, key):
        return [e["epoch"] for e in eh], [e.get(key) for e in eh]

    def scaling_plot(key, ylabel, title, name, best_epoch_2m=None, best_epoch_10m=None):
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        x2, y2 = series(eh2, key)
        x10, y10 = series(eh10, key)
        ax.plot(x2, y2, color="#d62728", marker=".", markersize=4, linewidth=1.3, label="2M (n_train=2,000,000)")
        ax.plot(x10, y10, color="#2ca02c", marker=".", markersize=4, linewidth=1.3, label="10M (n_train=10,000,000)")
        if best_epoch_2m is not None:
            ax.axvline(best_epoch_2m, color="#d62728", linestyle=":", alpha=0.5)
        if best_epoch_10m is not None:
            ax.axvline(best_epoch_10m, color="#2ca02c", linestyle=":", alpha=0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
        watermark(ax)
        save_all(fig, name)

    scaling_plot("validation_average_jet_accuracy", "validation_average_jet_accuracy",
                 "DEVELOPMENT: SPA-Net 2M vs. 10M -- assignment accuracy\n(NOT an event-classification metric)",
                 "fig_E1_epoch_vs_validation_average_jet_accuracy",
                 best_epoch_2m=r2m["primary_checkpoint_best_epoch"], best_epoch_10m=r10m["primary_checkpoint_best_epoch"])
    scaling_plot("val/loss/total_loss", "val/loss/total_loss",
                 "DEVELOPMENT: SPA-Net 2M vs. 10M -- secondary validation loss",
                 "fig_E2_epoch_vs_val_loss_total_loss")
    scaling_plot("loss/classification/EVENT/signal", "loss/classification/EVENT/signal (training)",
                 "DEVELOPMENT: SPA-Net 2M vs. 10M -- classification loss\n(training-time cross-entropy, not AUC)",
                 "fig_E3_epoch_vs_classification_loss")

    print("Wrote figures to", FIG_DIR)
    print("Panel F (SPA-Net 10M / BDT-KF normalized score distributions) SKIPPED: "
          "no frozen raw per-event score array available for either model; "
          "not regenerated per instruction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
