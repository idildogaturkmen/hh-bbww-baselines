"""Build figures/u_probability_fp32_tail.png from the real frozen
per-event SPA-Net 2M seed-0 score archive (read-only; see
census_fp32_scores.py for the data-provenance note). No logits archive
exists for this pipeline (verified by reading evaluate_classification.py
and evaluate_classification_10M.py -- neither persists per-event
z0/z1), so no u_logit_tail.png or u_prob_vs_u_logit.png is produced;
their absence is documented in the report rather than faked.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = ("/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/"
           "phase4AF_spanet_partial_events_population_correction_20260819_v1/"
           "spanet_2M_seed0_classification_evaluation_v1/work")
PROBS_PATH = f"{SRC_DIR}/val_signal_probs_400k.npy"
LABELS_PATH = f"{SRC_DIR}/val_process_labels_400k.npy"
OUT_DIR = "/uscms_data/d3/iturkmen/repos/hh-bbww-baselines-bdt/docs/checkpoints/track_b_harvey_fp32_score_precision_20260906_v1"

SPACING = 2.0 ** -24


def u_of(score_f64):
    with np.errstate(divide="ignore"):
        return -np.log10(1.0 - score_f64)


def main():
    probs = np.load(PROBS_PATH).astype(np.float64)
    labels = np.load(LABELS_PATH)

    is_signal = labels == "signal"
    is_bg = ~is_signal

    u_signal = u_of(probs[is_signal])
    u_bg = u_of(probs[is_bg])

    # Restrict to the finite tail (score < 1.0 exactly; score==1.0 -> u=inf,
    # plotted separately as an overflow annotation, not silently dropped).
    finite_signal = np.isfinite(u_signal)
    n_inf_signal = int((~finite_signal).sum())
    u_signal_finite = u_signal[finite_signal]

    u_cut = 5.5
    tail_signal = u_signal_finite[u_signal_finite > u_cut]
    tail_bg = u_bg[np.isfinite(u_bg) & (u_bg > u_cut)]

    lattice_k = np.arange(1, 12)
    lattice_u = -np.log10(lattice_k * SPACING)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bin_edges = np.arange(u_cut, 7.35, 0.02)
    ax.hist(tail_signal, bins=bin_edges, color="#2b6cb0", alpha=0.85,
            label=f"signal (n={len(tail_signal)}, u>{u_cut})")
    if len(tail_bg):
        ax.hist(tail_bg, bins=bin_edges, color="#c53030", alpha=0.7,
                label=f"all background (n={len(tail_bg)}, u>{u_cut})")
    else:
        ax.text(0.98, 0.60, "all background: 0 events with u > 5.5",
                transform=ax.transAxes, color="#c53030", fontsize=9, ha="right")

    for k, uk in zip(lattice_k, lattice_u):
        if uk < u_cut:
            continue
        ax.axvline(uk, color="#444444", linestyle=":", linewidth=0.8, zorder=0)
        ax.text(uk, ax.get_ylim()[1] * 0.0, f"k={k}", rotation=90,
                fontsize=6.5, ha="right", va="bottom", color="#444444")

    ax.axvline(6.9237, color="#805ad5", linestyle="--", linewidth=1.2,
               label="u=6.9237 (k=2 lattice point)")
    ax.axvline(6.6227, color="#dd6b20", linestyle="--", linewidth=1.2,
               label="u=6.6226 (k=4 lattice point)")

    ax.set_xlabel(r"$u = -\log_{10}(1 - \mathrm{score})$  (score stored as float32)")
    ax.set_ylabel("events / 0.02 in u")
    ax.set_title("SPA-Net 2M seed-0 dev-cohort tail: score=1 FP32 lattice pile-up\n"
                 f"({n_inf_signal} signal events at exact float32 1.0 not shown, plotted at u=inf)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(u_cut, 7.35)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figures/u_probability_fp32_tail.png", dpi=160)
    print(f"wrote {OUT_DIR}/figures/u_probability_fp32_tail.png")
    print(f"n_inf_signal (exact score==1.0): {n_inf_signal}")


if __name__ == "__main__":
    main()
