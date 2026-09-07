"""v3 Task 4: Harvey-quality figure focused on the literal 0.9997 /
0.99997 regions, plus a small tie-examples table. Pure post-processing
on the v2 parquet; no model/checkpoint/HDF5 touched.
"""
import csv
import math

import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V2_DIR = ("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines-bdt/docs/checkpoints/"
          "track_b_harvey_fp32_score_precision_20260906_v2_governing10m")
PARQUET_PATH = f"{V2_DIR}/SPA10M_EVENT_LOGITS_400K.parquet"
OUT_DIR = ("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines-bdt/docs/checkpoints/"
           "track_b_harvey_fp32_score_precision_20260907_v3_exact_harvey_thresholds")

U_A = 3.5228787452803854   # score > 0.9997
U_B = 4.522878745280707    # score > 0.99997


def main():
    t = pq.read_table(PARQUET_PATH)
    d = t.to_pydict()
    score32 = np.asarray(d["score_float32_production"])
    delta = np.asarray(d["delta"])
    u_logit = np.asarray(d["u_logit"])
    u_prob32 = np.asarray(d["u_prob32"])
    proc = np.asarray(d["process_label"])
    row_index = np.asarray(d["row_index"])

    is_signal = proc == "signal"
    plot_floor = U_A - 0.6

    finite = np.isfinite(u_prob32)
    sel = is_signal & (u_logit > plot_floor)
    inf_sel = is_signal & (~finite) & (u_logit > plot_floor)

    fig, ax = plt.subplots(figsize=(9, 8))

    finite_sel = sel & finite
    ax.scatter(u_logit[finite_sel], u_prob32[finite_sel], s=8, alpha=0.35,
               color="#2b6cb0", label="signal, finite stored score (float32)")

    y_inf_plot = u_prob32[finite_sel].max() + 0.35 if finite_sel.any() else U_B + 0.5
    if inf_sel.sum():
        ax.scatter(u_logit[inf_sel], np.full(inf_sel.sum(), y_inf_plot), s=18, marker="^",
                   color="#805ad5",
                   label=(f"score==1.0 exactly (n={inf_sel.sum()}): TRUE u_prob32=+infinity;\n"
                          f"plotted at y={y_inf_plot:.2f} as a FINITE coordinate for visualization "
                          f"ONLY -- not the true stored u"))

    lims = [plot_floor, max(u_logit[sel].max() if sel.any() else U_B + 1, y_inf_plot + 0.2)]
    ax.plot(lims, lims, color="#888888", linestyle=":", linewidth=1, label="y = x (no compression)")

    ax.axvline(U_A, color="#2f855a", linestyle="--", linewidth=1.4,
               label=f"u = {U_A:.6f}  (score > 0.9997)")
    ax.axvline(U_B, color="#c05621", linestyle="--", linewidth=1.4,
               label=f"u = {U_B:.6f}  (score > 0.99997)")
    ax.axhline(U_A, color="#2f855a", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axhline(U_B, color="#c05621", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_xlabel(r"$u_{\rm logit} = \mathrm{softplus}(\Delta)/\ln(10)$  (float64, continuous, from raw logits)")
    ax.set_ylabel(r"$u_{\rm prob32} = -\log_{10}(1-\mathrm{score}_{f32})$  (stored float32 production score)")
    ax.set_title("Governing SPA-Net 10M: FP32 storage compresses a continuous logit\n"
                 "margin into discrete steps near Harvey's 0.9997 / 0.99997 thresholds",
                 fontsize=11)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_xlim(*lims)
    ax.set_ylim(plot_floor, y_inf_plot + 0.2)
    fig.text(0.5, 0.005,
             "Caption: each horizontal band is one stored float32 probability shared by many events "
             "with distinct raw logit margins (Delta/u_logit). Triangles mark score==1.0 exactly, "
             "whose true u_prob32 is +infinity -- shown at a finite y purely so they can be plotted.",
             ha="center", fontsize=7.5, style="italic", wrap=True)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(f"{OUT_DIR}/figures/harvey_exact_threshold_compression.png", dpi=170)
    plt.close(fig)
    print(f"wrote {OUT_DIR}/figures/harvey_exact_threshold_compression.png")

    # ---- tie-examples table: the largest-multiplicity FP32 bin above 0.9997 ----
    mask_tail = score32 > 0.9997
    vals, counts = np.unique(score32[mask_tail], return_counts=True)
    imax = int(np.argmax(counts))
    tie_value = vals[imax]
    tie_count = int(counts[imax])
    tie_sel = np.nonzero(score32 == tie_value)[0]
    order = np.argsort(-delta[tie_sel])

    rows = []
    # a representative spread: 5 highest-Delta, 5 lowest-Delta members of this one FP32 bin
    picks = list(order[:5]) + list(order[-5:])
    for rank, i in enumerate(picks):
        idx = tie_sel[i]
        rows.append(dict(
            example_rank=rank + 1,
            row_index=int(row_index[idx]),
            process_label=str(proc[idx]),
            shared_score_float32=float(tie_value),
            delta=float(delta[idx]),
            u_logit=float(u_logit[idx]),
        ))

    with open(f"{OUT_DIR}/SPA10M_TIE_EXAMPLES_TABLE.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["example_rank", "row_index", "process_label",
                                           "shared_score_float32", "delta", "u_logit"])
        w.writeheader()
        w.writerows(rows)

    print(f"largest FP32 tie above 0.9997: score={tie_value!r}, {tie_count} events, "
          f"delta range [{delta[tie_sel].min():.6f}, {delta[tie_sel].max():.6f}], "
          f"{len(np.unique(delta[tie_sel]))} distinct delta values")
    print(f"wrote {OUT_DIR}/SPA10M_TIE_EXAMPLES_TABLE.csv")

    return dict(tie_value=float(tie_value), tie_count=tie_count,
                delta_min=float(delta[tie_sel].min()), delta_max=float(delta[tie_sel].max()),
                n_distinct_delta_in_tie=int(len(np.unique(delta[tie_sel]))))


if __name__ == "__main__":
    main()
