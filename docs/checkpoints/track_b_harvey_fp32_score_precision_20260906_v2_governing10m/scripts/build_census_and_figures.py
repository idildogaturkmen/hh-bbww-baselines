"""v2: build the exact governing-10M FP32 census, the FP32-vs-logit bin
summary, and all figures, from the real per-event parquet produced by
run_governing_10m_tail_inference.py. Pure post-processing -- no model,
no checkpoint, no HDF5 touched here.
"""
import csv
import json
import struct

import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/uscms_data/d3/iturkmen/repos/hh-bbww-baselines-bdt/docs/checkpoints/track_b_harvey_fp32_score_precision_20260906_v2_governing10m"
PARQUET_PATH = f"{OUT_DIR}/SPA10M_EVENT_LOGITS_400K.parquet"

SPACING = 2.0 ** -24
U_CUT = 5.5
K_MAX = 64


def f32_hex_be(x):
    return struct.pack(">f", np.float32(x)).hex()


def load():
    t = pq.read_table(PARQUET_PATH)
    d = t.to_pydict()
    for k in ("row_index", "z_background", "z_signal", "delta", "score_float32_production",
              "k_lattice", "u_prob32", "u_logit", "sigmoid_delta_float64"):
        d[k] = np.asarray(d[k])
    d["process_label"] = np.asarray(d["process_label"])
    d["score_float32_hex_be"] = np.asarray(d["score_float32_hex_be"])
    return d


def populations(d):
    proc = d["process_label"]
    return {
        "signal": proc == "signal",
        "all_background": proc != "signal",
        "qcd": proc == "qcd",
        "ttbar": proc == "ttbar",
        "other_background": (proc != "signal") & (proc != "qcd") & (proc != "ttbar"),
    }


def build_unique_value_census(d, pops):
    rows = []
    tail_mask_any = d["u_prob32"] > U_CUT
    for name, mask in pops.items():
        sel = mask & tail_mask_any
        n_tail = int(sel.sum())
        scores = d["score_float32_production"][sel].astype(np.float32)
        uniq, counts = np.unique(scores, return_counts=True)
        order = np.argsort(-uniq)
        uniq, counts = uniq[order], counts[order]
        for val, cnt in zip(uniq, counts):
            v64 = float(val)
            k = 0 if val >= np.float32(1.0) else int(round((1.0 - v64) / SPACING))
            u = float("inf") if val >= np.float32(1.0) else float(-np.log10(1.0 - v64))
            rows.append(dict(
                population=name,
                score_f32=repr(v64),
                score_f32_exact_decimal=f"{v64:.20f}".rstrip("0").rstrip("."),
                score_f32_hex_be=f32_hex_be(val),
                k_round=k,
                u=u,
                raw_count=int(cnt),
            ))
        print(f"{name:16s} n_total={mask.sum():7d} n_tail(u>5.5)={n_tail:6d} n_unique_tail={len(uniq):4d}")
    with open(f"{OUT_DIR}/SPA10M_FP32_UNIQUE_SCORE_CENSUS.csv", "w", newline="") as f:
        fieldnames = ["population", "score_f32", "score_f32_exact_decimal", "score_f32_hex_be",
                      "k_round", "u", "raw_count"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_DIR}/SPA10M_FP32_UNIQUE_SCORE_CENSUS.csv ({len(rows)} rows)")
    return rows


def build_lattice_bin_table(d, pops):
    """Explicit k=1..K_MAX census (including zero-count bins) plus exact
    score==1, per population -- Harvey's explicit request, distinct from
    the unique-value census above (which only lists k's that occur)."""
    rows = []
    score32 = d["score_float32_production"].astype(np.float32)
    for name, mask in pops.items():
        arr = score32[mask]
        n_eq1 = int((arr == np.float32(1.0)).sum())
        rows.append(dict(population=name, k="exact_1.0", u="inf", count=n_eq1))
        for k in range(1, K_MAX + 1):
            v = np.float32(1.0) - np.float32(k) * np.float32(SPACING)
            cnt = int((arr == v).sum())
            u = float(-np.log10(k * SPACING))
            rows.append(dict(population=name, k=k, u=f"{u:.6f}", count=cnt))
    with open(f"{OUT_DIR}/SPA10M_LATTICE_BINS_K1_TO_K64.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["population", "k", "u", "count"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_DIR}/SPA10M_LATTICE_BINS_K1_TO_K64.csv ({len(rows)} rows)")
    return rows


def spike_counts(d, pops):
    score32 = d["score_float32_production"].astype(np.float32)
    out = {}
    for name, mask in pops.items():
        arr = score32[mask]
        v2 = np.float32(1.0) - np.float32(2) * np.float32(SPACING)
        v4 = np.float32(1.0) - np.float32(4) * np.float32(SPACING)
        out[name] = dict(
            k2_u6p9237=int((arr == v2).sum()),
            k4_u6p6227=int((arr == v4).sum()),
            exact_1p0=int((arr == np.float32(1.0)).sum()),
        )
    return out


def build_fp32_vs_logit_bin_summary(d, pops):
    """For each major FP32 probability bin (k=0..~20, i.e. the ones that
    actually occur) in the signal population (the only population with
    any u>5.5 events -- see report), examine whether events tied at the
    identical float32 probability remain separated/orderable in Delta /
    u_logit space."""
    rows = []
    is_signal = pops["signal"]
    score32 = d["score_float32_production"][is_signal].astype(np.float32)
    delta = d["delta"][is_signal]
    u_logit = d["u_logit"][is_signal]
    u_prob32 = d["u_prob32"][is_signal]

    tail = u_prob32 > U_CUT
    uniq_vals = np.unique(score32[tail])
    uniq_vals = uniq_vals[np.argsort(-uniq_vals)]

    for val in uniq_vals:
        sel = score32 == val
        n = int(sel.sum())
        if n == 0:
            continue
        d_sel = delta[sel]
        ul_sel = u_logit[sel]
        k = 0 if val >= np.float32(1.0) else int(round((1.0 - float(val)) / SPACING))
        rows.append(dict(
            k_round=k,
            score_f32=repr(float(val)),
            n_events=n,
            n_distinct_delta=int(len(np.unique(d_sel))),
            n_distinct_u_logit=int(len(np.unique(ul_sel))),
            delta_min=float(d_sel.min()), delta_median=float(np.median(d_sel)), delta_max=float(d_sel.max()),
            u_logit_min=float(ul_sel.min()), u_logit_median=float(np.median(ul_sel)), u_logit_max=float(ul_sel.max()),
            all_events_separated_in_logit_space=bool(len(np.unique(d_sel)) == n),
        ))

    with open(f"{OUT_DIR}/SPA10M_FP32_VS_LOGIT_BIN_SUMMARY.csv", "w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_DIR}/SPA10M_FP32_VS_LOGIT_BIN_SUMMARY.csv ({len(rows)} rows)")
    return rows


def global_distinctness(d):
    tail = d["u_prob32"] > U_CUT
    n_distinct_prob = int(len(np.unique(d["score_float32_production"][tail].astype(np.float32))))
    n_distinct_delta = int(len(np.unique(d["delta"][tail])))
    n_distinct_ulogit = int(len(np.unique(d["u_logit"][tail])))
    return dict(n_distinct_prob32_u_gt_5p5=n_distinct_prob,
                n_distinct_delta_u_gt_5p5=n_distinct_delta,
                n_distinct_u_logit_u_gt_5p5=n_distinct_ulogit,
                n_events_u_gt_5p5=int(tail.sum()))


def quantify_u35_u45(d):
    out = {}
    for thr in (3.5, 4.5):
        k_at_thr = 10.0 ** (24 * np.log10(2.0) - thr)
        n_events = int((d["u_prob32"] > thr).sum())
        n_events_signal = int(((d["u_prob32"] > thr) & (d["process_label"] == "signal")).sum())
        n_events_bg = int(((d["u_prob32"] > thr) & (d["process_label"] != "signal")).sum())
        n_distinct = int(len(np.unique(d["score_float32_production"][d["u_prob32"] > thr].astype(np.float32))))
        out[str(thr)] = dict(
            k_lattice_steps_remaining_to_1=k_at_thr,
            n_events_above_threshold=n_events,
            n_events_above_threshold_signal=n_events_signal,
            n_events_above_threshold_background=n_events_bg,
            n_distinct_float32_values_above_threshold=n_distinct,
            mean_events_per_distinct_value=(n_events / n_distinct) if n_distinct else None,
        )
    return out


def make_figures(d, pops):
    is_signal = pops["signal"]
    is_bg = pops["all_background"]

    u_signal = d["u_prob32"][is_signal]
    u_bg = d["u_prob32"][is_bg]
    ul_signal = d["u_logit"][is_signal]
    ul_bg = d["u_logit"][is_bg]

    finite_u_signal = np.isfinite(u_signal)
    n_inf_signal = int((~finite_u_signal).sum())

    # ---- u_prob32 histogram (tail) ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bin_edges = np.arange(U_CUT, 7.35, 0.02)
    tail_signal = u_signal[finite_u_signal & (u_signal > U_CUT)]
    tail_bg = u_bg[np.isfinite(u_bg) & (u_bg > U_CUT)]
    ax.hist(tail_signal, bins=bin_edges, color="#2b6cb0", alpha=0.85,
            label=f"signal (n={len(tail_signal)}, finite, u>{U_CUT})")
    if len(tail_bg):
        ax.hist(tail_bg, bins=bin_edges, color="#c53030", alpha=0.7, label=f"all background (n={len(tail_bg)})")
    else:
        ax.text(0.98, 0.6, "all background: 0 events with u > 5.5", transform=ax.transAxes,
                color="#c53030", fontsize=9, ha="right")
    ax.axvline(6.9237, color="#805ad5", linestyle="--", linewidth=1.2, label="u=6.9237 (k=2)")
    ax.axvline(6.6227, color="#dd6b20", linestyle="--", linewidth=1.2, label="u=6.6227 (k=4)")
    ax.set_xlabel(r"$u_{\rm prob32} = -\log_{10}(1-\mathrm{score}_{f32})$")
    ax.set_ylabel("events / 0.02")
    ax.set_title(f"Governing SPA-Net 10M, full 400k cohort\n({n_inf_signal} signal events at exact float32 1.0 not shown)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(U_CUT, 7.35)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figures/u_prob32_tail_10M.png", dpi=160)
    plt.close(fig)

    # ---- u_logit histogram (tail) ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    tail_signal_ul = ul_signal[ul_signal > U_CUT]
    tail_bg_ul = ul_bg[ul_bg > U_CUT]
    bin_edges_ul = np.linspace(U_CUT, max(tail_signal_ul.max(), U_CUT + 0.1), 120)
    ax.hist(tail_signal_ul, bins=bin_edges_ul, color="#2b6cb0", alpha=0.85,
            label=f"signal (n={len(tail_signal_ul)})")
    if len(tail_bg_ul):
        ax.hist(tail_bg_ul, bins=bin_edges_ul, color="#c53030", alpha=0.7, label=f"all background (n={len(tail_bg_ul)})")
    ax.set_xlabel(r"$u_{\rm logit} = \mathrm{softplus}(\Delta)/\ln(10)$  (float64, from raw logits)")
    ax.set_ylabel("events / bin")
    ax.set_title("Governing SPA-Net 10M: logit-derived u is continuous (no FP32 comb)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figures/u_logit_tail_10M.png", dpi=160)
    plt.close(fig)

    # ---- scatter u_prob32 vs u_logit ----
    fig, ax = plt.subplots(figsize=(7, 7))
    sel = (u_signal > U_CUT - 0.5) & finite_u_signal
    ax.scatter(ul_signal[sel], u_signal[sel], s=6, alpha=0.35, color="#2b6cb0", label="signal (finite score32)")
    inf_sel = is_signal & (~np.isfinite(d["u_prob32"]))
    if inf_sel.sum():
        y_top = u_signal[finite_u_signal].max() + 0.3 if finite_u_signal.any() else 8.0
        ax.scatter(d["u_logit"][inf_sel], np.full(inf_sel.sum(), y_top), s=10, alpha=0.5,
                   color="#805ad5", marker="^", label=f"score32==1.0 exactly (n={inf_sel.sum()}), plotted at u_prob32={y_top:.2f}")
    lims = [U_CUT - 0.5, max(ul_signal[sel].max() if sel.any() else 8, 8)]
    ax.plot(lims, lims, color="#666666", linestyle=":", linewidth=1, label="y=x")
    ax.set_xlabel(r"$u_{\rm logit}$ (float64, from raw logits)")
    ax.set_ylabel(r"$u_{\rm prob32}$ (from stored float32 score)")
    ax.set_title("u_prob32 vs u_logit: the FP32 steps are a display/storage artifact,\nnot present in the underlying logit margin")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figures/u_prob32_vs_u_logit.png", dpi=160)
    plt.close(fig)

    # ---- Delta distribution grouped by FP32 probability bin (k) ----
    score32 = d["score_float32_production"][is_signal].astype(np.float32)
    delta = d["delta"][is_signal]
    tail_mask = d["u_prob32"][is_signal] > U_CUT
    uniq_vals = np.unique(score32[tail_mask])
    uniq_vals = uniq_vals[np.argsort(-uniq_vals)][:12]  # the 12 largest-probability bins

    fig, ax = plt.subplots(figsize=(10, 6))
    data_for_box = []
    labels = []
    for val in uniq_vals:
        sel = score32 == val
        data_for_box.append(delta[sel])
        k = 0 if val >= np.float32(1.0) else int(round((1.0 - float(val)) / SPACING))
        labels.append(f"k={k}\n(n={int(sel.sum())})")
    ax.boxplot(data_for_box, labels=labels, showfliers=True, widths=0.6)
    ax.set_ylabel(r"$\Delta = z_{\rm signal} - z_{\rm background}$ (float64, raw logits)")
    ax.set_title("Delta spread within each FP32-tied probability bin (signal, u>5.5)\nevents sharing one float32 score remain spread out in logit space")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figures/delta_by_fp32_bin.png", dpi=160)
    plt.close(fig)

    print("wrote figures/u_prob32_tail_10M.png, u_logit_tail_10M.png, "
          "u_prob32_vs_u_logit.png, delta_by_fp32_bin.png")


def main():
    d = load()
    pops = populations(d)

    census_rows = build_unique_value_census(d, pops)
    lattice_rows = build_lattice_bin_table(d, pops)
    spikes = spike_counts(d, pops)
    bin_summary_rows = build_fp32_vs_logit_bin_summary(d, pops)
    distinctness = global_distinctness(d)
    u35_u45 = quantify_u35_u45(d)
    make_figures(d, pops)

    print("\nSpike-focus exact counts:")
    for name, v in spikes.items():
        print(f"  {name:16s} k=2: {v['k2_u6p9237']:5d}  k=4: {v['k4_u6p6227']:5d}  exact1.0: {v['exact_1p0']:5d}")

    print("\nGlobal distinctness (u>5.5):", json.dumps(distinctness, indent=2))
    print("\nu>3.5 / u>4.5 quantification:", json.dumps(u35_u45, indent=2))

    summary = dict(
        n_total=len(d["row_index"]),
        n_signal=int(pops["signal"].sum()), n_qcd=int(pops["qcd"].sum()),
        n_ttbar=int(pops["ttbar"].sum()), n_all_background=int(pops["all_background"].sum()),
        u_cut=U_CUT,
        spike_counts=spikes,
        global_distinctness=distinctness,
        u35_u45_quantification=u35_u45,
    )
    with open(f"{OUT_DIR}/work/census_summary_v2.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {OUT_DIR}/work/census_summary_v2.json")


if __name__ == "__main__":
    main()
