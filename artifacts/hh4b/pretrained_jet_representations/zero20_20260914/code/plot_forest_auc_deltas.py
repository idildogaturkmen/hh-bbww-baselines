#!/usr/bin/env python3
"""Forest plot of the final 10k-bootstrap paired AUC differences:
  ParT20 - Native2M
  ZERO20 - Native2M
  ZERO20 - ParT20
with 95% CIs. Reads metrics/paired_statistics.json (already-assembled,
n_boot=10000 point estimates and CIs -- no bootstrap run here). A vertical
line at both 0 and the pre-registered +/-0.005 practical-effect floor
(PREREGISTRATION.md Sec.7.2) is drawn so statistical significance (CI
excludes zero) and practical significance (|delta|>=0.005) are visually
separable, not conflated -- the explicit point requested for the AUC
interpretation of this result."""
import argparse
import json

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

AUC_FLOOR = 0.005
PAIR_ORDER = ["ParT20_minus_native2M", "ZERO20_minus_native2M", "ZERO20_minus_ParT20"]
PAIR_LABEL = {"ParT20_minus_native2M": "ParT20 2M − Native 2M",
              "ZERO20_minus_native2M": "ZERO20 2M − Native 2M",
              "ZERO20_minus_ParT20": "ZERO20 2M − ParT20 2M"}
COLOR = {"ParT20_minus_native2M": "#DD8452", "ZERO20_minus_native2M": "#8172B2",
         "ZERO20_minus_ParT20": "#55A868"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired-statistics-json", required=True)
    ap.add_argument("--out-svg", required=True)
    args = ap.parse_args()

    with open(args.paired_statistics_json) as f:
        pw = json.load(f)

    present = [p for p in PAIR_ORDER if p in pw]
    fig, ax = plt.subplots(figsize=(7.5, 0.9 * len(present) + 1.6))

    for i, pair in enumerate(present):
        da = pw[pair]["delta_auc"]
        y = len(present) - i
        obs = da["observed_delta_test_minus_control"]
        lo, hi = da["ci_low"], da["ci_high"]
        ax.plot([lo, hi], [y, y], color=COLOR[pair], linewidth=2)
        ax.plot(obs, y, "o", color=COLOR[pair], markersize=7)
        sig = "sig." if da["excludes_zero"] else "n.s."
        prac = "practical" if abs(obs) >= AUC_FLOOR else "below 0.005 floor"
        ax.annotate(f"{obs:+.5f}  [{lo:+.5f}, {hi:+.5f}]  ({sig}, {prac})",
                    (max(hi, obs) + 0.001, y), va="center", fontsize=8)

    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="-")
    ax.axvline(AUC_FLOOR, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(-AUC_FLOOR, color="gray", linewidth=0.8, linestyle="--")
    ax.text(AUC_FLOOR, len(present) + 0.6, "+0.005 floor", fontsize=7, color="gray", ha="center")
    ax.text(-AUC_FLOOR, len(present) + 0.6, "−0.005 floor", fontsize=7, color="gray", ha="center")

    ax.set_yticks([len(present) - i for i in range(len(present))])
    ax.set_yticklabels([PAIR_LABEL[p] for p in present])
    ax.set_xlabel("paired ΔAUC (all-background), 95% CI")
    ax.set_title("Final 10k-bootstrap paired AUC differences")
    ax.set_xlim(-0.032, 0.032)
    ax.set_ylim(0.3, len(present) + 1.2)
    ax.grid(True, axis="x", alpha=0.3, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out_svg, format="svg")
    plt.close(fig)
    print(f"wrote {args.out_svg}")


if __name__ == "__main__":
    main()
