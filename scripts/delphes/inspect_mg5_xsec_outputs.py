#!/usr/bin/env python3

import gzip
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd


STORE = Path(os.environ["HH4B_STORE"])
REPO = Path(os.environ["HH4B_REPO"])

BASE = STORE / "mg5_xsec_checks_v2"
SUMMARY = STORE / "metadata/mg5_xsec_sanity_checks_v2.csv"

OUTDIR = REPO / "outputs/tables/mg5_xsec_sanity_checks_2026_07_08"
OUTDIR.mkdir(parents=True, exist_ok=True)


def parse_higgs_decay_block(param_card):
    """Parse DECAY 25 block and return width + BRs."""
    width = None
    brs = []
    in_higgs_decay = False

    for raw in param_card.read_text(errors="ignore").splitlines():
        line = raw.strip()

        if not line or line.startswith("#"):
            continue

        upper = line.upper()

        if upper.startswith("DECAY"):
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "25":
                in_higgs_decay = True
                try:
                    width = float(parts[2])
                except Exception:
                    width = None
                continue
            elif in_higgs_decay:
                break

        if in_higgs_decay:
            if upper.startswith("BLOCK"):
                break

            no_comment = line.split("#")[0].strip()
            parts = no_comment.split()
            if len(parts) >= 4:
                try:
                    br = float(parts[0])
                    nda = int(parts[1])
                    ids = tuple(int(x) for x in parts[2 : 2 + nda])
                    comment = raw.split("#", 1)[1].strip() if "#" in raw else ""
                    brs.append({"br": br, "ids": ids, "comment": comment})
                except Exception:
                    pass

    hbb = None
    for row in brs:
        if sorted(row["ids"]) == [-5, 5]:
            hbb = row["br"]

    return width, hbb, brs


def find_lhe(sample_dir):
    candidates = sorted((sample_dir / "Events").glob("*/unweighted_events.lhe*"))
    return candidates[-1] if candidates else None


def open_lhe(path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="ignore")
    return open(path, "rt", errors="ignore")


def count_lhe_final_states(lhe_path, max_events=300):
    """Count final-state b quarks and final-state Higgs bosons in LHE events."""
    counts = []
    in_event = False
    need_header = False
    current = None

    with open_lhe(lhe_path) as f:
        for raw in f:
            line = raw.strip()

            if line == "<event>":
                in_event = True
                need_header = True
                current = Counter()
                continue

            if line == "</event>":
                if current is not None:
                    counts.append(dict(current))
                    if len(counts) >= max_events:
                        break
                in_event = False
                current = None
                continue

            if not in_event:
                continue

            if need_header:
                need_header = False
                continue

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                pid = int(parts[0])
                status = int(parts[1])
            except Exception:
                continue

            if status == 1:
                if abs(pid) == 5:
                    current["final_b_or_bbar"] += 1
                if pid == 25:
                    current["final_higgs"] += 1
                if abs(pid) in [1, 2, 3, 4, 21]:
                    current["final_light_parton_or_gluon"] += 1

    out = {}
    for key in ["final_b_or_bbar", "final_higgs", "final_light_parton_or_gluon"]:
        out[key] = Counter(int(ev.get(key, 0)) for ev in counts)

    out["n_events_checked"] = len(counts)
    return out


def get_xsec(df, sample):
    return float(df.loc[df["sample"] == sample, "xsec_fb"].iloc[0])


def main():
    df = pd.read_csv(SUMMARY)

    print("\n=== Cross sections ===")
    print(df[["sample", "xsec_pb", "xsec_fb"]].to_string(index=False))

    # Param-card diagnostics
    param_rows = []
    for sample in df["sample"]:
        sample_dir = BASE / sample
        param = sample_dir / "Cards/param_card.dat"
        if not param.exists():
            continue

        width, hbb, brs = parse_higgs_decay_block(param)

        param_rows.append(
            {
                "sample": sample,
                "higgs_width": width,
                "param_card_BR_h_bb": hbb,
                "n_higgs_decay_modes_listed": len(brs),
            }
        )

    param_df = pd.DataFrame(param_rows)

    print("\n=== Param-card Higgs decay diagnostics ===")
    print(param_df.to_string(index=False))

    # LHE final-state diagnostics
    lhe_rows = []
    for sample in df["sample"]:
        sample_dir = BASE / sample
        lhe = find_lhe(sample_dir)
        if lhe is None:
            continue

        counts = count_lhe_final_states(lhe)

        lhe_rows.append(
            {
                "sample": sample,
                "lhe_file": str(lhe),
                "n_events_checked": counts["n_events_checked"],
                "final_b_or_bbar_counts": dict(counts["final_b_or_bbar"]),
                "final_higgs_counts": dict(counts["final_higgs"]),
                "final_light_parton_or_gluon_counts": dict(counts["final_light_parton_or_gluon"]),
            }
        )

    lhe_df = pd.DataFrame(lhe_rows)

    print("\n=== LHE final-state diagnostics ===")
    print(lhe_df[["sample", "n_events_checked", "final_b_or_bbar_counts", "final_higgs_counts"]].to_string(index=False))

    # Ratio diagnostics
    hbb_over_h = get_xsec(df, "check_ggf_h_bb_heft") / get_xsec(df, "check_ggf_h_stable_heft")
    gg_hh4b_over_hh = get_xsec(df, "check_ggf_hh_4b_heft") / get_xsec(df, "check_ggf_hh_stable_heft")
    vbf_4b_over_stable = get_xsec(df, "check_vbf_hhjj_4b_sm") / get_xsec(df, "check_vbf_hhjj_stable_sm")

    ratio_df = pd.DataFrame(
        [
            {
                "quantity": "HEFT H→bb / H stable",
                "value": hbb_over_h,
                "comparison": "effective MG5 HEFT BR(H→bb)",
            },
            {
                "quantity": "HEFT ggF HH→4b / HH stable",
                "value": gg_hh4b_over_hh,
                "comparison": "should be close to HEFT BR(H→bb)^2",
            },
            {
                "quantity": "SM VBF HHjj→4b / HHjj stable",
                "value": vbf_4b_over_stable,
                "comparison": "should be checked against SM-card BR or LHE final states",
            },
            {
                "quantity": "HEFT BR(H→bb)^2 from single-H ratio",
                "value": hbb_over_h**2,
                "comparison": "explains ggF HH→4b ratio if close",
            },
            {
                "quantity": "official-reference 0.5824^2",
                "value": 0.5824**2,
                "comparison": "modern SM BR(H→bb)^2 reference used earlier",
            },
        ]
    )

    print("\n=== Ratio diagnostics ===")
    print(ratio_df.to_string(index=False))

    # Save outputs
    param_df.to_csv(OUTDIR / "mg5_param_card_higgs_decay_diagnostics.csv", index=False)
    lhe_df.to_csv(OUTDIR / "mg5_lhe_final_state_diagnostics.csv", index=False)
    ratio_df.to_csv(OUTDIR / "mg5_ratio_diagnostics.csv", index=False)

    (OUTDIR / "mg5_param_card_higgs_decay_diagnostics.md").write_text(param_df.to_markdown(index=False) + "\n")
    (OUTDIR / "mg5_lhe_final_state_diagnostics.md").write_text(lhe_df.to_markdown(index=False) + "\n")
    (OUTDIR / "mg5_ratio_diagnostics.md").write_text(ratio_df.to_markdown(index=False) + "\n")

    print(f"\nWrote diagnostics to: {OUTDIR}")


if __name__ == "__main__":
    main()
