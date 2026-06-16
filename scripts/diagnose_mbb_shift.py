import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NPZ_DIR = Path("outputs/hbb_npz")
OUTDIR = Path("outputs/mbb_diagnostics")
PLOT_DIR = Path("outputs/plots/mbb_diagnostics")
M_H = 125.0

OUTDIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


METHOD_LABELS = {
    "matched": "truth-matched H->bb jets",
    "top2_btag": "top-2 b-tag jets",
    "closest_mass": "m_jj closest to 125 GeV",
    "combined_btag_mass": "combined b-tag + mass",
}


def load_all_splits():
    jets_list, mask_list, labels_list, split_list = [], [], [], []

    for split in ["train", "val", "test"]:
        data = np.load(NPZ_DIR / f"{split}.npz")
        jets = data["jets"]
        jets_list.append(jets)
        mask_list.append(data["mask"])
        labels_list.append(data["labels"])
        split_list.extend([split] * len(jets))

    return (
        np.concatenate(jets_list, axis=0),
        np.concatenate(mask_list, axis=0),
        np.concatenate(labels_list, axis=0),
        np.array(split_list),
    )


def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    while dphi > np.pi:
        dphi -= 2 * np.pi
    while dphi <= -np.pi:
        dphi += 2 * np.pi
    return dphi


def delta_r(j1, j2):
    return np.sqrt((j1[1] - j2[1]) ** 2 + delta_phi(j1[2], j2[2]) ** 2)


def invariant_mass(j1, j2):
    pt1, eta1, phi1, m1 = j1[:4]
    pt2, eta2, phi2, m2 = j2[:4]

    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    e1 = np.sqrt(px1**2 + py1**2 + pz1**2 + m1**2)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    e2 = np.sqrt(px2**2 + py2**2 + pz2**2 + m2**2)

    e = e1 + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    m2_tot = e**2 - px**2 - py**2 - pz**2
    return float(np.sqrt(max(m2_tot, 0.0)))


def pair_correct(pred, truth):
    return set(pred) == set(truth)


def top2_btag(jets, mask):
    real = np.where(mask > 0)[0]
    btags = jets[real, 4]
    return tuple(real[np.argsort(btags)[-2:]])


def closest_mass(jets, mask):
    real = np.where(mask > 0)[0]
    best_pair = None
    best_score = float("inf")

    for i, j in itertools.combinations(real, 2):
        mbb = invariant_mass(jets[i], jets[j])
        score = abs(mbb - M_H)
        if score < best_score:
            best_score = score
            best_pair = (i, j)

    return best_pair


def combined_btag_mass(jets, mask):
    real = np.where(mask > 0)[0]
    best_pair = None
    best_score = float("inf")

    for i, j in itertools.combinations(real, 2):
        mbb = invariant_mass(jets[i], jets[j])
        btag_sum = jets[i, 4] + jets[j, 4]
        score = abs(mbb - M_H) - 20.0 * btag_sum
        if score < best_score:
            best_score = score
            best_pair = (i, j)

    return best_pair


def summarize_values(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p16": float(np.percentile(values, 16)),
        "p84": float(np.percentile(values, 84)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def save_hist(values, bins, xlabel, title, filename, density=False):
    plt.figure()
    plt.hist(values, bins=bins, histtype="step", linewidth=1.6, density=density)
    plt.axvline(M_H, color="black", linestyle="--", linewidth=1.0, label="125 GeV")
    plt.xlabel(xlabel)
    plt.ylabel("Normalized events" if density else "Events")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=200)
    plt.close()


def save_overlay(values_by_name, bins, xlabel, title, filename, draw_higgs_line=True):
    plt.figure()
    for name, values in values_by_name.items():
        plt.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=1.5,
            density=True,
            label=METHOD_LABELS.get(name, name),
        )
    if draw_higgs_line:
        plt.axvline(M_H, color="black", linestyle="--", linewidth=1.0, label="125 GeV")
    plt.xlabel(xlabel)
    plt.ylabel("Normalized events")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=200)
    plt.close()


def save_hexbin(x, y, xlabel, ylabel, title, filename, gridsize=45):
    plt.figure()
    hb = plt.hexbin(x, y, gridsize=gridsize, mincnt=1, cmap="viridis")
    plt.axhline(M_H, color="white", linestyle="--", linewidth=1.0)
    plt.colorbar(hb, label="Events")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=200)
    plt.close()


def save_profile(df, x_col, y_col, bins, xlabel, title, filename):
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (df[x_col] >= lo) & (df[x_col] < hi)
        if not np.any(sel):
            continue
        vals = df.loc[sel, y_col].to_numpy()
        rows.append(
            {
                "x_mid": 0.5 * (lo + hi),
                "mean": np.mean(vals),
                "median": np.median(vals),
                "p16": np.percentile(vals, 16),
                "p84": np.percentile(vals, 84),
            }
        )

    prof = pd.DataFrame(rows)
    if prof.empty:
        return prof

    plt.figure()
    plt.fill_between(prof["x_mid"], prof["p16"], prof["p84"], alpha=0.25, label="16-84%")
    plt.plot(prof["x_mid"], prof["median"], marker="o", label="median")
    plt.axhline(M_H, color="black", linestyle="--", linewidth=1.0, label="125 GeV")
    plt.xlabel(xlabel)
    plt.ylabel("Matched $m_{bb}$ [GeV]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=200)
    plt.close()

    return prof


def build_event_table(jets, mask, labels, splits):
    methods = {
        "top2_btag": top2_btag,
        "closest_mass": closest_mass,
        "combined_btag_mass": combined_btag_mass,
    }

    rows = []
    method_mbb = {"matched": []}
    method_correct = {name: [] for name in methods}
    for name in methods:
        method_mbb[name] = []

    matched_jet_rows = []

    for ev in range(len(labels)):
        truth = tuple(int(x) for x in labels[ev])
        i, j = truth
        ji = jets[ev, i]
        jj = jets[ev, j]

        mbb = invariant_mass(ji, jj)
        dr = delta_r(ji, jj)
        pts = np.array([ji[0], jj[0]], dtype=float)
        masses = np.array([ji[3], jj[3]], dtype=float)
        btags = np.array([ji[4], jj[4]], dtype=float)
        real_n = int(np.sum(mask[ev] > 0))

        row = {
            "event": ev,
            "split": splits[ev],
            "n_jets": real_n,
            "matched_i": i,
            "matched_j": j,
            "matched_mbb": mbb,
            "matched_deltaR": dr,
            "matched_min_pt": float(np.min(pts)),
            "matched_max_pt": float(np.max(pts)),
            "matched_mean_pt": float(np.mean(pts)),
            "matched_min_mass": float(np.min(masses)),
            "matched_max_mass": float(np.max(masses)),
            "matched_mean_mass": float(np.mean(masses)),
            "matched_mean_btag": float(np.mean(btags)),
        }
        method_mbb["matched"].append(mbb)

        ordered = sorted([ji, jj], key=lambda x: x[0])
        for rank, jet in zip(["lower_pt", "higher_pt"], ordered):
            matched_jet_rows.append(
                {
                    "event": ev,
                    "split": splits[ev],
                    "pt_rank": rank,
                    "pt": float(jet[0]),
                    "eta": float(jet[1]),
                    "mass": float(jet[3]),
                    "btag": float(jet[4]),
                    "btagPhys": float(jet[5]),
                }
            )

        for name, fn in methods.items():
            pred = fn(jets[ev], mask[ev])
            pred_mbb = invariant_mass(jets[ev, pred[0]], jets[ev, pred[1]])
            correct = pair_correct(pred, truth)
            row[f"{name}_i"] = int(pred[0])
            row[f"{name}_j"] = int(pred[1])
            row[f"{name}_mbb"] = pred_mbb
            row[f"{name}_correct"] = bool(correct)
            method_mbb[name].append(pred_mbb)
            method_correct[name].append(correct)

        rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(matched_jet_rows), method_mbb, method_correct


def write_markdown_note(summary):
    note = f"""# Matched m_bb diagnostic note

This diagnostic uses the existing fixed-size HH -> bbWW reconstruction arrays in `outputs/hbb_npz/*.npz`. The truth-matched H -> bb pair is defined by the two AK4 jets matched to the generator-level H -> bb b quarks during preprocessing.

## Main observation

The truth-matched AK4 dijet mass is below the nominal Higgs mass:

- Mean matched m_bb: {summary["matched_mbb"]["mean"]:.2f} GeV
- Median matched m_bb: {summary["matched_mbb"]["median"]:.2f} GeV
- 16-84% interval: {summary["matched_mbb"]["p16"]:.2f} to {summary["matched_mbb"]["p84"]:.2f} GeV

This should not be interpreted as proof that the matched pair is wrong. It means that the reconstructed AK4 four-vectors in this simplified COLLIDE sample do not, by themselves, recover the full H -> bb four-momentum at 125 GeV.

## Comparison to simple pair choices

The `closest_mass` pair has an m_jj distribution near 125 GeV by construction, but its pair-assignment accuracy is low in the existing baseline. Therefore, a mass peak near 125 GeV is not sufficient evidence that the correct H -> bb jets were selected. Conversely, truth-matched jets can have m_bb below 125 GeV because of detector/reconstruction effects and b-jet response.

## Plausible causes to test further

The following mechanisms are physically plausible, and are also consistent with how CMS treats H -> bb reconstruction, but this diagnostic alone does not prove their relative importance:

- AK4 out-of-cone radiation: b jets can spread beyond R = 0.4, so a finite-radius jet may not capture all H -> bb energy.
- Semileptonic b-hadron decays: neutrinos inside b jets escape detection and reduce reconstructed jet energy.
- Detector/reconstruction-level response: the saved COLLIDE reconstructed jet four-vectors may not include CMS-style b-jet energy regression.
- Matching/reconstruction imperfections: nearest-jet matching can choose the right reconstructed jet object while still missing part of the b-quark shower energy.
- Dataset-specific calibration effects: COLLIDE is a simplified ML dataset, not a full CMS Run 2/Run 3 analysis chain.

## Outputs

Plots are saved in `outputs/plots/mbb_diagnostics/`. Event-level and matched-jet diagnostic tables are saved in `outputs/mbb_diagnostics/`.
"""
    (OUTDIR / "mbb_shift_note.md").write_text(note)


def main():
    jets, mask, labels, splits = load_all_splits()
    events, matched_jets, method_mbb, method_correct = build_event_table(jets, mask, labels, splits)

    events.to_csv(OUTDIR / "mbb_event_diagnostics.csv", index=False)
    matched_jets.to_csv(OUTDIR / "matched_jet_diagnostics.csv", index=False)

    summary = {
        "n_events": int(len(events)),
        "matched_mbb": summarize_values(events["matched_mbb"]),
        "matched_deltaR": summarize_values(events["matched_deltaR"]),
        "matched_min_pt": summarize_values(events["matched_min_pt"]),
        "matched_mean_pt": summarize_values(events["matched_mean_pt"]),
        "matched_jet_pt": summarize_values(matched_jets["pt"]),
        "matched_jet_mass": summarize_values(matched_jets["mass"]),
        "method_accuracy": {name: float(np.mean(vals)) for name, vals in method_correct.items()},
        "method_mbb": {name: summarize_values(vals) for name, vals in method_mbb.items()},
    }

    for lo, hi in [(90, 140), (100, 150), (110, 140)]:
        summary[f"matched_fraction_mbb_{lo}_{hi}"] = float(
            np.mean((events["matched_mbb"] >= lo) & (events["matched_mbb"] <= hi))
        )

    with open(OUTDIR / "mbb_diagnostic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(
        [
            {"quantity": key, **value}
            for key, value in summary.items()
            if isinstance(value, dict) and "mean" in value
        ]
    ).to_csv(OUTDIR / "mbb_distribution_summary.csv", index=False)

    save_hist(
        events["matched_mbb"],
        bins=np.linspace(0, 250, 80),
        xlabel="Matched $m_{bb}$ [GeV]",
        title="Truth-matched H->bb AK4 dijet mass",
        filename="matched_mbb.png",
    )

    save_overlay(
        method_mbb,
        bins=np.linspace(0, 250, 80),
        xlabel="$m_{jj}$ [GeV]",
        title="Dijet mass comparison for H->bb pair choices",
        filename="mbb_method_comparison.png",
    )

    save_hexbin(
        events["top2_btag_mbb"],
        events["matched_mbb"],
        xlabel="Top-2 b-tag $m_{bb}$ [GeV]",
        ylabel="Matched $m_{bb}$ [GeV]",
        title="Matched vs top-2-btag dijet mass",
        filename="matched_vs_top2_btag_mbb.png",
    )

    save_hexbin(
        events["closest_mass_mbb"],
        events["matched_mbb"],
        xlabel="Closest-to-125 $m_{jj}$ [GeV]",
        ylabel="Matched $m_{bb}$ [GeV]",
        title="Matched vs closest-to-125 dijet mass",
        filename="matched_vs_closest_mass_mbb.png",
    )

    save_hexbin(
        events["matched_deltaR"],
        events["matched_mbb"],
        xlabel="Matched $\\Delta R_{bb}$",
        ylabel="Matched $m_{bb}$ [GeV]",
        title="Matched m_bb vs DeltaR_bb",
        filename="matched_mbb_vs_deltaR.png",
    )

    save_hexbin(
        events["matched_min_pt"],
        events["matched_mbb"],
        xlabel="Lower matched-jet $p_T$ [GeV]",
        ylabel="Matched $m_{bb}$ [GeV]",
        title="Matched m_bb vs lower matched-jet pT",
        filename="matched_mbb_vs_min_matched_pt.png",
    )

    save_hexbin(
        events["matched_mean_pt"],
        events["matched_mbb"],
        xlabel="Mean matched-jet $p_T$ [GeV]",
        ylabel="Matched $m_{bb}$ [GeV]",
        title="Matched m_bb vs mean matched-jet pT",
        filename="matched_mbb_vs_mean_matched_pt.png",
    )

    save_overlay(
        {
            "lower_pt": matched_jets.loc[matched_jets["pt_rank"] == "lower_pt", "pt"],
            "higher_pt": matched_jets.loc[matched_jets["pt_rank"] == "higher_pt", "pt"],
        },
        bins=np.linspace(0, np.percentile(matched_jets["pt"], 99), 70),
        xlabel="Matched jet $p_T$ [GeV]",
        title="Matched H->bb jet pT distributions",
        filename="matched_jet_pt.png",
        draw_higgs_line=False,
    )

    save_overlay(
        {
            "lower_pt": matched_jets.loc[matched_jets["pt_rank"] == "lower_pt", "mass"],
            "higher_pt": matched_jets.loc[matched_jets["pt_rank"] == "higher_pt", "mass"],
        },
        bins=np.linspace(0, np.percentile(matched_jets["mass"], 99), 70),
        xlabel="Matched jet mass [GeV]",
        title="Matched H->bb jet mass distributions",
        filename="matched_jet_mass.png",
        draw_higgs_line=False,
    )

    prof_dr = save_profile(
        events,
        x_col="matched_deltaR",
        y_col="matched_mbb",
        bins=np.linspace(0, np.percentile(events["matched_deltaR"], 99), 12),
        xlabel="Matched $\\Delta R_{bb}$",
        title="Matched m_bb profile vs DeltaR_bb",
        filename="matched_mbb_profile_vs_deltaR.png",
    )
    prof_pt = save_profile(
        events,
        x_col="matched_min_pt",
        y_col="matched_mbb",
        bins=np.linspace(0, np.percentile(events["matched_min_pt"], 99), 12),
        xlabel="Lower matched-jet $p_T$ [GeV]",
        title="Matched m_bb profile vs lower matched-jet pT",
        filename="matched_mbb_profile_vs_min_pt.png",
    )
    prof_dr.to_csv(OUTDIR / "matched_mbb_profile_vs_deltaR.csv", index=False)
    prof_pt.to_csv(OUTDIR / "matched_mbb_profile_vs_min_pt.csv", index=False)

    write_markdown_note(summary)

    print("Matched m_bb diagnostics")
    print("-----------------------")
    print(f"Events: {len(events)}")
    print(f"Mean matched m_bb:   {summary['matched_mbb']['mean']:.3f} GeV")
    print(f"Median matched m_bb: {summary['matched_mbb']['median']:.3f} GeV")
    print(f"16-84% interval:     {summary['matched_mbb']['p16']:.3f} - {summary['matched_mbb']['p84']:.3f} GeV")
    print("\nMethod m_jj means:")
    for name, vals in summary["method_mbb"].items():
        print(f"  {name:20s}: mean={vals['mean']:.3f}, median={vals['median']:.3f}")
    print("\nMethod accuracies:")
    for name, acc in summary["method_accuracy"].items():
        print(f"  {name:20s}: {acc:.4f}")
    print(f"\nSaved plots to {PLOT_DIR}")
    print(f"Saved tables and note to {OUTDIR}")


if __name__ == "__main__":
    main()
