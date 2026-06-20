import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

# Allow imports from scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from make_hbb_bjet_regression_dataset import (
    COLUMNS,
    FEATURE_NAMES,
    find_hh_files,
    event_from_batch,
    get_scalar,
    build_reco_jets,
    find_hbb_bquarks,
    match_bquarks_to_reco_jets,
    pf_id_to_row_mapping,
    constituent_features,
    lepton_dr_features,
)

from train_hbb_bjet_regression import (
    RegressionMLP,
    apply_standardizer,
    invariant_mass_two,
)


M_H = 125.0
EPS = 1e-8


def width68(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return 0.5 * (np.percentile(values, 84) - np.percentile(values, 16))


def summarize_mass(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p16": float("nan"),
            "p84": float("nan"),
            "width68": float("nan"),
            "frac_90_140": float("nan"),
            "frac_100_150": float("nan"),
            "median_abs_offset_from_125": float("nan"),
        }

    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p16": float(np.percentile(values, 16)),
        "p84": float(np.percentile(values, 84)),
        "width68": width68(values),
        "frac_90_140": float(np.mean((values >= 90.0) & (values <= 140.0))),
        "frac_100_150": float(np.mean((values >= 100.0) & (values <= 150.0))),
        "median_abs_offset_from_125": float(abs(np.median(values) - M_H)),
    }


def load_dnn(best_run, hidden, dropout, cpu=False):
    best_run = Path(best_run)
    model_path = best_run / "outputs" / "dnn_model.pt"
    scaler_path = best_run / "outputs" / "scaler.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Could not find model: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Could not find scaler: {scaler_path}")

    with open(scaler_path) as f:
        scaler = json.load(f)

    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    scaler_feature_names = scaler.get("feature_names", FEATURE_NAMES)

    if list(scaler_feature_names) != list(FEATURE_NAMES):
        print("WARNING: scaler feature names differ from FEATURE_NAMES.")
        print("Using FEATURE_NAMES order from make_hbb_bjet_regression_dataset.py.")

    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    model = RegressionMLP(n_features=len(mean), hidden=hidden, dropout=dropout).to(device)

    # PyTorch 2.6 changed torch.load default to weights_only=True.
    try:
        state = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # For older PyTorch versions that do not support weights_only.
        state = torch.load(model_path, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.eval()

    return model, mean, std, device


def predict_dnn(model, X_std, device, batch_size=8192): # Predict in batches to avoid GPU memory issues with large numbers of candidate jets.
    '''
    Predict the DNN output for the given standardized features, in batches.
    Returns the predicted corrections as a numpy array.
    '''
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_std), batch_size):
            xb = torch.tensor(X_std[start:start + batch_size], dtype=torch.float32, device=device)
            pred = model(xb).detach().cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds).astype(np.float64)


def make_candidate_feature_matrix(event, reco_jets):
    """
    Build the same 62-feature input vector for every selected reco AK4 jet in an event.

    This intentionally reuses the same engineered feature names used to train the
    b-jet response DNN.
    """
    if len(reco_jets) == 0:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)

    id_to_row = pf_id_to_row_mapping(event)

    pt_order = sorted(range(len(reco_jets)), key=lambda i: reco_jets[i]["pt"], reverse=True)
    btag_order = sorted(range(len(reco_jets)), key=lambda i: reco_jets[i]["btag"], reverse=True)

    pt_rank = {idx: rank for rank, idx in enumerate(pt_order)}
    btag_rank = {idx: rank for rank, idx in enumerate(btag_order)}

    ht_retained = float(sum(j["pt"] for j in reco_jets))
    n_reco = float(len(reco_jets))

    met = get_scalar(event, "FullReco_MET_MET", 0.0)
    puppimet = get_scalar(event, "FullReco_PUPPIMET_MET", 0.0)
    pv_z = get_scalar(event, "FullReco_PrimaryVertex_Z", 0.0)
    pv_sumpt2 = get_scalar(event, "FullReco_PrimaryVertex_SumPT2", 0.0)

    rows = []

    for local_idx, jet in enumerate(reco_jets):
        reco_pt = max(float(jet["pt"]), EPS)
        reco_mass = max(float(jet["mass"]), 0.0)

        feat = {
            "log_reco_pt": math.log(reco_pt),
            "reco_eta": float(jet["eta"]),
            "abs_reco_eta": abs(float(jet["eta"])),
            "sin_reco_phi": math.sin(float(jet["phi"])),
            "cos_reco_phi": math.cos(float(jet["phi"])),
            "log_reco_mass": math.log(max(reco_mass, EPS)),
            "reco_btag": float(jet["btag"]),
            "reco_btagPhys": float(jet["btagPhys"]),
            "reco_charge": float(jet["charge"]),
            "jet_rank_by_pt": float(pt_rank[local_idx]),
            "jet_rank_by_btag": float(btag_rank[local_idx]),
            "n_reco_jets_retained": n_reco,
            "ht_retained": ht_retained,
            "log_ht_retained": math.log(max(ht_retained, EPS)),
            "met": met,
            "log_met": math.log(max(met, EPS)),
            "puppimet": puppimet,
            "log_puppimet": math.log(max(puppimet, EPS)),
            "pv_z": pv_z,
            "log_pv_sumpt2": math.log(max(pv_sumpt2, EPS)),
        }

        feat.update(constituent_features(event, jet, id_to_row))
        feat.update(lepton_dr_features(event, jet))

        rows.append([float(feat.get(name, 0.0)) for name in FEATURE_NAMES])

    return np.asarray(rows, dtype=np.float32)


def corrected_mbb(j0, j1, c0=1.0, c1=1.0):
    '''
    Calculate the invariant mass of two jets after applying pt and mass corrections. 
    The corrections are applied as a simple multiplicative factor to the jet pt and mass, while keeping eta and phi unchanged. 
    This is a common approach when using a DNN to predict a jet energy correction factor.
    '''
    return invariant_mass_two(
        float(j0["pt"]) * c0,
        float(j0["eta"]),
        float(j0["phi"]),
        max(float(j0["mass"]), 0.0) * c0,
        float(j1["pt"]) * c1,
        float(j1["eta"]),
        float(j1["phi"]),
        max(float(j1["mass"]), 0.0) * c1,
    )


def choose_pair(pair_rows, selector):
    '''
    Given a list of candidate pairs (with their features) and a selector name, choose one pair according to the specified selector strategy.
    '''
    if not pair_rows:
        return None

    if selector == "top2_btag":
        return max(pair_rows, key=lambda r: (r["sum_btag"], -r["mbb_unc_absdiff"]))

    if selector == "closest_mass_uncorrected":
        return min(pair_rows, key=lambda r: (r["mbb_unc_absdiff"], -r["sum_btag"]))

    if selector == "closest_mass_corrected":
        return min(pair_rows, key=lambda r: (r["mbb_corr_absdiff"], -r["sum_btag"]))

    if selector == "btag_mass_uncorrected":
        return max(pair_rows, key=lambda r: (r["sum_btag"] - r["mbb_unc_absdiff"] / 50.0))

    if selector == "btag_mass_corrected":
        return max(pair_rows, key=lambda r: (r["sum_btag"] - r["mbb_corr_absdiff"] / 50.0))

    raise ValueError(f"Unknown selector: {selector}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-run", required=True)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--n-files", type=int, default=2)
    parser.add_argument("--max-events", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--min-reco-pt", type=float, default=20.0)
    parser.add_argument("--max-abs-eta", type=float, default=2.5)
    parser.add_argument("--max-reco-jets", type=int, default=-1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--outdir", default="outputs/pairing_with_bjet_correction")
    parser.add_argument("--plot-dir", default="outputs/plots/pairing_with_bjet_correction")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    plot_dir = Path(args.plot_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("Loading best DNN correction model...")
    model, mean, std, device = load_dnn(args.best_run, args.hidden, args.dropout, args.cpu)

    files = find_hh_files(args.n_files)
    print(f"Using {len(files)} HH->bbWW files")

    selectors = [
        "top2_btag",
        "closest_mass_uncorrected",
        "closest_mass_corrected",
        "btag_mass_uncorrected",
        "btag_mass_corrected",
    ]

    selected_rows = []
    all_pair_rows = []

    counters = {
        "raw_events": 0,
        "events_with_two_truth_b": 0,
        "events_with_at_least_two_reco_jets": 0,
        "events_with_truth_pair_label": 0,
    }

    stop = False

    for file_path in files:
        print(f"Reading {file_path}")
        pf = pq.ParquetFile(file_path)

        for batch in pf.iter_batches(columns=COLUMNS, batch_size=args.batch_size):
            cols = batch.to_pydict()
            n_rows = len(next(iter(cols.values())))

            for row_idx in range(n_rows):
                counters["raw_events"] += 1

                if args.max_events > 0 and counters["raw_events"] > args.max_events:
                    stop = True
                    break

                event = event_from_batch(cols, row_idx)
                event_id = counters["raw_events"] - 1

                reco_jets, n_raw_reco = build_reco_jets(
                    event,
                    max_reco_jets=args.max_reco_jets,
                    min_reco_pt=args.min_reco_pt,
                    max_abs_eta=args.max_abs_eta,
                )

                if len(reco_jets) < 2:
                    continue
                counters["events_with_at_least_two_reco_jets"] += 1 # Only consider events with at least 2 selected reco jets, since we can't form a pair otherwise.

                b_indices = find_hbb_bquarks(event) # Find indices of the two b-quarks from the H->bb decay at the generator level. This will be used to label the "truth pair" among the reco jet pairs.
                if len(b_indices) < 2:
                    continue
                counters["events_with_two_truth_b"] += 1

                truth_matches = match_bquarks_to_reco_jets(event, b_indices, reco_jets)
                if not all(m["matched"] for m in truth_matches):
                    continue

                truth_reco_indices = [int(m["jet"]["idx"]) for m in truth_matches]
                if len(set(truth_reco_indices)) != 2:
                    continue

                truth_pair = frozenset(truth_reco_indices)
                counters["events_with_truth_pair_label"] += 1

                X_raw = make_candidate_feature_matrix(event, reco_jets)
                X_std = apply_standardizer(X_raw, mean, std)
                pred = predict_dnn(model, X_std, device)
                corr = np.exp(pred)

                pair_rows = []

                for ia, ib in itertools.combinations(range(len(reco_jets)), 2):
                    ja = reco_jets[ia]
                    jb = reco_jets[ib]

                    mbb_unc = corrected_mbb(ja, jb, 1.0, 1.0)
                    mbb_corr = corrected_mbb(ja, jb, float(corr[ia]), float(corr[ib]))

                    pair_set = frozenset([int(ja["idx"]), int(jb["idx"])])
                    is_truth = pair_set == truth_pair

                    pair_row = {
                        "event_id": event_id,
                        "jet_a_idx": int(ja["idx"]),
                        "jet_b_idx": int(jb["idx"]),
                        "is_truth_pair": bool(is_truth),
                        "sum_btag": float(ja["btag"] + jb["btag"]),
                        "mbb_uncorrected": mbb_unc,
                        "mbb_corrected": mbb_corr,
                        "mbb_unc_absdiff": abs(mbb_unc - M_H),
                        "mbb_corr_absdiff": abs(mbb_corr - M_H),
                        "corr_a": float(corr[ia]),
                        "corr_b": float(corr[ib]),
                        "pt_a": float(ja["pt"]),
                        "pt_b": float(jb["pt"]),
                        "btag_a": float(ja["btag"]),
                        "btag_b": float(jb["btag"]),
                    }

                    pair_rows.append(pair_row)
                    all_pair_rows.append(pair_row)

                for selector in selectors:
                    chosen = choose_pair(pair_rows, selector)
                    if chosen is None:
                        continue

                    # Use corrected mass only for explicitly corrected selectors.
                    # Important: "uncorrected" also ends with "corrected", so do not use endswith().
                    corrected_selectors = {
                        "closest_mass_corrected",
                        "btag_mass_corrected",
                    }

                    selected_mbb = (
                        chosen["mbb_corrected"]
                        if selector in corrected_selectors
                        else chosen["mbb_uncorrected"]
                    )

                    selected_rows.append({
                        "event_id": event_id,
                        "selector": selector,
                        "correct": bool(chosen["is_truth_pair"]),
                        "selected_mbb": float(selected_mbb),
                        "selected_mbb_uncorrected": float(chosen["mbb_uncorrected"]),
                        "selected_mbb_corrected": float(chosen["mbb_corrected"]),
                        "sum_btag": float(chosen["sum_btag"]),
                    })

            if stop:
                break
        if stop:
            break

    selected_df = pd.DataFrame(selected_rows)
    pairs_df = pd.DataFrame(all_pair_rows)

    selected_df.to_csv(outdir / "selected_pairs.csv", index=False)
    pairs_df.to_csv(outdir / "all_candidate_pairs.csv", index=False)

    metric_rows = []
    mass_rows = []

    for selector in selectors:
        part = selected_df[selected_df["selector"] == selector]
        if len(part) == 0:
            continue

        acc = float(np.mean(part["correct"]))
        s = summarize_mass(part["selected_mbb"].to_numpy())

        metric_rows.append({
            "selector": selector,
            "n_events": int(len(part)),
            "pair_accuracy": acc,
            **s,
        })

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(outdir / "pairing_metrics.csv", index=False)

    with open(outdir / "counters.json", "w") as f:
        json.dump(counters, f, indent=2)

    # Make a simple summary markdown
    lines = []
    lines.append("# Pairing with DNN b-jet correction")
    lines.append("")
    lines.append("This evaluates simple H->bb candidate-pair selectors before/after applying the best b-jet response DNN to all selected reco AK4 candidate jets.")
    lines.append("")
    lines.append("## Counters")
    lines.append("")
    for k, v in counters.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Pair-selection metrics")
    lines.append("")
    lines.append("| Selector | Events | Pair accuracy | Median selected m_bb [GeV] | Width68 [GeV] | Frac 90-140 | Frac 100-150 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in metrics_df.iterrows():
        lines.append(
            f"| {r['selector']} | {int(r['n_events'])} | {r['pair_accuracy']:.4f} | "
            f"{r['median']:.2f} | {r['width68']:.2f} | {r['frac_90_140']:.4f} | {r['frac_100_150']:.4f} |"
        )

    (outdir / "RESULTS_PAIRING_WITH_BJET_CORRECTION.md").write_text("\n".join(lines) + "\n")

    print("\n=== Counters ===")
    for k, v in counters.items():
        print(f"{k}: {v}")

    print("\n=== Pair-selection metrics ===")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved outputs in: {outdir}")

    # Plots
    if len(metrics_df):
        df = metrics_df.sort_values("pair_accuracy")
        plt.figure(figsize=(9, 4.8))
        plt.barh(df["selector"], df["pair_accuracy"])
        plt.xlabel("Truth-pair selection accuracy")
        plt.title("H->bb pair-selection accuracy")
        plt.tight_layout()
        plt.savefig(plot_dir / "pair_accuracy_by_selector.png", dpi=180)
        plt.close()

        df = metrics_df.sort_values("width68")
        plt.figure(figsize=(9, 4.8))
        plt.barh(df["selector"], df["width68"])
        plt.xlabel("Selected-pair m_bb width68 [GeV]")
        plt.title("Selected-pair m_bb resolution")
        plt.tight_layout()
        plt.savefig(plot_dir / "selected_mbb_width68_by_selector.png", dpi=180)
        plt.close()

        plt.figure(figsize=(8, 5))
        bins = np.linspace(0, 250, 70)
        for selector in selectors:
            part = selected_df[selected_df["selector"] == selector]
            if len(part) == 0:
                continue
            plt.hist(part["selected_mbb"], bins=bins, histtype="step", density=True, label=selector)
        plt.axvline(M_H, linestyle="--", linewidth=1.0)
        plt.xlabel("Selected pair m_bb [GeV]")
        plt.ylabel("Normalized events")
        plt.title("Selected H->bb candidate mass")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(plot_dir / "selected_pair_mbb_distributions.png", dpi=180)
        plt.close()

        print(f"Saved plots in: {plot_dir}")


if __name__ == "__main__":
    main()
