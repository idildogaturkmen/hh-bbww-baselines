"""Paper-ready tables (Markdown, CSV, LaTeX) built ONLY from
MASTER_DEVELOPMENT_RESULTS.json (itself built read-only from frozen
sources by build_master_results.py). No new numbers computed here.

Produces:
  TABLE_MODEL_AUC_COMPARISON.{md,csv,tex}
  TABLE_WORKING_POINTS.{md,csv,tex}
  TABLE_SPANET_2M_10M_SCALING.{md,csv,tex}

Every table is prominently labeled DEVELOPMENT ONLY.
"""
import csv
import json
import os

OUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER = os.path.join(OUT_DIR, "master_results", "MASTER_DEVELOPMENT_RESULTS.json")
TABLES_DIR = os.path.join(OUT_DIR, "tables")

MODEL_ORDER = ["BDT_K", "BDT_KF", "SPANET_2M", "SPANET_10M"]
DEV_BANNER = "**DEVELOPMENT ONLY -- not a final/governing result, not independent Stage-C inference, not physically normalized.**"


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


def fmt(x, nd=4):
    if x is None:
        return "--"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def fmt_r(x):
    if x is None:
        return "undefined"
    return f"{x:,.1f}"


def main():
    m = json.load(open(MASTER))
    models = m["models"]
    os.makedirs(TABLES_DIR, exist_ok=True)

    # ================= TABLE_MODEL_AUC_COMPARISON =================
    md = [f"# TABLE_MODEL_AUC_COMPARISON\n", DEV_BANNER, "",
          f"Cohort: {m['cohort']['n_events']:,} events (signal={m['cohort']['n_signal']:,}, "
          f"QCD={m['cohort']['n_qcd']:,}, ttbar={m['cohort']['n_ttbar']:,}), identical across all four models.\n",
          "| Model | Training population | Feature/input definition | Seed | Training budget | Checkpoint criterion | AUC all-bg | AUC QCD | AUC ttbar |",
          "|---|---|---|---|---|---|---|---|---|"]
    csv_rows = []
    for mid in MODEL_ORDER:
        mm = models[mid]
        md.append("| {} | {:,} train | {} | {} | {} | {} | {:.4f} | {:.4f} | {:.4f} |".format(
            mm["display_name"], mm["training_population"]["n_train"], mm["feature_definition"],
            mm["seed"], mm["training_budget"], mm["checkpoint_criterion"],
            mm["auc"]["all_background"], mm["auc"]["qcd"], mm["auc"]["ttbar"]))
        csv_rows.append(dict(
            model=mm["display_name"], training_population=mm["training_population"]["n_train"],
            feature_definition=mm["feature_definition"], seed=mm["seed"], training_budget=mm["training_budget"],
            checkpoint_criterion=mm["checkpoint_criterion"],
            auc_all_background=mm["auc"]["all_background"], auc_qcd=mm["auc"]["qcd"], auc_ttbar=mm["auc"]["ttbar"],
        ))
    with open(os.path.join(TABLES_DIR, "TABLE_MODEL_AUC_COMPARISON.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(TABLES_DIR, "TABLE_MODEL_AUC_COMPARISON.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys())); w.writeheader(); w.writerows(csv_rows)
    tex = [r"% DEVELOPMENT ONLY -- not a final/governing result, not independent Stage-C inference, not physically normalized.",
           r"\begin{table}[htbp]", r"\centering",
           r"\caption{DEVELOPMENT ONLY: AUC comparison on the matched 400{,}000-event development cohort. Not a final/governing result; not independent Stage-C inference; not physically normalized.}",
           r"\begin{tabular}{lrrr}", r"\hline",
           r"Model & AUC (all-bg) & AUC (QCD) & AUC (ttbar) \\", r"\hline"]
    for mid in MODEL_ORDER:
        mm = models[mid]
        tex.append(r"{} & {:.4f} & {:.4f} & {:.4f} \\".format(esc(mm["display_name"]), mm["auc"]["all_background"], mm["auc"]["qcd"], mm["auc"]["ttbar"]))
    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(TABLES_DIR, "TABLE_MODEL_AUC_COMPARISON.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")

    # ================= TABLE_WORKING_POINTS =================
    md = [f"# TABLE_WORKING_POINTS\n", DEV_BANNER, "",
          "R = 1/epsB (background rejection). Finite-support warnings (Poisson-limited or zero-survivor "
          "operating points) are listed per row where applicable -- treat those R values as unreliable "
          "or undefined, not precise estimates.\n",
          "| Model | target epsS | achieved epsS | R all-bg | R QCD | R ttbar | n_sig pass | n_bg pass | n_QCD pass | n_ttbar pass | finite-support warning |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    csv_rows = []
    for mid in MODEL_ORDER:
        mm = models[mid]
        for wp in mm["working_points"]:
            warn = "; ".join(wp["finite_support_warnings"]) if wp["finite_support_warnings"] else ""
            md.append("| {} | {:.2f} | {:.4f} | {} | {} | {} | {:,} | {:,} | {:,} | {:,} | {} |".format(
                mm["display_name"], wp["target_epsS"], wp["achieved_epsS"],
                fmt_r(wp["R_all_background"]), fmt_r(wp["R_qcd"]), fmt_r(wp["R_ttbar"]),
                wp["n_signal_pass"], wp["n_all_background_pass"], wp["n_qcd_pass"], wp["n_ttbar_pass"], warn))
            csv_rows.append(dict(
                model=mm["display_name"], target_epsS=wp["target_epsS"], achieved_epsS=wp["achieved_epsS"],
                threshold=wp["threshold"], R_all_background=wp["R_all_background"], R_qcd=wp["R_qcd"], R_ttbar=wp["R_ttbar"],
                n_signal_pass=wp["n_signal_pass"], n_signal_total=wp["n_signal_total"],
                n_all_background_pass=wp["n_all_background_pass"], n_all_background_total=wp["n_all_background_total"],
                n_qcd_pass=wp["n_qcd_pass"], n_qcd_total=wp["n_qcd_total"],
                n_ttbar_pass=wp["n_ttbar_pass"], n_ttbar_total=wp["n_ttbar_total"],
                finite_support_warnings=warn,
            ))
    with open(os.path.join(TABLES_DIR, "TABLE_WORKING_POINTS.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(TABLES_DIR, "TABLE_WORKING_POINTS.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys())); w.writeheader(); w.writerows(csv_rows)
    tex = [r"% DEVELOPMENT ONLY -- not a final/governing result, not independent Stage-C inference, not physically normalized.",
           r"\begin{table}[htbp]", r"\centering",
           r"\caption{DEVELOPMENT ONLY: working points on the matched 400{,}000-event development cohort. $R=1/\epsilon_B$. Rows with finite-support warnings (see text) are Poisson-limited or zero-survivor and should not be read as precise estimates.}",
           r"\begin{tabular}{lrrrrr}", r"\hline",
           r"Model & $\epsilon_S$ (target) & $R_{\rm all-bg}$ & $R_{\rm QCD}$ & $R_{\rm ttbar}$ & finite-support \\", r"\hline"]
    for mid in MODEL_ORDER:
        mm = models[mid]
        for wp in mm["working_points"]:
            flag = "yes" if wp["finite_support_warnings"] else ""
            r_ttbar = "undefined" if wp["R_ttbar"] is None else f"{wp['R_ttbar']:.1f}"
            tex.append(r"{} & {:.2f} & {:.1f} & {:.1f} & {} & {} \\".format(
                esc(mm["display_name"]), wp["target_epsS"], wp["R_all_background"], wp["R_qcd"], r_ttbar, flag))
    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(TABLES_DIR, "TABLE_WORKING_POINTS.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")

    # ================= TABLE_SPANET_2M_10M_SCALING =================
    sc = m["scaling_study_2M_vs_10M"]
    md = [f"# TABLE_SPANET_2M_10M_SCALING\n", DEV_BANNER, "",
          "Same architecture, same seed=0, same batch_size, same 50-epoch budget, same frozen 400k "
          "validation population, same checkpoint criterion (max validation_average_jet_accuracy). "
          "Only training-population size differs.\n",
          "| Metric | 2M seed-0 | 10M seed-0 |", "|---|---|---|",
          "| Primary best epoch | {} | {} |".format(sc["run_2M"]["primary_best_epoch"], sc["run_10M"]["primary_best_epoch"]),
          "| validation_average_jet_accuracy (primary) | {:.6f} | {:.6f} |".format(sc["run_2M"]["validation_average_jet_accuracy"], sc["run_10M"]["validation_average_jet_accuracy"]),
          "| Secondary best val/loss/total_loss | {:.6f} | {:.6f} |".format(sc["run_2M"]["secondary_best_val_loss_total_loss"], sc["run_10M"]["secondary_best_val_loss_total_loss"]),
          "| Total runtime (s) | {:.2f} | {:.2f} |".format(sc["run_2M"]["total_fit_wall_s"], sc["run_10M"]["total_fit_wall_s"]),
          "| Events/s (train-only) | {:.2f} | {:.2f} |".format(sc["run_2M"]["events_per_s_train_only"], sc["run_10M"]["events_per_s_train_only"]),
          "| Peak GPU reserved (bytes) | {:,} | {:,} |".format(sc["run_2M"]["gpu_peak_reserved_bytes"], sc["run_10M"]["gpu_peak_reserved_bytes"]),
          "| Training population | {:,} | {:,} |".format(sc["run_2M"]["n_train_events"], sc["run_10M"]["n_train_events"]),
          "| AUC all-background (classification, separate eval) | {:.4f} | {:.4f} |".format(sc["classification_auc_comparison"]["spanet_2M"]["all_background"], sc["classification_auc_comparison"]["spanet_10M"]["all_background"]),
          "| AUC QCD | {:.4f} | {:.4f} |".format(sc["classification_auc_comparison"]["spanet_2M"]["qcd"], sc["classification_auc_comparison"]["spanet_10M"]["qcd"]),
          "| AUC ttbar | {:.4f} | {:.4f} |".format(sc["classification_auc_comparison"]["spanet_2M"]["ttbar"], sc["classification_auc_comparison"]["spanet_10M"]["ttbar"]),
          "", sc["classification_auc_comparison"]["note"]]
    with open(os.path.join(TABLES_DIR, "TABLE_SPANET_2M_10M_SCALING.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    csv_rows = [
        dict(metric="primary_best_epoch", value_2M=sc["run_2M"]["primary_best_epoch"], value_10M=sc["run_10M"]["primary_best_epoch"]),
        dict(metric="validation_average_jet_accuracy", value_2M=sc["run_2M"]["validation_average_jet_accuracy"], value_10M=sc["run_10M"]["validation_average_jet_accuracy"]),
        dict(metric="secondary_best_val_loss_total_loss", value_2M=sc["run_2M"]["secondary_best_val_loss_total_loss"], value_10M=sc["run_10M"]["secondary_best_val_loss_total_loss"]),
        dict(metric="total_fit_wall_s", value_2M=sc["run_2M"]["total_fit_wall_s"], value_10M=sc["run_10M"]["total_fit_wall_s"]),
        dict(metric="events_per_s_train_only", value_2M=sc["run_2M"]["events_per_s_train_only"], value_10M=sc["run_10M"]["events_per_s_train_only"]),
        dict(metric="gpu_peak_reserved_bytes", value_2M=sc["run_2M"]["gpu_peak_reserved_bytes"], value_10M=sc["run_10M"]["gpu_peak_reserved_bytes"]),
        dict(metric="n_train_events", value_2M=sc["run_2M"]["n_train_events"], value_10M=sc["run_10M"]["n_train_events"]),
        dict(metric="auc_all_background", value_2M=sc["classification_auc_comparison"]["spanet_2M"]["all_background"], value_10M=sc["classification_auc_comparison"]["spanet_10M"]["all_background"]),
        dict(metric="auc_qcd", value_2M=sc["classification_auc_comparison"]["spanet_2M"]["qcd"], value_10M=sc["classification_auc_comparison"]["spanet_10M"]["qcd"]),
        dict(metric="auc_ttbar", value_2M=sc["classification_auc_comparison"]["spanet_2M"]["ttbar"], value_10M=sc["classification_auc_comparison"]["spanet_10M"]["ttbar"]),
    ]
    with open(os.path.join(TABLES_DIR, "TABLE_SPANET_2M_10M_SCALING.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value_2M", "value_10M"]); w.writeheader(); w.writerows(csv_rows)
    tex = [r"% DEVELOPMENT ONLY -- not a final/governing result, not independent Stage-C inference, not physically normalized.",
           r"\begin{table}[htbp]", r"\centering",
           r"\caption{DEVELOPMENT ONLY: SPA-Net seed-0 training-scaling comparison, 2M vs.\ 10M events. Same architecture/seed/batch/epoch-budget/validation population/checkpoint criterion.}",
           r"\begin{tabular}{lrr}", r"\hline", r"Metric & 2M seed-0 & 10M seed-0 \\", r"\hline",
           r"Primary best epoch & {} & {} \\".format(sc["run_2M"]["primary_best_epoch"], sc["run_10M"]["primary_best_epoch"]),
           r"validation\_average\_jet\_accuracy & {:.4f} & {:.4f} \\".format(sc["run_2M"]["validation_average_jet_accuracy"], sc["run_10M"]["validation_average_jet_accuracy"]),
           r"Total runtime (s) & {:.1f} & {:.1f} \\".format(sc["run_2M"]["total_fit_wall_s"], sc["run_10M"]["total_fit_wall_s"]),
           r"Events/s (train-only) & {:.1f} & {:.1f} \\".format(sc["run_2M"]["events_per_s_train_only"], sc["run_10M"]["events_per_s_train_only"]),
           r"Training population & {:,} & {:,} \\".format(sc["run_2M"]["n_train_events"], sc["run_10M"]["n_train_events"]),
           r"AUC all-background & {:.4f} & {:.4f} \\".format(sc["classification_auc_comparison"]["spanet_2M"]["all_background"], sc["classification_auc_comparison"]["spanet_10M"]["all_background"]),
           r"AUC QCD & {:.4f} & {:.4f} \\".format(sc["classification_auc_comparison"]["spanet_2M"]["qcd"], sc["classification_auc_comparison"]["spanet_10M"]["qcd"]),
           r"AUC ttbar & {:.4f} & {:.4f} \\".format(sc["classification_auc_comparison"]["spanet_2M"]["ttbar"], sc["classification_auc_comparison"]["spanet_10M"]["ttbar"]),
           r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(TABLES_DIR, "TABLE_SPANET_2M_10M_SCALING.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")

    print("Wrote 3 tables x 3 formats to", TABLES_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
