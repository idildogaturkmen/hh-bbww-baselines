'''
Look at the composition of processes in the multiclass DNN predictions, in a CMS-style table.
diagnose which process groups dominate the HH node
'''

from pathlib import Path
import numpy as np
import pandas as pd


def neff(w):
    w = np.asarray(w, dtype=float)
    den = np.sum(w * w)
    return float(np.sum(w) ** 2 / den) if den > 0 else 0.0


def summarize_region(df, mask, label, top_n=20):
    sub = df[mask].copy()

    if len(sub) == 0:
        return pd.DataFrame([{"region": label, "status": "empty"}])

    group_cols = []
    for c in ["process_group", "sample", "process", "file"]:
        if c in sub.columns:
            group_cols.append(c)

    if not group_cols:
        raise RuntimeError("No process/sample/file columns found in prediction table.")

    weight_col = "physics_weight_nominal" if "physics_weight_nominal" in sub.columns else "physics_weight"

    rows = []
    for keys, g in sub.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        d = {"region": label}
        for c, v in zip(group_cols, keys):
            d[c] = v

        y = g["y_binary"] if "y_binary" in g.columns else g["target"]
        w = g[weight_col].astype(float)

        d.update({
            "raw_events": len(g),
            "signal_raw": int((y == 1).sum()),
            "background_raw": int((y == 0).sum()),
            "weighted_events": float(w.sum()),
            "neff": neff(w),
            "mean_score_HH": float(g["score_HH"].mean()) if "score_HH" in g.columns else np.nan,
        })
        rows.append(d)

    out = pd.DataFrame(rows)
    out = out.sort_values("weighted_events", ascending=False).head(top_n)
    return out


def main():
    pred_path = Path("outputs/tabular_dnn_multiclass_recoMET_topology_cmsstyle/multiclass_dnn_predictions.parquet")
    outdir = Path("outputs/tabular_dnn_multiclass_recoMET_topology_cmsstyle")
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(pred_path)

    weight_col = "physics_weight_nominal" if "physics_weight_nominal" in df.columns else "physics_weight"

    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("Weight column:", weight_col)

    regions = {
        "val_argmax_HH": (df["split"] == "val") & (df["predicted_class_name"] == "HH_ggF_like"),
        "test_argmax_HH": (df["split"] == "test") & (df["predicted_class_name"] == "HH_ggF_like"),

        "val_argmax_HH_score_ge_0p35": (df["split"] == "val") & (df["predicted_class_name"] == "HH_ggF_like") & (df["score_HH"] >= 0.35),
        "test_argmax_HH_score_ge_0p35": (df["split"] == "test") & (df["predicted_class_name"] == "HH_ggF_like") & (df["score_HH"] >= 0.35),

        "val_global_HH_score_ge_0p125": (df["split"] == "val") & (df["score_HH"] >= 0.125),
        "test_global_HH_score_ge_0p125": (df["split"] == "test") & (df["score_HH"] >= 0.125),
    }

    all_rows = []
    for name, mask in regions.items():
        summary = summarize_region(df, mask, name)
        all_rows.append(summary)

        print(f"\n=== {name} ===")
        cols_to_show = [c for c in [
            "region", "process_group", "sample", "process", "raw_events",
            "signal_raw", "background_raw", "weighted_events", "neff", "mean_score_HH"
        ] if c in summary.columns]
        print(summary[cols_to_show].to_string(index=False))

    out = pd.concat(all_rows, ignore_index=True)
    out_path = outdir / "cmsstyle_multiclass_process_composition.csv"
    out.to_csv(out_path, index=False)
    print("\nWrote:", out_path)


if __name__ == "__main__":
    main()
