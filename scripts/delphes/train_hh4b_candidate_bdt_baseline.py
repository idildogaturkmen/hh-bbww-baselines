#!/usr/bin/env python3
from pathlib import Path
import os
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier

store = Path(os.environ["HH4B_STORE"])
inp = store / "ml" / "hh4b_candidate_training_10k_v0.parquet"
outdir = store / "ml"
outdir.mkdir(exist_ok=True)

df = pd.read_parquet(inp)

features = [
    "n_selected_bjets",
    "mbb1",
    "mbb2",
    "avg_mbb",
    "delta_mbb",
    "mhh",
    "drbb1",
    "drbb2",
    "j1_pt",
    "j2_pt",
    "j3_pt",
    "j4_pt",
]

df = df.dropna(subset=features + ["label"]).copy()

X = df[features]
y = df["label"].astype(int)

X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X, y, df, test_size=0.30, random_state=12345, stratify=y
)

clf = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    random_state=12345,
)
clf.fit(X_train, y_train)

score = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, score)

df_out = df_test[["sample", "event", "process", "label", "process_group"] + features].copy()
df_out["bdt_score"] = score
score_out = outdir / "hh4b_candidate_bdt_10k_v0_scores.parquet"
df_out.to_parquet(score_out, index=False)

report = {
    "input": str(inp),
    "n_total": int(len(df)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "features": features,
    "auc_unweighted": float(auc),
    "test_signal_rows": int((y_test == 1).sum()),
    "test_background_rows": int((y_test == 0).sum()),
    "score_output": str(score_out),
}

report_out = outdir / "hh4b_candidate_bdt_10k_v0_report.json"
report_out.write_text(json.dumps(report, indent=2) + "\n")

print(json.dumps(report, indent=2))
print()
print(classification_report(y_test, score > 0.5))
