#!/usr/bin/env python3
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt

store = Path(os.environ["HH4B_STORE"])

inp = store / "metadata" / "qcd_bbbb_ptb_threshold_scan_2000.csv"
outdir = store / "plots" / "qcd_ptb_threshold_scan_2026_07_07"
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(inp).sort_values("ptb_min_GeV")

def save(name):
    out = outdir / name
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print("Wrote:", out)

plt.figure()
plt.plot(df["ptb_min_GeV"], df["xsec_pb"], marker="o")
plt.yscale("log")
plt.xlabel("Generator-level b-quark pT minimum [GeV]")
plt.ylabel("Cross section [pb]")
plt.title("QCD bbbb cross section vs generator-level b-quark pT cut")
save("qcd_bbbb_xsec_vs_ptb.png")

plt.figure()
plt.plot(df["ptb_min_GeV"], df["candidate_eff"], marker="o")
plt.xlabel("Generator-level b-quark pT minimum [GeV]")
plt.ylabel("4-b-tag candidate efficiency")
plt.title("HH4b candidate efficiency vs generator-level b-quark pT cut")
save("qcd_bbbb_candidate_eff_vs_ptb.png")

plt.figure()
plt.plot(df["ptb_min_GeV"], df["effective_candidate_xsec_pb"], marker="o")
plt.yscale("log")
plt.xlabel("Generator-level b-quark pT minimum [GeV]")
plt.ylabel("Effective candidate cross section [pb]")
plt.title("QCD bbbb candidate-weighted cross section vs pT cut")
save("qcd_bbbb_effective_candidate_xsec_vs_ptb.png")

plt.figure()
plt.plot(df["ptb_min_GeV"], df["median_mhh"], marker="o")
plt.xlabel("Generator-level b-quark pT minimum [GeV]")
plt.ylabel("Median reconstructed mHH [GeV]")
plt.title("Median reconstructed mHH vs generator-level b-quark pT cut")
save("qcd_bbbb_median_mhh_vs_ptb.png")

# Markdown summary table
md = outdir / "qcd_bbbb_ptb_threshold_scan_2000.md"
md.write_text(df.to_markdown(index=False) + "\n")
print("Wrote:", md)
