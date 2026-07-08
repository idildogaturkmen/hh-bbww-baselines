# HH4b Delphes baseline update, 2026-07-08

## Current status

I built a first Delphes-based HH→4b baseline on LPC using compact parquet outputs.

Completed samples:
- ggF HH4b 10k
- VBF HH4b 10k
- QCD bbbb inclusive 100k
- QCD bbbb HT-sliced 10k/slice
- Zbbbb 100k
- inclusive ttbar 100k
- ttbb 50k diagnostic sample

## Background-model decision

The weighted region-yield table shows that QCD bbbb remains the dominant background in all tested HH-like regions. Inclusive ttbar is consistently the second-largest background. Zbbbb and ttbb are smaller but useful for resonant/background-shape checks.

Current nominal background model:
- QCD bbbb HT-sliced 10k/slice
- inclusive ttbar 100k
- Zbbbb 100k

Diagnostic/cross-check samples:
- inclusive QCD bbbb 100k
- ttbb 50k diagnostic

The ttbb sample has much higher HH4b-candidate efficiency than inclusive ttbar, but I am not adding it directly to inclusive ttbar because this could double-count shower heavy flavor without a truth-level split/veto.

## MC statistics

The current background samples are sufficient for steering decisions. The tightest regions are still MC-limited:
- QCD HT sliced, R_HH < 30 and mHH > 400: about 15% MC stat uncertainty
- inclusive ttbar, R_HH < 30 and mHH > 400: about 19% MC stat uncertainty

## BDT/cut baseline

A first weighted candidate-level BDT gives:
- all-feature BDT weighted AUC: about 0.785
- topology-only BDT weighted AUC: about 0.729

The all-feature BDT is mostly driven by reconstructed mass-plane variables, especially R_HH, mHH, mbb1/mbb2. The topology-only BDT still shows nontrivial separation, but the current BDT results should be treated as a diagnostic baseline rather than a final sensitivity result.

## Signal-normalization issue

The current generated signal cross sections are:
- ggF HH4b MG5 median xsec: about 0.994 fb
- VBF HH4b MG5 median xsec: about 0.948 fb

For comparison, approximate SM 13 TeV inclusive HH cross sections quoted in the recent HH→4b jet-free paper are:
- ggF HH: about 31.1 fb
- VBF HH: about 1.73 fb

Multiplying by BR(H→bb)^2 ≈ 0.5824^2 ≈ 0.339 gives approximate SM HH→4b cross sections:
- ggF HH→4b: about 10.55 fb
- VBF HH→4b: about 0.587 fb

Thus the naive scale factors relative to the generated samples are asymmetric:
- ggF: about 10.6
- VBF: about 0.62

I do not want to interpret these as simple K-factors yet. The mismatch may reflect process definitions, HEFT vs loop treatment, decay/cut conventions, or normalization choices. Before scaling signal production or presenting significance estimates, I want to confirm the correct signal normalization convention.
