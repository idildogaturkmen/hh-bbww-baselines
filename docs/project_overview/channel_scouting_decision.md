# Channel scouting decision after bbWW baseline

Date: 2026-07-01  
Branch: `pivot-channel-scouting`  
Repository: `hh-bbww-baselines`  
Input: `outputs/collide_selected_backgrounds/**/*.parquet`  
Metadata: `config/collide_sample_metadata_rough.csv`  
Output directory: `outputs/channel_scouting/full_run_fixed2`

---

## 1. Purpose

This note summarizes the first full channel-scouting pass after completing the corrected `HH→bbWW*` one-lepton baseline study.

The goal of the scout was to decide which HH channel or kinematic region should become the next main benchmark before training new models.

The candidate regions were:

- `HH→bbbb` resolved 4b,
- `HH→bbbb` resolved 4b with Higgs-mass pairing,
- `HH→bbbb` boosted,
- `HH→bbbb` VBF-like / ggF-bin-style region,
- `HH→bbγγ` with and without a diphoton mass window,
- the previous `HH→bbWW*` one-lepton region as a reference.

The full scouting output is stored in:

```text
outputs/channel_scouting/full_run_fixed2/channel_scouting_summary.csv
outputs/channel_scouting/full_run_fixed2/channel_scouting_dominant_backgrounds.csv
outputs/channel_scouting/full_run_fixed2/channel_choice_draft.md
```

---

## 2. Important caveats

The scouting uses rough proxy MC weights from:

```text
config/collide_sample_metadata_rough.csv
```

These weights are diagnostic only. They are useful for comparing candidate regions, but they are not final CMS-quality normalizations.

Known caveats:

1. QCD samples currently have missing proxy cross sections:
   - `QCD_HT50toInf`
   - `QCD_HT50tobb`

2. Therefore, selected QCD events are included in raw counts but excluded from weighted yields.

3. This affects HH4b most directly because QCD is expected to be a major real background.

4. The current weighted \(B_w\), S/B, and \(Z_A\) for HH4b should therefore be treated as provisional.

5. Generic inclusive CMS QCD cross sections should not be inserted blindly, because the available samples may have generator-level HT cuts, flavor filters, or phase-space restrictions.

The safer strategy is:

- keep QCD in raw counts,
- mark weighted yields as incomplete when QCD has missing metadata,
- ask for or derive sample-specific QCD metadata later,
- and only add manual proxy QCD weights if clearly labeled as rough/manual.

---

## 3. Full scouting summary

The table below summarizes the full channel scout.

| Region | Raw signal | Raw background | Missing-weight bkg raw | \(S_w\) | \(B_w\) | S/B | \(S/\sqrt{S+B}\) | \(Z_A\) | \(N_\mathrm{eff}^{bkg}\) | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `HH4b_resolved_4b_basic` | 1313 | 15128 | 6 | 95.49 | \(1.45\times10^6\) | \(6.58\times10^{-5}\) | 0.0793 | 0.0793 | 86.13 | Best rough sensitivity among tested regions |
| `HH4b_resolved_4b_Hmass` | 1035 | 8355 | 2 | 75.27 | \(9.57\times10^5\) | \(7.86\times10^{-5}\) | 0.0769 | 0.0769 | 43.19 | Slightly better purity; physically meaningful Higgs-pairing region |
| `HH4b_boosted_1AK8_2b` | 1873 | 54944 | 153 | 136.22 | \(1.84\times10^7\) | \(7.41\times10^{-6}\) | 0.0318 | 0.0318 | 212.51 | Higher signal yield but too much background |
| `HH4b_resolved_VBF_like` | 254 | 4397 | 1 | 18.47 | \(4.45\times10^5\) | \(4.15\times10^{-5}\) | 0.0277 | 0.0277 | 27.06 | Interesting category, but lower yield and lower rough sensitivity |
| `bbWW_onelep_reference_like` | 2587 | 127032 | 308 | 154.65 | \(4.78\times10^7\) | \(3.23\times10^{-6}\) | 0.0224 | 0.0224 | 606.38 | Previous channel; worse than resolved HH4b in this scout |
| `bbgamma_gamma_mgg_window` | 1698 | 163 | 0 | 0.962 | \(2.46\times10^3\) | \(3.92\times10^{-4}\) | 0.0194 | 0.0194 | 1.85 | Cleanest S/B, but very small signal yield and unstable background |
| `HH4b_boosted_2AK8` | 2915 | 150434 | 2309 | 212.00 | \(4.49\times10^8\) | \(4.72\times10^{-7}\) | 0.0100 | 0.0100 | 1265.14 | Too broad; background-dominated |
| `bbgamma_gamma_basic` | 1786 | 900 | 0 | 1.012 | \(1.48\times10^5\) | \(6.85\times10^{-6}\) | 0.00263 | 0.00263 | 12.33 | Not promising without the photon mass window |

---

## 4. Interpretation of HH4b vs bbγγ

The `bbγγ` mass window has the best S/B:

\[
S/B \approx 3.9\times10^{-4}
\]

However, its weighted signal yield is tiny:

\[
S_w \approx 0.96
\]

and its background estimate is unstable:

\[
N_\mathrm{eff}^{bkg} \approx 1.85
\]

This makes `bbγγ` clean but fragile for a first ML benchmark.

By contrast, resolved HH4b has much larger weighted signal yield:

\[
S_w \approx 75\text{--}95
\]

and more usable background effective statistics:

\[
N_\mathrm{eff}^{bkg} \approx 43\text{--}86
\]

The resolved HH4b regions also outperform the previous `bbWW*` one-lepton reference region in rough expected sensitivity:

\[
Z_\mathrm{HH4b,resolved} \approx 0.077\text{--}0.079
\]

compared with:

\[
Z_\mathrm{bbWW,reference} \approx 0.022
\]

So the resolved HH4b benchmark is approximately:

\[
0.079 / 0.022 \approx 3.6
\]

times better than the current `bbWW*` one-lepton reference in this rough scouting metric.

---

## 5. Dominant backgrounds

### 5.1 `HH4b_resolved_4b_basic`

| Background group | Raw background | Weighted yield | Fraction of \(B_w\) | \(N_\mathrm{eff}\) |
|---|---:|---:|---:|---:|
| ttbar | 650 | \(7.88\times10^5\) | 54.3% | 559.5 |
| DY/Z+jets | 15 | \(5.90\times10^5\) | 40.7% | 14.9 |
| diboson | 303 | \(5.65\times10^4\) | 3.90% | 242.5 |
| ttV/ttH/tttt | 13467 | \(1.11\times10^4\) | 0.77% | 7006.8 |
| single Higgs | 165 | \(5.15\times10^3\) | 0.35% | 49.3 |

### 5.2 `HH4b_resolved_4b_Hmass`

| Background group | Raw background | Weighted yield | Fraction of \(B_w\) | \(N_\mathrm{eff}\) |
|---|---:|---:|---:|---:|
| DY/Z+jets | 13 | \(5.17\times10^5\) | 54.0% | 13.0 |
| ttbar | 318 | \(4.04\times10^5\) | 42.1% | 280.5 |
| diboson | 150 | \(2.77\times10^4\) | 2.89% | 118.2 |
| ttV/ttH/tttt | 7490 | \(6.28\times10^3\) | 0.66% | 4102.4 |
| single Higgs | 92 | \(2.76\times10^3\) | 0.29% | 27.4 |

### 5.3 `HH4b_resolved_VBF_like`

| Background group | Raw background | Weighted yield | Fraction of \(B_w\) | \(N_\mathrm{eff}\) |
|---|---:|---:|---:|---:|
| ttbar | 192 | \(2.43\times10^5\) | 54.5% | 168.4 |
| DY/Z+jets | 5 | \(1.86\times10^5\) | 41.7% | 4.94 |
| diboson | 60 | \(1.26\times10^4\) | 2.84% | 52.7 |
| ttV/ttH/tttt | 3960 | \(3.30\times10^3\) | 0.74% | 2015.6 |
| single Higgs | 46 | \(8.40\times10^2\) | 0.19% | 9.91 |

### 5.4 `bbgamma_gamma_mgg_window`

| Background group | Raw background | Weighted yield | Fraction of \(B_w\) | \(N_\mathrm{eff}\) |
|---|---:|---:|---:|---:|
| ttbar | 1 | \(1.77\times10^3\) | 72.1% | 1.0 |
| diboson | 4 | \(6.15\times10^2\) | 25.0% | 2.90 |
| ttV/ttH/tttt | 45 | \(4.47\times10^1\) | 1.82% | 26.6 |
| single Higgs | 112 | \(2.58\times10^1\) | 1.05% | 57.6 |

The `bbγγ` mass-window background is dominated by very few effective background events, especially one ttbar event with a large weight. This makes the estimate unstable.

---

## 6. Decision

The next main benchmark should be:

\[
\boxed{HH\to b\bar b b\bar b \text{ resolved reconstruction}}
\]

Specifically:

- Use `HH4b_resolved_4b_basic` as the broad training preselection.
- Include Higgs-candidate mass-pairing variables as BDT inputs.
- Evaluate the trained BDT in both:
  - the broad resolved 4b region,
  - and the H-mass-paired region.

The reason for using the broader 4b preselection for training is that it preserves more signal and gives the BDT room to learn the value of Higgs-pairing information instead of hard-cutting too early.

The `HH4b_resolved_4b_Hmass` region is physically meaningful and has slightly better purity, so it should be used as a validation/evaluation category.

The `HH4b_resolved_VBF_like` region should be kept as a secondary category because it is connected to the idea of checking whether special topologies may be more favorable to advanced methods.

The `bbγγ` mass-window region should remain a backup channel because it is clean but has very small weighted signal yield and unstable effective background statistics.

---

## 7. Recommended next workflow

The next workflow should be:

1. Build a resolved HH4b feature table.
2. Define a broad resolved 4b preselection.
3. Reconstruct Higgs candidates from b-jet pairings.
4. Train a BDT baseline.
5. Evaluate BDT score categories.
6. Compare the BDT to the cut-only H-mass-pairing region.
7. Only after this, decide whether SPA-Net is justified.
8. Consider ParT/JP-JEPA embeddings only after the BDT baseline is stable.
9. Treat RINO as optional and later.

The immediate modeling sequence is:

```text
cut-based HH4b resolved scout
→ resolved HH4b BDT baseline
→ assignment-aware SPA-Net
→ SPA-Net or BDT with ParT/JP-JEPA embeddings, if justified
→ optional RINO-style comparison if time allows
```

---

## 8. First BDT target

The first BDT should be:

```text
Signal: HH_4b
Background: all non-HH4b selected backgrounds with usable weights
Preselection: >=4 b-tag AK4 jets
Primary evaluation categories:
  - broad resolved 4b
  - resolved 4b + Higgs mass window
  - VBF-like resolved 4b
```

Important input variables should include:

- four leading b-tagged jet \(p_T\),
- four leading b-tagged jet \(\eta\),
- four leading b-tag scores,
- number of AK4 jets,
- number of b-tagged AK4 jets,
- scalar \(H_T\),
- best Higgs candidate masses \(m_{H1}\), \(m_{H2}\),
- \(|m_{H1}-125|\),
- \(|m_{H2}-125|\),
- pairing score \(|m_{H1}-125| + |m_{H2}-125|\),
- approximate \(m_{HH}\),
- \(\Delta R_{bb}\) for each Higgs candidate,
- VBF candidate \(m_{jj}\),
- VBF candidate \(|\Delta\eta_{jj}|\),
- AK8/fat-jet counts as optional features.

The BDT should report:

- raw AUC,
- weighted AUC,
- average precision,
- score thresholds,
- \(S_w\),
- \(B_w\),
- S/B,
- \(S/\sqrt{S+B}\),
- Asimov \(Z_A\),
- \(N_\mathrm{eff}^{bkg}\),
- dominant background composition in top-score regions.

---

## 9. Current conclusion

The full channel scout indicates that resolved HH4b is the most promising next benchmark among the tested regions.

The basic resolved 4b selection gives the highest rough significance, while the Higgs-mass-paired resolved selection gives slightly better purity with similar sensitivity. Both outperform the previous `bbWW*` one-lepton reference region in rough expected sensitivity.

The `bbγγ` mass window is cleaner, but it has very small expected signal yield and unstable background effective statistics, so it is better kept as a backup channel.

The next step is to build a resolved HH4b BDT baseline using the broad 4b selection and Higgs-candidate reconstruction variables, while keeping QCD normalization marked as incomplete until sample-specific metadata is available.
"""

Path("notes/channel_scouting_decision.md").write_text(content, encoding="utf-8")
print("Wrote notes/channel_scouting_decision.md")
