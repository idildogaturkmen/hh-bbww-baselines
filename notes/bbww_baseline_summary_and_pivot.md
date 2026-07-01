# HH→bbWW* semileptonic baseline study and pivot summary

Date: 2026-07-01  
Author: Idil Doga Turkmen  
Repository: `hh-bbww-baselines`  
Main dataset context: COLLIDE-1M / CMS-like simulated selected samples  
Current analysis stage: simulation-level expected-yield and model-comparison study

---

## 1. Purpose of this note

This note documents the current status of the `HH→bbWW*` semileptonic baseline study, including:

1. the original motivation for studying the one-lepton `HH→bbWW*` topology,
2. the construction of cut, BDT, and DNN baselines,
3. the corrected reconstructed-MET update,
4. the interpretation of BDT and DNN performance after the MET correction,
5. the limitations found in the current `bbWW*` region,
6. and the motivation for pivoting toward a more promising HH channel or kinematic region before investing further in advanced architectures.

The goal is to preserve the work completed so far and make the reason for the pivot explicit. The `HH→bbWW*` work remains useful as a baseline and diagnostic study, but the current one-lepton region does not yet provide a strong enough signal region to justify continued architecture tuning as the main direction.

The next direction should not be an abstract comparison of machine-learning architectures. The analysis should remain tied to a concrete physics objective: finding a more promising HH channel, defining a stable signal region, and then testing whether advanced reconstruction or representation-learning methods improve the expected sensitivity over strong classical baselines.

---

## 2. Original analysis goal

The initial goal was to build a CMS-style exploratory analysis for semileptonic `HH→bbWW*`, focusing on the one-lepton final state.

The physics target was:

\[
HH \to b\bar{b}WW^*
\]

with one Higgs boson decaying to `bb` and the other to `WW*`.

The analysis focused on one-electron or one-muon events because the semileptonic topology gives:

- one charged lepton,
- missing transverse momentum from a neutrino,
- two b jets from `H→bb`,
- additional hadronic activity from the other W decay,
- and a final state that is more common than fully leptonic `bbWW*`.

The intended model comparison was:

1. simple cut baseline,
2. BDT baseline,
3. tabular DNNs,
4. multiclass DNNs,
5. reduced object-level DNNs,
6. full object-level DNNs,
7. later possible SPA-Net / assignment-aware models.

---

## 3. Current normalization and yield calculation

The current study uses rough MC-based event weights, not a full likelihood fit.

For each sample, the approximate event weight is:

\[
w_i \approx \frac{\mathcal{L}\sigma_{\text{proxy}}}{N_{\text{local sample}}}
\]

where:

- \(\mathcal{L}=138~\mathrm{fb}^{-1}\),
- \(\sigma_{\text{proxy}}\) comes from the rough sample metadata file,
- \(N_{\text{local sample}}\) is the number of locally available events in the processed sample or skim.

The selected weighted signal and background yields are computed as:

\[
S_w = \sum_{\text{selected signal}} w_i
\]

\[
B_w = \sum_{\text{selected background}} w_i
\]

This is conceptually similar to:

\[
S = \mathcal{L}\sigma\epsilon
\]

and

\[
B = \mathcal{L}\sigma_{\text{background}}\epsilon_{\text{background}}
\]

but implemented by summing event weights.

Important caveat:

- These weights are rough diagnostic weights.
- They are not final CMS-quality normalization.
- They do not yet include a full treatment of generator sum weights, filter efficiencies, k-factors, systematic uncertainties, control regions, or a likelihood fit.
- Therefore, the quoted \(S_w\), \(B_w\), S/B, and \(Z_A\) values should be interpreted as diagnostics for model comparison and region scouting, not final physics results.

A later CMS-style statistical treatment would require:

- official sample metadata,
- cross sections matching the exact generated samples,
- generated event counts or sum of generator weights,
- filter efficiencies and branching fractions where relevant,
- control-region validation,
- systematic uncertainties,
- and possibly binned likelihood or CLs-style limit setting.

---

## 4. Metrics used

The main metrics used in this study are:

| Metric | Meaning | Role |
|---|---|---|
| raw AUC | signal/background ranking without event weights | checks generic separation |
| weighted AUC | signal/background ranking with event weights | checks separation under rough expected yields |
| AP / average precision | precision-recall performance, useful for rare signal | checks whether top-score region is signal-enriched |
| \(S_w\) | weighted signal yield | expected selected signal |
| \(B_w\) | weighted background yield | expected selected background |
| S/B | purity | selected signal-to-background ratio |
| \(S/(S+B)\) | signal fraction | selected signal fraction |
| \(Z_A\) | approximate Asimov significance | rough expected sensitivity |
| \(N_\mathrm{eff}^{bkg}\) | effective weighted background sample size | stability diagnostic for weighted background |

The relationship between \(Z_A\) and \(N_\mathrm{eff}^{bkg}\) is important:

- \(Z_A\) is computed from the selected weighted signal and background yields.
- \(N_\mathrm{eff}^{bkg}\) is not itself the significance.
- \(N_\mathrm{eff}^{bkg}\) indicates whether the weighted background estimate is statistically stable.
- A region with high apparent \(Z_A\) but very low \(N_\mathrm{eff}^{bkg}\) may be unreliable because the background estimate could be dominated by only a few high-weight MC events.

In this study, \(N_\mathrm{eff}^{bkg}\) is used as a warning flag, not as a replacement for \(Z_A\).

For quick scouting, the following rough significance proxies are useful:

\[
Z \approx \frac{S}{\sqrt{B}}
\]

when \(S \ll B\), and

\[
Z \approx \frac{S}{\sqrt{S+B}}
\]

when \(S\) and \(B\) are of similar size.

For the new channel scouting step, both should be reported when useful, especially for regions where \(S\) is not negligible compared with \(B\).

---

## 5. Initial cut baseline

The one-lepton preselection gave:

| Region | Signal raw | Signal weighted | Main issue |
|---|---:|---:|---|
| one-lepton preselection | 1225 | 73.23 | very large background |
| \(H_T>100\), corrected \(80<m_{bb}<150\) | — | 50.51 | still very low purity |

The rough weighted cut baseline after the mass window was approximately:

\[
S_w = 50.51
\]

\[
B_w = 9.88\times 10^6
\]

\[
S/B \approx 5.1\times 10^{-6}
\]

\[
Z_A \approx 0.016
\]

This confirmed that simple cuts alone do not produce a sufficiently clean region. A multivariate classifier was therefore needed.

---

## 6. MET correction / recoMET update

A major issue was found in the MET handling.

The analysis had used inconsistent MET-derived quantities in different places. The correction was to use reconstructed `FullReco_MET` consistently, especially:

- `FullReco_MET_MET`
- `FullReco_MET_Phi`

The key finding was:

- MET magnitude was essentially unchanged.
- MET direction changed substantially.
- MET-angular variables changed substantially.

Summary:

| Quantity | Old vs corrected behavior | Interpretation |
|---|---|---|
| MET source | corrected table uses `FullReco_MET` for all events | detector-level reconstructed MET source is now consistent |
| `met` magnitude | mean difference ≈ 0, correlation ≈ 1.00 | MET size was effectively unchanged |
| `met_phi` | std. difference ≈ 2.05, correlation ≈ 0.36 | MET direction changed substantially |
| `mT_lep_met` | mean shift ≈ 18.1, std. ≈ 35.5, correlation ≈ 0.74 | lepton-MET transverse mass changed |
| `dphi_lep_met` | mean shift ≈ 0.29, std. ≈ 0.99, correlation ≈ 0.48 | lepton-MET angular relation changed |
| affected topology variables | `dphi_bb_met`, `min_dphi_met_b`, `mt_bb_met` | MET-direction-based topology features changed |

Important interpretation:

The correction did not mean that the previous MET magnitude was wrong. Rather, the issue was that the MET angular information was inconsistent. After the correction, MET-angular variables became physically consistent, but the model performance decreased compared with the earlier inconsistent-MET version.

---

## 7. Corrected recoMET BDTs

After the recoMET correction, several BDT feature sets were tested.

### 7.1 BDT feature sets

| BDT model | Input content | Purpose |
|---|---|---|
| Lepton/MET | lepton kinematics, MET, \(m_T(\ell,\mathrm{MET})\), \(\Delta\phi(\ell,\mathrm{MET})\) | tests leptonic W-side separation |
| Simple topology | lepton/MET + b-jet system + \(m_{bb}\), \(H_T\), b-jet kinematics, event topology | nominal event-level topology baseline |
| Hadronic-W/top | simple topology + non-b jet and W/top candidate variables | tests whether explicit hadronic-side reconstruction helps |
| Hadronic-W/top regularized | more conservative version of W/top BDT | tests whether high-purity tails are stable |

### 7.2 Corrected BDT performance

| BDT model | Test raw AUC | Test weighted AUC | Best-val threshold | Test \(S_w\) | Test \(B_w\) | Test S/B | Test \(Z_A\) | Test \(N_\mathrm{eff}^{bkg}\) | Comment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Lepton/MET only | 0.706 | 0.676 | 0.440 | 9.45 | \(1.98\times10^6\) | \(4.8\times10^{-6}\) | 0.00671 | 30.1 | broad, not high-purity |
| Simple topology | 0.700 | 0.672 | 0.655 | 1.79 | \(1.64\times10^5\) | \(1.1\times10^{-5}\) | 0.00443 | 6.18 | best-val threshold unstable on test |
| Hadronic-W/top | 0.711 | 0.691 | 0.740 | 0.060 | 962 | \(6.2\times10^{-5}\) | 0.00193 | 3.14 | high-purity tail too statistically unstable |

The naive best-validation thresholds were not sufficient by themselves because some thresholds selected very few effective background events. Therefore, stable-threshold scans were also considered.

### 7.3 Stable-threshold interpretation

The stable-threshold scans showed that the best-looking high-score tails must be interpreted together with \(N_\mathrm{eff}^{bkg}\).

A selection with slightly higher S/B but very low \(N_\mathrm{eff}^{bkg}\) is not necessarily better than a lower-purity selection with a more stable background estimate.

The practical interpretation was:

- the Lepton/MET BDT is useful but incomplete,
- the Hadronic-W/top BDT can produce high-purity tails but can become unstable,
- the Simple-topology BDT gives the best balance for a nominal corrected baseline.

### 7.4 BDT score categories

For the corrected BDT score categories:

| BDT category | \(S_w\) | \(B_w\) | S/B | \(Z_A\) | \(N_\mathrm{eff}^{bkg}\) |
|---|---:|---:|---:|---:|---:|
| < 0.625 | 21.64 | \(1.061\times10^7\) | \(2.04\times10^{-6}\) | 0.00664 | 104.81 |
| 0.625–0.70 | 2.39 | \(1.516\times10^5\) | \(1.58\times10^{-5}\) | 0.00614 | 20.23 |
| 0.70–0.75 | 2.63 | \(5.242\times10^4\) | \(5.02\times10^{-5}\) | 0.01149 | 65.68 |
| ≥ 0.75 | 3.89 | \(3.609\times10^4\) | \(1.08\times10^{-4}\) | 0.02045 | 47.77 |
| Inclusive > 0.625 | 8.91 | \(2.401\times10^5\) | \(3.70\times10^{-5}\) | ~0.01817 | ~47.69 |

Interpretation:

- The BDT score is physically meaningful.
- Higher-score bins have better S/B.
- The ≥0.75 bin has the best signal enrichment and highest approximate \(Z_A\).
- However, the absolute \(Z_A\) remains small.
- The BDT is useful as a baseline, but the one-lepton `bbWW*` region does not yet look strong enough for a high-impact sensitivity result.

---

## 8. Tabular DNNs

Several tabular DNN baselines were tested.

A tabular DNN takes one flat list of engineered variables per event, for example:

- lepton kinematics,
- MET,
- \(m_T(\ell,\mathrm{MET})\),
- \(m_{bb}\),
- \(H_T\),
- b-jet variables,
- topology variables,
- optional hadronic-W/top features.

### 8.1 Binary tabular DNN

The binary DNN was trained to classify:

\[
HH \quad \text{vs.} \quad \text{all backgrounds}
\]

It used a flat event-level feature vector and produced one HH-like score.

Result:

| Model | Test HH weighted AUC | Test HH raw AUC | Test \(Z_A\) | \(N_\mathrm{eff}^{bkg}\) | Comment |
|---|---:|---:|---:|---:|---|
| Binary tabular DNN | 0.609 | 0.655 | 0.00583 | 37.68 | weaker than BDT |

### 8.2 Small-regularized DNN

A smaller/more regularized DNN was tested to check whether the larger tabular DNN was overfitting.

| Model | Test HH weighted AUC | Test HH raw AUC | Test \(Z_A\) | \(N_\mathrm{eff}^{bkg}\) | Comment |
|---|---:|---:|---:|---:|---|
| Small-reg DNN | 0.583 | 0.658 | 0.00271 | 5.81 | unstable high-score tail |

### 8.3 Linear DNN

A very simple linear model was tested as a sanity check.

| Model | Test HH weighted AUC | Test HH raw AUC | Test \(Z_A\) | \(N_\mathrm{eff}^{bkg}\) | Comment |
|---|---:|---:|---:|---:|---|
| Linear DNN | 0.621 | 0.665 | 0.00665 | 73.68 | stable but weak separation |

### 8.4 Multiclass tabular DNN

The multiclass DNN predicted three categories:

- `HH_ggF_like`
- `Top_Higgs`
- `WJets_Other`

The goal was to make the DNN more CMS-like by separating different background categories rather than using only binary signal/background labels.

| Model | Test HH weighted AUC | Test HH raw AUC | Test \(Z_A\) | \(N_\mathrm{eff}^{bkg}\) | Comment |
|---|---:|---:|---:|---:|---|
| Multiclass tabular DNN | 0.611 | 0.647 | 0.00553 | 13.25 | no gain over BDT |

Interpretation:

The tabular DNNs learned some signal ranking, but none clearly beat the corrected BDT baseline.

---

## 9. Reduced object-level DNNs: v0/v1

The first object-level DNNs used a reduced object representation from the processed feature table.

### 9.1 v0/v1 object input

Objects:

- `b1`
- `b2`
- `lepton`
- `MET`

Object features:

- `log_pt`
- `eta`
- `sin_phi`
- `cos_phi`
- `log_mass`
- `present`

Object tensor:

\[
X_{\text{obj}} = (33679, 4, 6)
\]

Pair features:

- 4 objects gives \(\binom{4}{2}=6\) pairs.
- Each pair had 6 features.
- Pair tensor size:

\[
X_{\text{pair}} = (33679, 36)
\]

Auxiliary features:

\[
X_{\text{aux}} = (33679, 65)
\]

### 9.2 v1 model variants

| v1 model | Inputs | Purpose |
|---|---|---|
| object-only | `X_obj` | test object representation alone |
| object+pair | `X_obj + X_pair` | test explicit pair relationships |
| object+aux | `X_obj + X_aux` | test objects plus engineered scalar features |
| full | `X_obj + X_pair + X_aux` | complete reduced-object model |

Best v1 result:

| Model | Test HH weighted AUC | Test HH raw AUC | Test \(Z_A\) | \(N_\mathrm{eff}^{bkg}\) | Comment |
|---|---:|---:|---:|---:|---|
| LBN v1 full | 0.625 | 0.664 | 0.00571 | 28.37 | object input helps slightly |

The best v1 variant was object+aux/no-pair in some category-level comparisons, showing that auxiliary engineered features were important.

Interpretation:

- The reduced object representation helped somewhat.
- Object-only models overfit or performed weakly.
- Auxiliary scalar features remained essential.
- Pair features did not clearly improve the final HH-like region.

---

## 10. Full object-level DNNs: v2

The v2 model was built to move beyond reduced features and use full object collections from upstream selected parquet files.

### 10.1 v2 object construction

Objects:

- `b1`
- `b2`
- `lepton`
- `MET`
- `j1`
- `j2`
- `j3`
- `j4`

Here `j1`–`j4` are up to four additional non-b AK4 jet slots.

Object features:

- `log_pt`
- `eta`
- `sin_phi`
- `cos_phi`
- `log_mass`
- `btag`
- `btagphys`
- `charge`
- `is_bslot`
- `is_nonbjet`
- `is_lepton`
- `is_met`
- `present`

Object tensor:

\[
X_{\text{obj}} = (33679, 8, 13)
\]

This means:

\[
\text{events} \times \text{object slots} \times \text{features per object}
\]

or:

\[
33679 \times 8 \times 13
\]

### 10.2 Pair tensor

With 8 object slots, the number of unique unordered object pairs is:

\[
\binom{8}{2} = 28
\]

Each pair has 6 features:

- `log_mass`
- `log_pt`
- `abs_deta`
- `abs_dphi`
- `dr`
- `present`

Therefore:

\[
28 \times 6 = 168
\]

Pair tensor:

\[
X_{\text{pair}} = (33679, 168)
\]

### 10.3 v2 input quality

| Quantity | Value |
|---|---:|
| Events | 33,679 |
| Source files | 134 |
| Object slots | 8 |
| Objects | b1, b2, lepton, MET, j1, j2, j3, j4 |
| Object tensor | \(33679\times8\times13\) |
| Pair tensor | \(33679\times168\) |
| Aux features | 65 |
| Build failures | 0 |
| Mean non-b jets present | 3.5765 |
| Events with 4 non-b jets | 23,618 |

Interpretation:

The v2 input builder worked cleanly. Most events had several non-b jet slots filled, so the v2 model had access to additional hadronic-side information rather than mostly padding.

### 10.4 v2 models tested

| v2 model | Inputs | Purpose |
|---|---|---|
| full | object + pair + aux | complete v2 model |
| obj+aux, no pair | object + aux | tests whether explicit pair branch helps |
| obj+pair only | object + pair | tests whether object/pair inputs can replace engineered scalar features |

### 10.5 v2 DNN results

| v2 model | Test raw AUC | Test weighted AUC | Test AP | Test argmax-HH \(Z_A\) | Test argmax-HH \(N_\mathrm{eff}^{bkg}\) | Test global \(Z_A\) |
|---|---:|---:|---:|---:|---:|---:|
| Full: obj+pair+aux | 0.662 | 0.645 | 0.0656 | 0.00592 | 22.02 | 0.00657 |
| Obj+aux, no pair | 0.676 | 0.601 | 0.0694 | 0.00591 | 29.70 | 0.00608 |
| Obj+pair only | 0.642 | 0.574 | 0.0615 | 0.00541 | 31.40 | 0.00599 |

Interpretation:

- The full v2 model had the best weighted AUC among the v2 DNNs.
- The obj+aux/no-pair model had the best raw AUC/AP but weaker weighted AUC.
- The obj+pair-only model was weaker than the full model.
- This indicates that object and pair information alone did not replace engineered auxiliary physics variables.
- The auxiliary scalar branch remains important.
- The full-object DNN improved representation quality relative to earlier reduced-object studies, but it still did not beat the corrected BDT in the final HH-like region.

---

## 11. Main conclusions from the `HH→bbWW*` study

The corrected `HH→bbWW*` study produced several useful conclusions.

### 11.1 Corrected MET matters

The previous MET setup contained inconsistent MET-derived angular information. After correcting this with consistent `FullReco_MET_MET` and `FullReco_MET_Phi`, the BDT performance became more realistic and lower than the inconsistent-MET version.

### 11.2 The corrected BDT is the strongest practical baseline

Among the corrected models, the simple-topology BDT is the most reliable nominal baseline. It balances:

- AUC,
- S/B,
- \(Z_A\),
- and \(N_\mathrm{eff}^{bkg}\)

better than the DNNs.

The BDT score categories show physically meaningful enrichment: higher BDT scores correspond to better S/B and higher \(Z_A\).

### 11.3 DNNs learn some ranking but do not improve the final region

The DNNs learn non-random HH ranking, especially the v2 full-object model. However, their final HH-like categories do not provide better expected sensitivity than the corrected BDT.

### 11.4 High-weight background tails are a limiting issue

The DNN high-score regions and some BDT high-purity regions are limited by weighted background tails. In several cases, a small number of high-weight background events dominate the weighted yield. Therefore, \(N_\mathrm{eff}^{bkg}\) must be considered together with \(Z_A\).

### 11.5 The one-lepton `bbWW*` region has low starting significance

Even the best corrected BDT category has small absolute \(Z_A\). This suggests that continued tuning of DNN architectures in this exact region is unlikely to produce a large physics improvement.

---

## 12. Reason for pivoting direction

The next step should not be to keep tuning increasingly complex DNNs on the same `HH→bbWW*` one-lepton region.

The current evidence suggests:

1. The signal region has low starting significance.
2. The corrected BDT is already difficult to beat.
3. The DNNs improve ranking slightly but not final sensitivity.
4. Full-object inputs help, but not enough to overcome background dominance.
5. Transformer-like or larger models may require more events to stabilize and may not help if the underlying signal region is weak.
6. The project should remain connected to a realistic HH physics outcome rather than becoming an abstract architecture comparison.

Therefore, the analysis should pivot toward finding a more promising HH channel or boosted kinematic region before investing in advanced ML models.

The pivot does not invalidate the `bbWW*` study. Instead, the `bbWW*` study becomes a completed baseline showing that:

- corrected MET handling was necessary,
- BDTs are strong in this region,
- current DNNs do not improve the final signal region,
- and the channel/region is not the best immediate target for demonstrating a strong ML gain.

---

## 13. New reference context: HH→bbbb, GloParT, RINO, and boosted regions

New reference material points toward `HH→bbbb` as a more relevant HH benchmark for advanced object-level and jet-representation models.

The important interpretation is not that RINO should immediately become the main project. Rather, the useful lessons are:

1. BDTs can remain very strong even when more sophisticated ML models are available.
2. Advanced models should be tested only after defining a promising physics region.
3. A model that does not beat a BDT in the inclusive signal region may still be useful in a particular category or background-dominated region.
4. For HH4b, the question is not simply “does a transformer beat the BDT?” but rather:
   - where does it help,
   - whether it needs more data,
   - whether it performs better in special topologies,
   - and whether it improves sensitivity in a physically meaningful category.

Reference notes from the HH4b/GloParT context:

- RINO is powerful for top-vs-QCD tagging and domain/generalization studies.
- For `HH→bbbb` vs QCD + ttbar, GloParT was not clearly better than the BDT in the referenced comparison.
- It may be worth checking whether GloParT or other embeddings perform better with more data or in more specific regions.
- One possible region of interest is ggF with a VBF-like topology.
- `ggF Bin 1` appears promising in the referenced HH4b categorization.
- If \(S\) and \(B\) are of the same order, use \(S/\sqrt{S+B}\) as a rough significance proxy, not \(S/\sqrt{B}\).

This supports the following strategy:

> First find the promising HH channel/category/kinematic region. Then test BDT, SPA-Net, and embedding-based variants only in that well-defined benchmark.

RINO should remain an optional later comparison if time allows, especially if the analysis becomes focused on jet representation robustness, low-label performance, or domain-shift behavior. It should not be prioritized over defining the channel/region, building a strong BDT baseline, and testing SPA-Net or SPA-Net with jet embeddings.

---

## 14. New proposed direction

The next research direction should be:

\[
\textbf{Identify a more promising HH channel or kinematic region, then compare strong baselines and assignment-aware/object-level ML models.}
\]

The most promising near-term channels to scout are:

1. boosted or resolved `HH→bbbb`,
2. `HH→bbγγ`,
3. possibly `HH→bbττ` only if the workflow is feasible.

### 14.1 Why `HH→bbbb`

`HH→bbbb` is attractive because:

- it has a large branching fraction,
- it is fully hadronic,
- it has a difficult combinatorics problem,
- it is naturally suited to b-jet assignment and boosted Higgs reconstruction,
- and it is a better match for SPA-Net or jet-embedding methods.

The main challenge is the large QCD/multijet background. Therefore, boosted regions may be especially important because they can improve signal-to-background behavior.

Potential first selections:

- at least 4 b-tagged AK4 jets for resolved reconstruction,
- or AK8/large-radius Higgs candidates if available,
- Higgs candidate masses near 125 GeV,
- boosted cuts such as high Higgs candidate \(p_T\) or high \(m_{HH}\).

### 14.2 Why `HH→bbγγ`

`HH→bbγγ` is attractive because:

- photons provide a clean mass peak,
- the final state is experimentally clean,
- the signal region can be sharply defined,
- and it may be easier to build a clear baseline.

The main drawback is the small branching fraction. It may have low event counts, but it is worth scouting because the background rejection can be much cleaner than `bbWW*`.

### 14.3 Why not immediately focus on RINO

RINO is an interesting self-supervised jet representation method, but it should not be the immediate priority.

RINO should be considered later if:

- a promising boosted jet-based benchmark is found,
- there is time to test jet embeddings,
- and the question becomes robustness, low-label performance, or domain generalization.

The immediate priority should be:

1. define a better physics benchmark,
2. establish a BDT/cut baseline,
3. test whether assignment-aware models such as SPA-Net help,
4. then optionally test embeddings from ParT, JP-JEPA, or RINO-like methods.

---

## 15. Proposed next workflow

### Step 1: Freeze and archive the `bbWW*` work

Keep the completed `bbWW*` outputs as a documented baseline.

Important output directories include:

- `outputs/lepton_met_features_lepemu_rough_recoMET`
- `outputs/topology_features_lepemu_rough_recoMET`
- `outputs/hadronic_w_top_features_lepemu_rough_recoMET`
- `outputs/lbn_inputs_lepemu_rough_recoMET_v0`
- `outputs/lbn_inputs_lepemu_fullobjects_v2`
- `outputs/lbn_fullobjects_dnn_v2_full`
- `outputs/lbn_fullobjects_dnn_v2_obj_aux_no_pair`
- `outputs/lbn_fullobjects_dnn_v2_obj_pair_only`

The `bbWW*` study can be summarized as:

> After correcting recoMET usage, the simple-topology BDT remains the most reliable nominal baseline. Full-object DNN inputs improve signal ranking somewhat but do not improve the final HH-like signal region. The low absolute expected significance motivates pivoting to a more promising HH channel or kinematic region.

---

### Step 2: Make a compact `bbWW*` final summary table

Create one final summary table for the `bbWW*` study.

