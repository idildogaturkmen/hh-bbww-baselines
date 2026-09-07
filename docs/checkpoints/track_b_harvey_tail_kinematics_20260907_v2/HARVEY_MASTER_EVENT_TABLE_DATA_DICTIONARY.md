# Data dictionary: HARVEY_MASTER_EVENT_TABLE.parquet

**27,449 rows (u>3.5), of which 18,170 also satisfy u>4.5** (flagged via the `u` column itself, not a separate file). One row per event. Built from: signal — fresh read-only extraction across all 200 `signal_holdout_A.tsv` files (u>3.5 interpreted as score>0.99968377, per Harvey's "u>3/5" flagged as u>3.5 for this analysis); QCD — the frozen `FULL_QCD_SURVIVOR_SIDECAR.h5` (local, no live read needed, restricted to `pass_mask_spanet_10m` bit0); non-QCD (ttbar/ZJetsToQQ/SingleTop/TTbarW/TW/TTbarZ/ttH) — targeted fresh read-only extraction of the 42 events identified via the frozen `job_summary_*.json` survivor lists joined to `nonqcd_tail_topology.json`'s source paths.

## Identity

| column | type | description |
|---|---|---|
| `process` | str | `signal`, `QCD`, `inference_ttbar_1`, `inference_ttbar_2`, `ZJetsToQQ`, `SingleTop`, `TTbarW`, `TW`, `TTbarZ`, `ttH` |
| `background_class` | str | `signal`, `QCD`, `ttbar` (both ttbar lanes merged), `other_background` (everything else) |
| `source_file` | str | remote ROOT file path |
| `file_index` | int | index into the process's own frozen file list |
| `entry_index` | int | native ROOT tree entry index within that file |
| `event_uid` | str | `process\|file_index\|entry_index`, verified unique across the table |
| `label` | int | 1=signal, 0=background |
| `weight_450fb` | float | frozen per-process flat analysis weight at 450 fb⁻¹ (signal: `0.000473265`; backgrounds: the already-published per-process constants) |
| `score` | float | frozen SPA10M class-1 softmax score |
| `u` | float | `-log10(1-score)`, monotonic re-expression, **not a calibrated probability** |

## Global / event

`n_selected_jets`, `HT`, `mHH` (leading-four invariant mass), `pT_HH`, `eta_HH`, `RHH` (leading-four argmin-pairing di-Higgs-mass residual), `met_pt`, `met_phi` (MET; **not extracted for QCD** — the frozen sidecar predates this request and a live MET-only re-read was judged out of scope for the QCD side given its already-complete kinematics; QCD rows have `met_pt=NaN`, excluded automatically from every MET-based statistic), `n_btag_loose` (count of selected jets with `probB > 0.0243`, the Yang & Li arXiv:2508.15048v2 loose SophonAK4 working point — an already-published threshold, reused not re-derived).

## Higgs candidates (leading-four-by-pT geometric pairing, argmin R_HH over 3 combinatorial pairings — identical formula/convention to every prior package in this project)

`mH1`, `mH2`, `pT_H1`, `pT_H2`, `DeltaR_bb_H1`, `DeltaR_bb_H2`, `DeltaR_HH`, `DeltaEta_HH`, `DeltaPhi_HH` (new this package, same pairing convention), `mass_asym`, `pt_asym` (`|pT_H1-pT_H2|/(pT_H1+pT_H2)`).

## Individual selected jets, ranks 1–6 (`jet{k}_*`, k=1..6, descending pT, zero-padded if fewer selected)

`jet{k}_pt`, `jet{k}_eta`, `jet{k}_phi`, `jet{k}_mass`, `jet{k}_probB`, `jet{k}_probC`, `jet{k}_probL`.

## Assignment

`has_genuine_assignment` (bool) — **True only for QCD**; no genuine frozen SPA-Net assignment exists for signal or any non-QCD process anywhere in this project (established repeatedly across every prior package). `assignment_eq_leading4`, `j5_in_assignment`, `n_unassigned_selected_jets` — **QCD only**, NaN elsewhere.

## Fifth/extra jet (rank-5, 0-indexed slot 4)

`pt5`, `eta5`, `probB5`, `probC5`, `probL5`, `min_dR_j5_leading4` (min ΔR to any of the leading-4-by-pT jets), `nearest_of_four_idx`, `pt5_over_pt4`, `pt5_over_HT`, `has_5th_jet` (bool, `n_selected_jets>=5`).

## Signal truth (this analysis's own convention — see caveat below)

`n_truth_bhadrons_fromhh` (count of `gen_bhadron_fromhh==True` per event, expect 4), `j5_is_truth_matched_to_HH_b`, `n_truth_matched_in_leading4`, `truth_all4_in_leading4` — all **signal only**, NaN elsewhere.

**Truth-matching caveat, stated explicitly:** a selected jet is called "truth-matched" if it is the nearest selected jet (min ΔR) to a `gen_bhadron` with `fromhh==True`, and that ΔR < 0.4 (AK4 cone). **This is this analysis's own, explicitly-stated convention** — it is not verified to be identical to whatever convention (if any) was used when SPA-Net's own training targets were built at H5-construction time; it is a standard, defensible choice, not an invented or arbitrary one, but the distinction is real and disclosed.
