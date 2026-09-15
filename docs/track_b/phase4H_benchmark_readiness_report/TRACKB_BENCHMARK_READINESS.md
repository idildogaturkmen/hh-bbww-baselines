# Track B Benchmark Readiness -- Bounded Audit Report

Scope: bounded benchmark-readiness audit of the `pku-hep-group/jetfree-hh4b`
frozen upstream release (commit `4a0b313d55b1939b8665e980dddbf224bc773736`) and
the Track B Phase-4 evidence tree at
`/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812`.

Date: 2026-08-13. No model was trained. No full 150M-event scan was executed.
Track A was not accessed or modified. `gen_weight` was not used as a physical
weight anywhere. `HIGH_STAT_TRANSFER_EXECUTION_AUTHORIZED=false` throughout.

---

## 1. Dataset inventory (re-verified)

| Dataset ID | Files | Exact stored events | Role |
|---|---:|---:|---|
| training_signal_h3var | 500 | 69,750,093 | training signal (h3->h1h2->4b, variable mass) |
| training_qcd | 400 | 62,941,859 | training background |
| training_ttbar | 200 | 11,599,999 | training background |
| inference_qcd_1 | 176 | 27,683,689 | inference background |
| inference_qcd_2 | 381 | 59,939,617 | inference background |
| gghh_kl0 | 1000 | 2,102,188 | inference signal |
| gghh_kl1 | 1000 | 2,538,851 | inference signal |
| gghh_kl5 | 1000 | 1,050,267 | inference signal |
| vbf_cv1_c2v1_kl1 | 300 | 419,198 | inference signal |
| vbf_cv1_c2v2_kl1 | 300 | 1,284,000 | inference signal |
| vbf_cv1_c2v1_kl2 | 300 | 601,268 | inference signal |
| vbf_cv1_c2v1_kl0 | 300 | 430,736 | inference signal |
| vbf_cv0p5_c2v1_kl1 | 300 | 1,153,048 | inference signal |
| vbf_cv1p5_c2v1_kl1 | 300 | 828,202 | inference signal |

All 14 datasets, 6457 files, 1,557,638,260,877 bytes total. Numbers re-verified
against `phase4B_remote_dataset_inventory_r2` and `phase4C1R1_C2_exact_entry_metadata_scan`
evidence in this audit; unchanged from the prior handoff.

QCD lanes combined: training 62,941,859 + inference_1 27,683,689 + inference_2
59,939,617 = **150,565,165** raw stored QCD events.

## 2. Domain / provenance contract

See `TRACKB_DOMAIN_CONTRACT.tsv` for the full field-by-field contract
(generator, shower, Delphes card, gen-level/pre-Delphes/post-Delphes QCD filters,
stored event selection, particle/jet feature lists, continuous SophonAK4 scores,
truth-assignment information, and licensing). Highlights:

- QCD is Pythia HardQCD with `pTHatMin=75`, a custom FastJet-based generator
  filter, a pre-Delphes stable-particle genHT>300 filter, and a post-Delphes
  4-jet kinematic filter -- **conditioned QCD, not inclusive generic QCD**.
- SophonAK4 loose WP = `probB > 0.0243` (eps_B~=10%), tight WP = `probB > 0.643`
  (eps_B~=0.1%); both independently confirmed against the analyzer source code
  AND the paper's own Supplementary Material text (p.10) in this audit.
- Continuous `jet_sophonAK4_probB/probC/probL` scores are retained per jet, not
  just the two derived boolean pass flags.
- No event/run/lumi/job-identity branches exist in the QCD ntuples.

## 3. Bounded tag-multiplicity canary and full-sample tag audit

Phase-4D (8-file bounded canary) and Phase-4E (full 400-file scan, re-verified
in this audit) both confirm exact loose-2/3/>=4 and tight-2/3/>=4 populations are
non-empty and stable across files within `training_qcd`:

| Population | Full-sample (Phase-4E) | Per-file relative stdev |
|---|---:|---:|
| tight_exact2 | 37,886,782 | 0.20% |
| tight_exact3 | 1,787,986 | 1.49% |
| tight_exact4 | 103,221 | 6.59% |
| loose_exact2 | 28,574,627 | 0.27% |
| loose_exact3 | 31,466,909 | 0.24% |
| loose_exact4 | 2,900,323 | 1.19% |

**Conclusion: clean exclusive 2-tag, 3-tag, >=4-tag populations ARE
constructible within the stored `4j3bor2b`-conditioned phase space**, using
`jet_sophonAK4_probB` thresholded among the first 4 selected jets. The stored OR
preselection does not erase the exclusive hierarchy; it was verified event-by-
event (zero mismatches) that `pass_4j3b_selection`/`pass_4j2b_selection` exactly
reproduce the recomputed flags from the continuous scores.

## 4. Deterministic file-disjoint split (Phase-4F, re-verified in this audit)

| Split | Files | Events | tight2 | tight3 | tight4 |
|---|---:|---:|---:|---:|---:|
| development | 240 | 37,759,585 | 22,731,060 | 1,072,091 | 61,948 |
| holdout_A | 80 | 12,595,807 | 7,582,197 | 357,597 | 20,501 |
| holdout_B | 80 | 12,586,467 | 7,573,525 | 358,298 | 20,772 |

Re-verified in this audit: 400 unique file SHA-256 hashes, 400 unique basenames,
zero duplicates across splits, event/tag sums reconcile exactly to the Phase-4E
full-scan totals. **File-level disjointness is proven** (not statistical
independence -- see Section 6).

## 5. Track-B CMS-inspired mass-plane SR/CR analog (`TRACKB_CMS_INSPIRED_TAG4_SR_ANALOG`)

Constructed and tested on a bounded 8-file / 1,257,684-event canary (Phase-4G,
same files as Phase-4D):

- **Population:** `tight_exact4` (all 4 of the first 4 selected jets pass
  `jet_sophonAK4_probB > 0.643`).
- **Pairing:** deterministic minimum-|m1-m2| combinatorial pairing over the 4
  jets' stored 4-vectors only (`jet_pt/eta/phi/energy`) -- no ML/tagger-score
  input to the pairing itself, per the instruction not to let the jet-free
  model bias the region definition.
- **Region:** `R_HH = sqrt((m1-125)^2 + (m2-125)^2)`; SR = R_HH<30, CR =
  30<=R_HH<55.

| Quantity | Value |
|---|---:|
| tight_exact4 events (canary) | 1,991 |
| SR (R_HH<30) | 92 (4.62% of tight4) |
| CR (30<=R_HH<55) | 225 (11.30% of tight4) |
| outside (R_HH>=55) | 1,674 (84.08% of tight4) |

Cross-check: canary `tight_exact4`=1,991 exactly matches Phase-4D's independently
computed value for the same 8 files. **SR/CR are constructible with non-trivial
raw support within the stored phase space.** An order-of-magnitude projection to
the full 400-file training_qcd sample (103,221 total tight4 x 4.62% canary SR
fraction) suggests ~4,700 raw SR events and ~11,700 raw CR events -- this is
explicitly a projection from a bounded canary, not a scan result, and is NOT to
be used for any yield claim (see `TRACKB_PROVENANCE_GAPS.md` Section 4 for the
caveat that a more paper-faithful uncorrected-kinematic analog would likely
re-center near (110,105)/r=25 rather than (125,125)/R<30).

## 6. QCD statistical independence

Two distinct questions, resolved differently:

- **Generation-provenance-level:** RESOLVED as author-documented. This audit
  independently fetched and read the paper's Supplementary Material (Appendix
  C.1.a, p.10) and confirmed verbatim: "These samples are generated following
  the same procedures used for the inference samples ... while remaining
  statistically orthogonal to the datasets used for model inference." Raw
  generated/post-trigger counts quoted in the paper (2.0e11 raw / 6.3e7
  post-trigger for training QCD) are consistent with the distributed
  training_qcd count (62,941,859).
- **Event-identity-level:** UNRESOLVED. No event/run/lumi/job-ID branches exist
  in the distributed ntuples (87-branch schema identical across all three QCD
  lanes), so no direct event-duplicate check is possible. Zero file-path/
  basename overlap was confirmed, which is necessary but not sufficient for
  event-level independence.

`QCD_STATISTICAL_INDEPENDENCE_AUTHOR_DOCUMENTED=true`;
`QCD_STATISTICAL_INDEPENDENCE_EVENT_ID_VERIFIED=false`.

## 7. Training-QCD physical weight

**`TRAINING_QCD_WEIGHT_CONTRACT_RESOLVED=false`.** Unchanged from the prior
handoff. This audit additionally cross-checked the `weight_dict["QCD"]` formula
denominator `(17600+38072)*5e6=2.7836e11` against the paper's own stated raw
training-QCD generation count of `2.0e11` (Supp. Mat. p.10): same order of
magnitude, ~28% discrepancy, not an exact match. This reinforces rather than
resolves the prior finding that the `17600`/`38072` terms most likely describe
the two inference-QCD campaigns, not the training-QCD campaign. No physical
weight is applied to training_qcd anywhere in this audit.

## 8. Model input readiness

See `TRACKB_MODEL_INPUT_READINESS.tsv` for the full 7-model table with reasons.
Summary:

| Model | Status |
|---|---|
| Enriched XGBoost baseline | READY_NOW |
| Dense DNN | READY_NOW |
| Native SPA-Net | READY_AFTER_DETERMINISTIC_ADAPTER (truth-assignment construction) |
| Two-head SPA-Net | READY_AFTER_DETERMINISTIC_ADAPTER (assignment head gates it) |
| Genuine pairwise-aware ParT | READY_NOW |
| SPA-Net + frozen ParT embeddings | READY_AFTER_DETERMINISTIC_ADAPTER |
| SPA-Net + frozen JP-JEPA embeddings | READY_AFTER_DETERMINISTIC_ADAPTER (JP-JEPA contract unverified) |

## 9. Licensing / citation documentation

No top-level LICENSE in jetfree-hh4b; README requests citation only
(Yang & Li, arXiv:2508.15048, "Yang:2025txy"). `training/weaver-core/` has its
own vendored MIT License. Delphes reader and Pythia main files carry their
respective upstream GPL headers as inherited boilerplate. No legal conclusion is
drawn; see `TRACKB_PROVENANCE_GAPS.md` Section 7.

---

## Final answers

**A. Can Track B currently be used for an unweighted/raw-count ML benchmark?**
**YES -- `RAW_COUNT_ML_BENCHMARK_READY=true`.** Exact stored-event inventories,
a proven-disjoint deterministic file split, a verified continuous-score tag
contract, non-trivial exclusive 2/3/>=4-tag populations, and a constructible
mass-plane SR/CR analog are all in place using raw counts only. This is
independent of the unresolved physical-weight question.

**B. Can it currently be used for a physically normalized 450 fb^-1 or 138
fb^-1 result?** **NOT for training_qcd** --
`PHYSICALLY_NORMALIZED_ANALYSIS_READY=false` for any result that includes
training_qcd yields, because `TRAINING_QCD_WEIGHT_CONTRACT_RESOLVED=false`.
The author's `weight_dict` and 450 fb^-1 convention ARE fully recovered and
could in principle support a physically normalized result restricted to
processes whose weight mapping is closed (e.g. inference QCD, if its own
provenance is separately closed) -- but no such restricted result was computed
in this audit and none is authorized to use training_qcd's yield.

**C. Can training and inference QCD currently be treated as statistically
independent?** **Qualified: YES at the generation-provenance level (author-
documented, independently verified against the primary source in this audit),
NO at the event-identity level (unverifiable from the distributed data).** Use
inference QCD as an author-documented external replication sample only after
all internal Track-B modeling choices are frozen on the development/holdout
splits inside training_qcd; do not claim event-level independence.

**D. Can it support SPA-Net assignment labels?** **READY_AFTER_DETERMINISTIC_
ADAPTER.** Truth Higgs 4-vectors and boolean b-hadron-from-Higgs flags are
stored, but no direct per-jet h1-vs-h2 index is stored. A deltaR/mass
combinatorial matching adapter (design only, not built in this audit) is
required, and applies only to genuine-Higgs signal samples.

**E. Can it support ParT/JP-JEPA constituent encoders?** **ParT: YES
(READY_NOW)** -- full PF-candidate 4-vectors, charge, PID, impact parameters,
and jet-membership labels are stored. **JP-JEPA: READY_AFTER_DETERMINISTIC_
ADAPTER**, conditional on independently confirming Track B's branch set against
JP-JEPA's actual (external, unverified-here) input contract.

**F. What is the one remaining blocker with highest priority?** **The
training-QCD physical-weight contract.** Every raw-count/ML-benchmark question
is answered YES; every physically-normalized-yield question is blocked
specifically because the authors' published `weight_dict["QCD"]` formula cannot
yet be proven to apply to the 400-file training-QCD production (the (17600+38072)
job-count terms structurally match the two inference-QCD campaigns and are ~28%
off from the training campaign's own stated raw-generation count). Closing this
requires either (a) a separate author-confirmed training-QCD generation-job
count, or (b) explicit written confirmation from the authors that the same
`weight_dict["QCD"]` entry is intended to apply uniformly across all three QCD
productions.
