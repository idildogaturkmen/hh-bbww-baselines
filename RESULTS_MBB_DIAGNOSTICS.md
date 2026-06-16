# Matched m_bb diagnostics

## Purpose

The truth-matched H -> bb AK4 dijet mass in the current COLLIDE-1M HH -> bbWW preprocessing has a mean near 109 GeV, below the nominal Higgs mass of 125 GeV. This note documents the diagnostic plots and gives a cautious interpretation.

Script:

```bash
/tmp/hh-bbww-venv/bin/python scripts/diagnose_mbb_shift.py
```

Outputs:

- `outputs/plots/mbb_diagnostics/`
- `outputs/mbb_diagnostics/mbb_diagnostic_summary.json`
- `outputs/mbb_diagnostics/mbb_shift_note.md`

## Local COLLIDE-1M result

Using all 12,171 usable HH -> bbWW events in `outputs/hbb_npz/{train,val,test}.npz`:

| Quantity | Value |
|---|---:|
| Mean truth-matched m_bb | 108.71 GeV |
| Median truth-matched m_bb | 103.18 GeV |
| 16-84% interval | 80.40-129.67 GeV |
| Fraction with 90 < m_bb < 140 GeV | 61.5% |
| Fraction with 100 < m_bb < 150 GeV | 47.7% |

Comparison of pair choices:

| Pair choice | Mean m_jj | Median m_jj | Pair accuracy |
|---|---:|---:|---:|
| Truth-matched pair | 108.71 GeV | 103.18 GeV | 1.000 |
| Top-2 b-tag jets | 140.14 GeV | 103.12 GeV | 0.362 |
| Closest m_jj to 125 GeV | 124.19 GeV | 124.78 GeV | 0.093 |
| Combined b-tag + mass | 121.58 GeV | 123.09 GeV | 0.336 |

The closest-to-125 method produces an m_jj distribution centered near 125 GeV by construction, but it is usually not the truth-matched pair. Therefore, a mass peak near 125 GeV is not sufficient evidence of correct H -> bb assignment.

## Kinematic trends

The matched m_bb shift is strongest at low matched-jet pT. In the profile of matched m_bb versus the lower-pT matched jet, the lowest bin has median m_bb about 87 GeV, while bins with lower matched-jet pT around 50-100 GeV have medians closer to 110-125 GeV. This is consistent with a jet-response or threshold-related effect.

The dependence on DeltaR_bb is weaker and non-monotonic in the current profile. Very small DeltaR has a larger mean but broad spread, while the bulk of events across DeltaR roughly 1-3 has medians near 100-105 GeV. This does not isolate out-of-cone radiation as the sole explanation.

## CMS comparison and interpretation

This behavior is not surprising in H -> bb reconstruction. CMS HH -> bbWW analyses reconstruct small-radius jets with anti-kT R=0.4, apply jet energy calibrations, and explicitly note that b-jet energy estimates can be biased by neutrinos in semileptonic b decays and detector response. CMS applies a DNN b-jet energy regression to b-tagged jets, improving b-jet energy resolution by 12-15% in the published HH -> bbWW analysis. See CMS-HIG-21-005 / arXiv:2403.09430, especially the object reconstruction discussion.

CMS also has a dedicated b-jet energy regression paper, CMS-HIG-18-027 / arXiv:1912.06046, describing a DNN method for estimating b-jet energies and resolutions using jet composition, shape, and secondary-vertex information. That paper motivates why raw or simply calibrated b-jet four-vectors are not always optimal for H -> bb reconstruction.

So the qualitative issue is general: b-jet energy reconstruction is hard, and CMS analyses use corrections/regression rather than assuming raw AK4 dijet masses peak exactly at 125 GeV. The specific mean value of 108.7 GeV, however, should be treated as a COLLIDE-1M/simplified-dataset result until checked against generator-level H -> bb masses, particle-level jets, or a CMS-style b-jet regression.

## Current conclusion

The downward truth-matched m_bb shift is likely driven by a combination of reconstructed b-jet response, semileptonic b-hadron decays with neutrinos, finite-radius AK4 jet capture, and simplified COLLIDE calibration. The plots support a strong low-pT dependence, but they do not uniquely identify the cause.

For the learned H -> bb assignment baseline, do not force the model to select pairs only because m_jj is close to 125 GeV. The target should remain the truth-matched pair, and m_bb should be reported as a diagnostic distribution of the selected pair.
