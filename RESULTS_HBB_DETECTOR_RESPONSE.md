# H->bb Detector Response Diagnostic

This diagnostic follows up on the low truth-matched reconstructed H->bb mass in the COLLIDE-1M HH->bbWW sample.

It is a detector/reconstruction check, not a new classifier.

## Command

```bash
/tmp/hh-bbww-venv/bin/python scripts/diagnose_hbb_detector_response.py --n-files 2 --bootstrap 500
```

## Event Selection

- Select status-23 generator b quarks using the same practical convention as the existing HH preprocessing.
- Match each generator b quark to the closest retained AK4 jet.
- Require two distinct matched AK4 jets with DeltaR < 0.4.
- Use MAX_JETS = 12.

## Event Counts

- scanned events: 17967
- usable matched events: 12174
- Usable fraction: about 67.8%
- no two status-23 b quarks: 0
- unmatched b quark: 5755
- both b quarks matched to same jet: 38

## Overall Matched m_bb

| n | Mean [GeV] | Median [GeV] | 16-84% [GeV] | 90 < m_bb < 140 |
|---:|---:|---:|---:|---:|
| 12174 | 108.71 | 103.18 | 80.39-129.66 | 0.615 |

## Matched m_bb vs Lower Matched-Jet pT

| lower matched-jet pT bin [GeV] | n | median m_bb [GeV] | bootstrap median 16-84% [GeV] | event 16-84% [GeV] |
|---:|---:|---:|---:|---:|
| 20-30 | 2745 | 90.40 | 90.02-90.78 | 71.67-111.61 |
| 30-40 | 2445 | 99.35 | 98.82-99.71 | 82.68-119.58 |
| 40-50 | 1953 | 106.73 | 106.36-107.36 | 89.93-126.82 |
| 50-65 | 1799 | 115.40 | 114.89-116.02 | 95.67-138.93 |
| 65-80 | 843 | 118.81 | 117.95-119.64 | 100.19-158.97 |
| 80-95 | 424 | 124.57 | 123.15-126.38 | 105.26-174.38 |
| 95-110 | 272 | 123.53 | 121.11-125.78 | 103.67-192.24 |
| 110-125 | 165 | 127.93 | 126.52-130.88 | 108.78-194.43 |
| 125-140 | 99 | 126.24 | 123.09-128.93 | 101.86-182.25 |
| 140-160 | 99 | 133.60 | 129.25-139.82 | 108.12-256.61 |

## Eta Category Summary

| Eta category | n | Mean [GeV] | Median [GeV] | 16-84% [GeV] | 90 < m_bb < 140 |
|---|---:|---:|---:|---:|---:|
| both_central_abs_eta_lt_1.5 | 7077 | 109.41 | 103.80 | 80.78-130.51 | 0.616 |
| both_forward_abs_eta_ge_1.5 | 1515 | 103.28 | 100.84 | 77.20-123.12 | 0.603 |
| one_forward_abs_eta_ge_1.5 | 3582 | 109.61 | 103.11 | 80.85-130.92 | 0.616 |

## Lower Matched-Jet pT Cut Summary

| pT cut [GeV] | Signal retention | n | Mean [GeV] | Median [GeV] | 16-84% [GeV] |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.901 | 10971 | 111.43 | 105.39 | 83.67-131.69 |
| 30 | 0.676 | 8226 | 117.77 | 110.28 | 90.25-136.88 |
| 40 | 0.475 | 5781 | 124.47 | 114.89 | 95.45-144.15 |
| 50 | 0.314 | 3828 | 132.01 | 119.34 | 99.56-157.92 |
| 70 | 0.137 | 1664 | 147.37 | 125.42 | 104.05-184.47 |
| 90 | 0.072 | 880 | 158.72 | 128.91 | 107.01-207.44 |
| 110 | 0.040 | 490 | 170.22 | 132.86 | 109.79-222.10 |

## Interpretation

- If the median matched m_bb moves toward 125 GeV as the lower matched-jet pT increases, the low mass is correlated with low-pT matched b jets.
- If reco/gen pT response is below 1, the matched reconstructed AK4 jet captures less pT than the generator-level b quark.
- If the response is worse at low pT or high |eta|, this supports detector response, acceptance, thresholds, or AK4 containment as contributors to the low m_bb.
- If pT cuts move m_bb closer to 125 GeV but remove many events, then this effect is relevant but cannot simply be fixed by a hard cut without losing signal.

## Outputs

- `outputs/hbb_detector_response/event_response_diagnostics.csv`
- `outputs/hbb_detector_response/min_pt_profile.csv`
- `outputs/hbb_detector_response/perjet_response_profile_vs_reco_pt.csv`
- `outputs/hbb_detector_response/perjet_response_profile_vs_abs_eta.csv`
- `outputs/hbb_detector_response/eta_category_summary.csv`
- `outputs/hbb_detector_response/min_pt_cut_summary.csv`
- `outputs/hbb_detector_response/summary.json`
- `outputs/plots/hbb_detector_response/*.png`
    - matched_mbb_profile_vs_min_pt_with_errors.png -> It shows the median truth-matched reconstructed m_bb as a function of the lower-pT matched b jet.

    20–30 GeV lower jet pT: median m_bb ≈ 90.4 GeV
    30–40 GeV: median ≈ 99.3 GeV
    40–50 GeV: median ≈ 106.7 GeV
    50–65 GeV: median ≈ 115.4 GeV
    80–95 GeV: median ≈ 124.6 GeV
    110–125 GeV: median ≈ 127.9 GeV
    140–160 GeV: median ≈ 133.6 GeV

    The mathced m_bb rises with the lower matched b-jet pT. This suggests the low inclusive m_bb is strongly correlated with low-pT matched b jets.

    - n_events_vs_min_pt_bin.png

    20–30 GeV: 2745 events
    30–40 GeV: 2445 events
    40–50 GeV: 1953 events
    50–65 GeV: 1799 events
    65–80 GeV: 843 events
    80–95 GeV: 424 events
    95–110 GeV: 272 events
    110–125 GeV: 165 events
    125–140 GeV: 99 events
    140–160 GeV: 99 events

    Most usable events have a low-pT matched b jet. Since these low-pT bins also have lower reconstructed m_bb, they dominate the inclusive mass distribution.


    - reco_over_gen_pt_response_vs_reco_pt.png -> checks the matched b-jet response:

        response = reconstructed AK4 jet pT / generator-level b-quark pT

    The reco/gen pT response is lower at low reconstructed jet pT and approaches 1 at higher pT. This supports the idea that low-pT matched b jets are under-reconstructed or less well captured, contributing to the low m_bb.

    - reco_over_gen_pt_response_vs_abs_eta.png -> This plot checks whether response changes with detector region, using |eta|.

    The median response stays roughly around 0.8–0.87 across eta bins. There is no dramatic eta trend here. There may be some variation, but it looks weaker than the pT dependence.

    - matched_mbb_by_eta_category.png -> compares m_bb for:

        both b jets central: |eta| < 1.5
        one b jet forward: one |eta| >= 1.5
        both b jets forward: both |eta| >= 1.5

        both central: n = 7077, mean = 109.4, median = 103.8
        one forward: n = 3582, mean = 109.6, median = 103.1
        both forward: n = 1515, mean = 103.3, median = 100.8

        The both-forward category is slightly lower, but not dramatically. The central and one-forward categories are very similar.

    - matched_mbb_by_min_pt_cut.png -> This plot asks: what happens if you require the lower-pT matched b jet to be above a threshold?

    The median moves upward:
        lower b-jet pT > 20 GeV: median 105.4 GeV
        > 30 GeV: median 110.3 GeV
        > 40 GeV: median 114.9 GeV
        > 50 GeV: median 119.3 GeV
        > 70 GeV: median 125.4 GeV
        > 90 GeV: median 128.9 GeV
        > 110 GeV: median 132.9 GeV

    But notice the mean becomes much larger than the median at high pT. For example:

        pT > 90 GeV: mean 158.7, median 128.9
        pT > 110 GeV: mean 170.2, median 132.9

    That means the distribution has high-mass tails. So the median is safer to discuss.

    - signal_retention_by_min_pt_cut.png -> his is the “cost” plot. It shows how many usable HH events survive each pT cut.
    The retention drops quickly:

        pT > 20 GeV: keeps 90.1%
        pT > 30 GeV: keeps 67.6%
        pT > 40 GeV: keeps 47.5%
        pT > 50 GeV: keeps 31.4%
        pT > 70 GeV: keeps 13.7%
        pT > 90 GeV: keeps 7.2%
        pT > 110 GeV: keeps 4.0%

    So although cutting harder improves the mass scale, it removes a huge fraction of signal. This is why a hard pT cut is not a simple solution.

## Short Interpretation of Detector-Response Diagnostic

This diagnostic shows that the low truth-matched reconstructed H→bb mass is strongly correlated with the lower-pT matched b jet. In the full two-file scan, 17,967 HH→bbWW events were scanned, and 12,174 events had two status-23 generator b quarks matched to two distinct retained AK4 jets with ΔR < 0.4.

The inclusive matched m_bb has mean 108.7 GeV and median 103.2 GeV. However, when the lower matched b-jet pT is binned, the median m_bb increases from about 90.4 GeV in the 20–30 GeV bin to about 124.6 GeV in the 80–95 GeV bin and about 127.9 GeV in the 110–125 GeV bin. This indicates that the low inclusive mass is mainly associated with events where at least one matched b jet is relatively soft.

The matched b-jet reco/gen pT response also increases with reconstructed jet pT, approaching 1 at high pT. This supports the interpretation that low-pT matched b jets are less well reconstructed or capture less of the generator-level b-quark momentum, contributing to the downward m_bb shift.

The eta dependence is weaker. Events with both matched b jets forward have a slightly lower median m_bb than central events, but the difference is smaller than the lower-pT dependence. This suggests that pT dependence is the dominant effect in this diagnostic.

Applying cuts on the lower matched b-jet pT moves the median m_bb closer to 125 GeV, but at a large signal-efficiency cost. For example, requiring the lower matched b jet to have pT > 70 GeV gives median m_bb ≈ 125.4 GeV, but retains only about 13.7% of usable HH events. Therefore, the low m_bb should not simply be “fixed” with a hard pT cut; it motivates understanding detector response, jet containment, and possible b-jet energy corrections/regression.
