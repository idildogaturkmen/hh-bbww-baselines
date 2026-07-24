# HH4b exact observed-transition cut optimization

## Result classification

This is a **paper-quality train-only optimization result**.

It uses every distinct observed train-sample transition in
\(R_{HH}\) between 31 and 36.5, with the object acceptance fixed to

\[
p_{\mathrm{T}}^{\min}=30~\mathrm{GeV},
\qquad
|\eta|^{\max}=2.5.
\]

No validation or test events were accessed. Physical background
normalization and significance were not used.

## Exact optimum

The exact observed-transition maximum is

\[
R_{HH} <
34.127420538,
\]

with

\[
\frac{\epsilon_S^{\mathrm{bal}}}
{\sqrt{\epsilon_B^{\mathrm{fam}}}}
=
1.247785.
\]

It selects 6,156 signal rows and
11,096 background rows.

## Nominal rounded train-only working point

The frozen nominal train-only point is

\[
\boxed{
p_{\mathrm{T}}^{\min}=30~\mathrm{GeV},
\quad
|\eta|^{\max}=2.5,
\quad
R_{HH}<34
}.
\]

The rounded threshold is inside the contiguous 99.5%-of-maximum interval

\[
33.7908
<
R_{HH}^{\max}
<
34.1368.
\]

This stable component contains
233 observed transitions.

The exact transition improves the proxy over the rounded cut by only
0.221%.
The rounded threshold is therefore preferred for reproducibility,
portability across samples, and interpretability.

## Efficiency–purity shortlist

| Role | Threshold | Signal | Background | Balanced proxy | Purity proxy |
|---|---:|---:|---:|---:|---:|
| exact_proxy_maximum | 34.1274 | 6,156 | 11,096 | 1.247785 | 2.521104 |
| stable_high_purity | 33.839 | 6,107 | 10,948 | 1.243401 | 2.523397 |
| higher-purity rounded anchor | 31.5 | 5,709 | 9,790 | 1.233757 | 2.657536 |
| fine-grid proxy optimum | 34 | 6,136 | 11,042 | 1.245031 | 2.518168 |
| higher-efficiency rounded anchor | 35.5 | 6,367 | 11,805 | 1.243596 | 2.421135 |

## Interpretation

- The exact scan confirms a genuine interior optimum near 34.
- Signal and background efficiencies both rise as the threshold is loosened.
- Purity decreases with increasing threshold.
- \(R_{HH}<31.5\) is the higher-purity rounded alternative.
- \(R_{HH}<35.5\) is the higher-efficiency rounded alternative.
- \(R_{HH}<34\) is the nominal train-only compromise.

The final physical working point remains conditional on process-level
background normalization, QCD stitching, finite-MC uncertainty, and the
background-systematic model.

## Figures

- [Exact balanced-proxy scan](exact_rhh_balanced_proxy.svg)
- [Exact efficiency curves](exact_rhh_efficiencies.svg)
- [Exact purity curve](exact_rhh_purity_proxy.svg)
- [Exact Pareto trajectory](exact_rhh_pareto.svg)

PDF versions are included for direct paper use.

## Reproducibility

- Source implementation commit:
  `0615ce2506979c7d21e72112e8445224cf5f8547`
- Exact summary SHA-256:
  `50b9d5f3d123031ba302a9e1b48130b81fb080ccec586f44913b8ebc779a45c1`
- Exact shortlist SHA-256:
  `6196cfcc4a476310a0a93d8aaf1686b1cf156f6d66471dc76e62a845f33e1da2`
- Rounded anchors SHA-256:
  `407eb7a86324a7c4c76769034853002e6019d16d177e986d61fbbb6074365886`
- Cache SHA-256:
  `ac9d088152fa333dcce36c77e6c891d0c446b29f35aeb12c09d33d3133690be0`

## Next gate

Freeze the physical background normalization and systematic model, then
re-rank the frozen rounded working points without reopening geometric
cut optimization.
