# HH4b fine cut-scan checkpoint

## Status

This checkpoint records a **provisional train-only fine-scan result**. It is not yet the final physical optimized cut baseline.

- Grid shape: 9 jet-threshold values × 3 eta values × 29 RHH values
- Grid points: 783
- Pareto-frontier points: 72
- Points within 1% of the optimum: 13
- Shortlisted Pareto points: 11
- Validation rows evaluated: 0
- Test candidate files opened: 0
- Physical significance computed: no
- Source implementation commit: `ad7f08c67c98290cfb93ccb8eecc4ca43c8af493`

## Fine proxy optimum

\[
p_{\mathrm{T}}^{\min}=30~\mathrm{GeV},
\qquad
|\eta|^{\max}=2.5,
\qquad
R_{HH}<34.
\]

The train-only family- and mode-balanced proxy is

\[
\frac{\epsilon_S^{\mathrm{bal}}}
{\sqrt{\epsilon_B^{\mathrm{fam}}}}
=
1.245031.
\]

The fine scan reproduces the coarse optimum exactly, with a proxy change of +0.000% from the coarse result and +31.98% from the fixed reference point.

## Main interpretation

1. Raising the common jet threshold above 30 GeV reduces the best achievable proxy. No additional common jet-\(p_{\mathrm{T}}\) cut is favored beyond the reconstruction requirement.
2. The proxy improves monotonically from \(|\eta|<2.4\) to \(|\eta|<2.5\), favoring the full reconstructed acceptance.
3. The meaningful interior optimization is in \(R_{HH}\). The global fine-grid maximum remains at 34.
4. The objective is step-like because the selected sample changes only when a threshold crosses an observed event value.
5. Thirteen points lie within 1% of the maximum. This supports a stable efficiency–purity plateau rather than a uniquely determined physical working point.
6. The final point may move after process cross sections, QCD overlap removal, finite-MC uncertainty, and background systematics are included.

## Named-point comparison

| Quantity | Reference | Coarse/fine proxy optimum |
|---|---:|---:|
| Point | `pt40_eta2p4_rhh30` | `pt30_eta2p5_rhh34` |
| Selected signal rows | 2,942 | 6,136 |
| Selected background rows | 4,225 | 11,042 |
| Balanced signal efficiency | 0.294979 | 0.615568 |
| Family-balanced background efficiency | 0.097782 | 0.244451 |
| Balanced \(S/\sqrt{B}\)-like proxy | 0.943327 | 1.245031 |
| Balanced \(S/B\)-like proxy | 3.016712 | 2.518168 |

## Fine-grid shortlist

| Point | \(p_{\mathrm{T}}^{\min}\) [GeV] | \(|\eta|^{\max}\) | \(R_{HH}^{\max}\) | Signal | Background | Balanced \(S/\sqrt{B}\)-like proxy | Balanced \(S/B\)-like proxy |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pt30_eta2p5_rhh34` | 30.0 | 2.50 | 34.0 | 6,136 | 11,042 | 1.245031 | 2.518168 |
| `pt30_eta2p5_rhh35p5` | 30.0 | 2.50 | 35.5 | 6,367 | 11,805 | 1.243596 | 2.421135 |
| `pt30p5_eta2p5_rhh34` | 30.5 | 2.50 | 34.0 | 5,972 | 10,669 | 1.242127 | 2.575466 |
| `pt30p5_eta2p5_rhh35p5` | 30.5 | 2.50 | 35.5 | 6,196 | 11,403 | 1.240565 | 2.476036 |
| `pt30_eta2p5_rhh33p5` | 30.0 | 2.50 | 33.5 | 6,045 | 10,769 | 1.239592 | 2.533673 |
| `pt30_eta2p5_rhh35` | 30.0 | 2.50 | 35.0 | 6,293 | 11,555 | 1.238681 | 2.430359 |
| `pt30p5_eta2p5_rhh33p5` | 30.5 | 2.50 | 33.5 | 5,883 | 10,406 | 1.237014 | 2.592831 |
| `pt30_eta2p5_rhh36` | 30.0 | 2.50 | 36.0 | 6,422 | 12,064 | 1.234908 | 2.366967 |
| `pt30_eta2p5_rhh31p5` | 30.0 | 2.50 | 31.5 | 5,709 | 9,790 | 1.233757 | 2.657536 |
| `pt30_eta2p45_rhh35p5` | 30.0 | 2.45 | 35.5 | 6,257 | 11,657 | 1.233152 | 2.422257 |
| `pt30_eta2p5_rhh34p5` | 30.0 | 2.50 | 34.5 | 6,216 | 11,304 | 1.233024 | 2.438024 |

## Figures

- [Best proxy versus jet threshold](fine_best_proxy_vs_jet_pt_min_GeV.svg)
- [Best proxy versus eta acceptance](fine_best_proxy_vs_jet_abs_eta_max.svg)
- [Best proxy versus RHH threshold](fine_best_proxy_vs_rhh_sr_max.svg)
- [Heatmap for eta 2.4](fine_proxy_heatmap_eta2p4.svg)
- [Heatmap for eta 2.45](fine_proxy_heatmap_eta2p45.svg)
- [Heatmap for eta 2.5](fine_proxy_heatmap_eta2p5.svg)
- [Fine-grid Pareto plane](fine_scan_pareto.svg)

## Reproducibility

- Cache SHA-256: `ac9d088152fa333dcce36c77e6c891d0c446b29f35aeb12c09d33d3133690be0`
- Configuration SHA-256: `5958cc97bd38b0595298fb3b275a1ac9eb796d370478c119f29a18caae53906f`
- Fine summary SHA-256: `9905bedbc297409936a58b7cb0efd79a45c646adce9d6504521935d934b2c9b8`
- Fine shortlist SHA-256: `53cdd8279244c23067eb8bda9a00250dabf46201a15ffb685b180080fe8160bb`
- Fine plateau SHA-256: `c529f59af3e32bf91bd89353aa1c7d2a62260bfad5af13d193b72f7e8cc581d5`

The event-level cache is intentionally not committed.

## Next gate

Fix the acceptance at the reconstruction-level values

\[
p_{\mathrm{T}}^{\min}=30~\mathrm{GeV},
\qquad
|\eta|^{\max}=2.5,
\]

and scan every distinct train-sample \(R_{HH}\) transition in the fine plateau. This provides the exact discrete equivalent of a continuous threshold optimization.

Physical ranking, validation confirmation, and test evaluation remain separate later gates.
