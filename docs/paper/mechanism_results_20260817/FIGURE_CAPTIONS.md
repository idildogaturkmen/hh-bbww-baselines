# Figure Captions

## Figure 1 -- `figures/trackb_holdoutA_mechanism_rejection.{pdf,png}`

**Track-B `holdout_A` mechanism ablation.** Background rejection $R_B =
1/\epsilon_B$ (log scale) versus signal efficiency $\epsilon_S$ for the three
Track-B event-Transformer arms -- E0 (five-jet kinematics only), E1 (E0 +
continuous reconstructed SophonAK4 `probB`/`probC`/`probL`), and E2 (E1 +
the frozen, project-defined pairwise attention bias using $[m_{ij},
|\Delta\eta_{ij}|, |\Delta\phi_{ij}|, \Delta R_{ij}]$) -- evaluated once on
the external `jetfree-hh4b` Delphes benchmark's `holdout_A` sample. Points
are the mean over 3 independently trained seeds; error bars are the
seed-to-seed standard deviation. Marker style encodes the predeclared
statistical-support tier: filled = well-supported ($\geq100$ raw surviving
QCD events); half-transparent filled = limited but reportable (10-99); open
= exploratory only ($<$10), not a primary quantitative claim. The population
is Track-B's `tight_exact4` selection with **no mass-plane cut**. Rejection
values shown are **conditional on this selected population** (they are not,
and are not presented as, an absolute background-rate prediction). `holdout_A`
was opened **exactly once**, by Phase-4M (2026-08-13), reusing the 9 already
-trained Phase-4L checkpoints with no retraining; it is not reopened by this
figure or this package. This is a **Delphes simulation** result on an
**external `jetfree-hh4b` benchmark** -- it is **not CMS data and not a CMS
reproduction**.

## Figure 2 -- `figures/trackb_mechanism_gain.{pdf,png}`

**Track-B mechanism gain.** $R_B(E1)/R_B(E0)$ (gain from adding continuous
reconstructed flavor information) and $R_B(E2)/R_B(E1)$ (additional gain from
the frozen pairwise-attention-bias mechanism) versus $\epsilon_S$, computed
from the same `holdout_A` values as Figure 1. Each point/error bar is the
mean/std of the three seed-paired per-seed ratios (i.e. seed $i$'s E1 divided
by seed $i$'s E0, not an independent-error propagation of the two arms'
means). A horizontal dashed line marks ratio $=1$ (no effect). Marker style
encodes the statistical-support tier of the weaker of the two arms entering
each ratio, using the same three-tier convention as Figure 1. No value is
extrapolated beyond the six measured $\epsilon_S$ grid points. The
continuous-flavor gain ($E1/E0$) sits above 1 at every measured point; the
added pairwise-bias gain ($E2/E1$) sits below 1 at every well-supported or
limited-but-reportable point -- a degradation, not an improvement -- with a
single exception exactly at the `eps_S=0.20`, exploratory/limited boundary
(raw QCD = 10 for one of the three seeds), reported as a discrepancy, not
smoothed over (see `MECHANISM_RESULTS_PAPER_DRAFT.md` section 1). Delphes
simulation, external `jetfree-hh4b` benchmark -- not CMS data, not a CMS
reproduction.

## Figure 3 -- `figures/tracka_epsS040_model_comparison.{pdf,png}`

**Track-A model comparison at $\epsilon_S=0.40$**, from the authoritative,
corrected Track-A evaluation (`post_training_evaluation_20260815_v2`; `v1`
is superseded and not used). Bars: the historical $R_{HH}<34$ rectangular
mass-window cut (a single fixed operating point, not a continuous score);
BDT `CONTROL_0` (34 features); the dense DNN and the **CMS-inspired
five-jet pairwise event Transformer** (both measured in this benchmark,
5-fold out-of-fold $\times$ 3 seeds, error bar = seed-to-seed standard
deviation); `FiveJetParTNet` (a **frozen historical late-fusion jet-level
Transformer reference from Phase-3, cited and not retrained here -- this is
NOT canonical ParT**, and should not be conflated with it); and BDT
`NEW_C` (55 features, +5th jet +pairwise). Hatched bars are frozen reference
points with no confidence interval computed in any artifact this figure
draws from -- no error bar is invented for them. The CMS-inspired
Transformer's point estimate sits above `CONTROL_0` and the dense DNN, but
below both `FiveJetParTNet` and `NEW_C`. Categorized BDT and LBN are not
shown: both are unavailable as trained, population-comparable artifacts
(`EVALUATION_RESULTS.json` -> `references.CATEGORIZED_BDT` /
`references.LBN_Lorentz_Boost_Network`), not omitted for space. Delphes
simulation, Track A -- not CMS data, not a CMS reproduction.

## Figure 4 -- `figures/tracka_dnn_vs_transformer_rejection.{pdf,png}`

**Track-A dense DNN vs. CMS-inspired transformer, $R_B$ vs. $\epsilon_S$**
(top, log scale), **with the frozen paired-bootstrap comparison from v2**
(bottom): $\Delta R_B = R_B^{\rm transformer} - R_B^{\rm DNN}$, point
estimate (primary seed 20260811) with 68%/95% joint (same-resampled-event
-multiset), source-group-stratified bootstrap confidence bands (2000
replicates). The top panel's solid markers/error bars are the 3-seed
fixed-$\epsilon_S$ grid, entirely within the statistical-support gate
(`raw_background_rows`$\,\geq100$ AND `qcd_raw_rows`$\,\geq10$ AND
`qcd_Neff`$\,\geq10$) for both models at every one of the 3 seeds. The open,
dotted-connected markers at the upper left are a **separate, single
-primary-seed-only diagnostic extension** using the fixed-$\epsilon_B$
slicing at $\epsilon_B\in\{10^{-3},10^{-4}\}$ ($R_B\approx1000$ and
$\approx10{,}000$); the shaded red band marks where this extension **fails**
the support gate (`qcd_Neff<10` for both models) and is explicitly **not
presented as a supported physical claim** -- shown only for context, per
Table C. The bottom panel makes the working-point-dependent ordering flip
visible directly: the transformer is favored (68% CI excludes zero) at
$\epsilon_S\gtrsim0.5$, marginal at $\epsilon_S=0.40$, a coin flip at
$\epsilon_S=0.25$, and disfavored (point estimate) at $\epsilon_S=0.10$,
though every interval at that point still includes zero. Delphes simulation,
Track A -- not CMS data, not a CMS reproduction.
