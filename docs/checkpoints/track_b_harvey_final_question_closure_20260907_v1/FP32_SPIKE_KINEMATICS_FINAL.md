# Are the u~6.6/6.9 FP32 spike bins kinematically special?

Track A already established that the spikes near u≈6.62 and u≈6.92 are
float32-probability quantization artifacts of the two-class softmax (the
`k=4` and `k=2` steps on the `2^-24` lattice below 1.0), not discrete
classifier modes — **this task does not reopen that finding**. It asks a
purely kinematic question instead: within the physical signal sample
(`HARVEY_MASTER_EVENT_TABLE.parquet`, u>3.5, 27,339 signal events — **a
different sample from Track A's 400k development cohort; event identity
between the two is NOT assumed or claimed**), do events landing in these
two float32 bins look different from events in neighboring populated bins?

Read-only. No new inference. No frozen file modified.

## Population sizes in the physical sample

| bin | u (nominal) | n signal events (physical sample) | n signal events (Track A 400k dev cohort, for reference only — different sample) |
|---|---:|---:|---:|
| k=2 | 6.9237 | **255** | 99 |
| k=4 | 6.6227 | **228** | 100 |

Both populations are genuinely "hundreds of events," confirming Harvey's
suspicion at face value — in the physical sample. (The two sample sizes
differ because they are different event populations scored the same way;
this is expected and disclosed, not an inconsistency.)

## Method

For each of k=2 and k=4, compared against its nearest populated neighbor
bins (k=4,6 for the k=2 target; k=2,6,8 for the k=4 target — "populated"
defined as ≥5 events), across 13 kinematic features (`n_selected_jets`,
`n_btag_loose`, `HT`, `mHH`, `RHH`, `pT_H1`, `pT_H2`, `jet1_pt`, `jet2_pt`,
`DeltaR_bb_H1`, `DeltaR_bb_H2`, `pt5`, `min_dR_j5_leading4`) using
bootstrap 68%-CI medians and Cliff's delta (a non-parametric, scale-free
effect size; |δ|≥0.33 is the conventional "medium" threshold used here as
the large-effect cutoff).

## Result

**26 (feature × target-bin) comparisons, 0 with |Cliff's δ| ≥ 0.33.** The
single largest effect found is `pt5` at k=2 (δ=0.268, target median 39.2
GeV vs. neighbor median 33.2 GeV) — a real but modest shift, below the
large-effect threshold. Every other feature/bin combination has |δ|<0.27,
most below 0.1. Bootstrap medians for the target bins and their neighbors
overlap within CI for essentially every feature (e.g. k=2 `HT`: 459.5
[454.1,465.8] vs. neighbor 470.0 [461.6,477.7] GeV; k=4 `mHH`: 573.7
[559.3,599.7] vs. neighbor 557.2 [546.5,567.8] GeV).

Plotting median kinematics against the float32 lattice index k across every
populated bin (`figures/fp32_spike_feature_comparison.png`) shows smooth,
monotonic-ish trends through k=2 and k=4 with no visible kink, jump, or
outlier at either target bin. The HT-vs-mHH scatter
(`figures/fp32_spike_ht_mhh.png`) shows the spike-bin events scattered
throughout the bulk of the full u>3.5 signal cloud, not clustered in a
distinct region. Fifth-jet presence at k=2/k=4 (`figures/fp32_spike_
fifth_jet_fraction.png`) is statistically indistinguishable from the
overall u>3.5 signal population's fraction.

Full per-feature table: `FP32_SPIKE_KINEMATICS_TABLE.csv`. Full
median-vs-k dictionary (every populated k in this sample): `work/
fp32_spike_kinematics_result.json`.

## Verdict

**`SPIKE_KINEMATICS = NO_CLEAR_SPECIAL_MODE`**

Events landing in the k=2 and k=4 float32 quantization bins are
kinematically unremarkable — statistically consistent with a smooth draw
from the surrounding signal-tail kinematic distribution, not a distinct
physical population. This is exactly the expected outcome given Track A's
established mechanism (the spikes are an artifact of how the classifier's
already-continuous output is *stored*, not of the physics generating the
events), and this task now confirms that expectation kinematically rather
than merely asserting it.
