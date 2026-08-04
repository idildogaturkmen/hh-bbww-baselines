# Full 5.2M sample split and physical-weighting note

**Date:** 4 August 2026

**Status:** working paper note; audited through PN-c7 Step 0D

**Scope:** Delphes simulation study. Validation and test candidate payloads
remain unopened at this stage.

This note records the generated-event universe, grouped data split,
candidate-level materialization, Run-2-equivalent normalization, physical
yield formulas, multijet transfer, training-weight policy, and remaining
analysis gates. It is intended to provide a stable reference when writing
the JHEP manuscript and when implementing the final full-statistics model
comparison.

## 1. Generated-event universe

The authoritative expanded manifest contains exactly 5,000,000 generated
background events and 200,000 generated signal events.

| Split | Background | Signal | Total |
|---|---:|---:|---:|
| Train | 3,649,873 | 150,000 | 3,799,873 |
| Validation | 965,584 | 37,000 | 1,002,584 |
| Test | 384,543 | 13,000 | 397,543 |
| **Total** | **5,000,000** | **200,000** | **5,200,000** |

The source-member split is:

| Split | Background members | Signal members | Total members |
|---|---:|---:|---:|
| Train | 377 | 87 | 464 |
| Validation | 102 | 19 | 121 |
| Test | 41 | 4 | 45 |
| **Total** | **520** | **110** | **630** |

The split is source-grouped: no source member may contribute events to more
than one split.

Authoritative manifest:

    docs/checkpoints/hh4b_full_5m_200k_expanded_manifest_split_20260728_v1/
      grouped_split_assignments.tsv

Manifest SHA-256:

    5c517dbea1a7932d3bca56be0c9afd87a4988c4bead06d5a1043eddec7e77ea1

The full-sample compact audit output has SHA-256:

    1431482b57db438bc88f695e90ba8b82492bd3e075724e4fdae94a06e58d5589

### Meaning of full-sample utilization

All 5.2M generated events belong to the acceptance and normalization
accounting. Machine-learning models receive only events that satisfy the
frozen reconstruction and candidate requirements.

The split roles are:

- **Train:** model fitting and source-group out-of-fold development.
- **Validation:** one predeclared evaluation after the development lock.
- **Test:** one final evaluation after the complete analysis lock.

Validation and test events must not contribute to model fitting for the
quoted paper results.

## 2. End-to-end processing chain

The analysis chain is:

    5.2M generated events
      -> authoritative grouped train/validation/test split
      -> Delphes detector simulation
      -> frozen object and event selection
      -> candidate tables and zero-candidate source accounting
      -> generator-weight transport
      -> process normalization
      -> Run-2-equivalent physical weights
      -> exactly-3b and >=4b populations
      -> exactly-3b -> >=4b QCD transfer
      -> split-specific model tables
      -> source-group OOF model development on train
      -> frozen validation gate
      -> final sealed test gate
      -> categorized likelihood and paper results

The current c7q--c7y checkpoints are train-only. They do not contain
validation/test results, observed data, or CMS-approved results.

## 3. Run-2 physical normalization

The analysis exposure is

\[
\mathcal{L}_{\mathrm{Run\,2}}
=
138~\mathrm{fb}^{-1}
=
138000~\mathrm{pb}^{-1}.
\]

For source or process component \(p\), define the approved cross-section
coefficient per unit generator weight as

\[
c_p =
\frac{\sigma^{\mathrm{eff}}_p}
     {\sum_{j\in p} w^{\mathrm{gen}}_j}.
\]

Here, \(\sigma^{\mathrm{eff}}_p\) includes the frozen applicable branching
fractions, filter efficiencies, higher-order factors, overlap-removal
conventions, QCD stitching, and importance-sampling corrections.

The corresponding repository column is
`cross_section_coefficient_pb_per_generator_weight`.

The candidate cross-section contribution is

\[
\Delta\sigma_i
=
w_i^{\mathrm{gen}} c_p.
\]

It is implemented as

    generator_cross_section_contribution_pb
      = generator_nominal_weight
      * cross_section_coefficient_pb_per_generator_weight

The Run-2 yield coefficient per unit generator weight is

\[
k_p^{\mathrm{Run\,2}}
=
\mathcal{L}_{\mathrm{Run\,2}} c_p.
\]

It is implemented as

    run2_yield_coefficient_per_generator_weight
      = run2_luminosity_pb_inverse
      * cross_section_coefficient_pb_per_generator_weight

The signed Run-2 candidate physical weight is

\[
w_i^{\mathrm{Run\,2}}
=
w_i^{\mathrm{gen}} k_p^{\mathrm{Run\,2}}
=
\mathcal{L}_{\mathrm{Run\,2}} \Delta\sigma_i.
\]

It is implemented as

    run2_candidate_physical_weight
      = generator_nominal_weight
      * run2_yield_coefficient_per_generator_weight

Signed physical weights remain signed in yields, histograms, categories,
significance calculations, and likelihood inputs. Absolute weights may be
used only for explicitly defined training-loss stabilization or diagnostic
quantities.

## 4. Direct and transferred multijet projections

### 4.1 Direct >=4b projection

For signal and ordinary simulated backgrounds in the direct >=4b
population,

\[
w_i^{\mathrm{direct}}
=
w_i^{\mathrm{Run\,2}}.
\]

This is implemented as

    direct_projection_physical_weight
      = run2_candidate_physical_weight

The direct >=4b QCD sample is a secondary closure sample, not the primary
multijet prediction.

### 4.2 Primary exactly-3b to >=4b multijet projection

For the exactly-3b QCD template, the inclusive transferred weight is

\[
w_i^{\mathrm{primary,inclusive}}
=
w_i^{\mathrm{Run\,2}}
T_{3b\rightarrow\geq4b}^{\mathrm{inclusive}}.
\]

This is implemented as

    primary_projection_physical_weight_inclusive
      = run2_candidate_physical_weight
      * inclusive transfer factor

For the mass-category-dependent transfer,

\[
w_i^{\mathrm{primary},k}
=
w_i^{\mathrm{Run\,2}}
T_{3b\rightarrow\geq4b}^{k}.
\]

This is implemented as

    primary_projection_physical_weight_mhh_category
      = run2_candidate_physical_weight
      * category-dependent transfer factor

For signal and ordinary >=4b backgrounds, the primary projection weights
reduce to the direct Run-2 candidate physical weight. The transfer modifies
only the promoted QCD template.

## 5. Physical-yield formulas

For process or class \(p\) in selected region \(R\), the expected physical
yield is

\[
Y_{p,R}
=
\sum_{i\in p\cap R} w_i^{\mathrm{phys}}.
\]

For a model operating point with selection indicator \(I_i\),

\[
S
=
\sum_{i\in\mathrm{signal}} I_i w_i^{\mathrm{phys}},
\]

and

\[
B
=
\sum_{i\in\mathrm{background}} I_i w_i^{\mathrm{phys}}.
\]

The corresponding purity proxy is

\[
\frac{S}{B}.
\]

The weighted Monte Carlo variance of a yield is

\[
\mathrm{Var}_{\mathrm{MC}}(Y)
=
\sum_i \left(w_i^{\mathrm{phys}}\right)^2,
\]

with statistical standard deviation

\[
\sigma_{\mathrm{MC}}(Y)
=
\sqrt{\sum_i \left(w_i^{\mathrm{phys}}\right)^2}.
\]

The signed-weight effective event count is

\[
N_{\mathrm{eff}}
=
\frac{
\left(\sum_i w_i^{\mathrm{phys}}\right)^2
}{
\sum_i \left(w_i^{\mathrm{phys}}\right)^2
}.
\]

The source-member bootstrap resamples complete source members. It
propagates finite-source uncertainty but is not a replacement for physical
nuisance parameters in the final likelihood.

## 6. Current train-only aggregate normalization checks

The following are whole-table normalization checks before a final
classifier selection. They are not final paper signal-region yields.

| Population | Rows | Weight column | Aggregate yield | Signed \(N_{\mathrm{eff}}\) |
|---|---:|---|---:|---:|
| Direct >=4b c7q table | 30,705 | `run2_candidate_physical_weight` | 9,211,904.516 | 12.4777 |
| Exactly-3b c7q base table | 108,678 | `run2_candidate_physical_weight` | 166,276,131.750 | 191.4539 |
| Primary inclusive c7s projection | 31,225 | `primary_projection_physical_weight_inclusive` | 4,150,691.071 | 218.0126 |
| Primary mass-dependent c7s projection | 31,225 | `primary_projection_physical_weight_mhh_category` | 3,475,547.206 | 164.6146 |

The exactly-3b base sum precedes application of the
\(3b\rightarrow\geq4b\) transfer. It must not be interpreted as the
signal-region QCD prediction.

The current train-only c7y two-head SPA-Net point estimate is:

| Quantity | Value |
|---|---:|
| Selected signal yield | 44.4647 |
| Selected background yield | 1,322,799.6695 |
| \(S/B\) | \(3.3614\times10^{-5}\) |

These are development OOF point estimates. Common bootstrap evaluation,
full-statistics physical-loss reruns, validation, test, category
covariance, and the final likelihood remain pending.

### Step 0D reporting caveat

The Step 0D exploratory process-summary block accidentally selected the
string-valued `physical_category` column as though it were a numeric
physical-weight column. The printed per-process zeros and `NaN` effective
counts are therefore an audit-reporting error, not zero physical yields.

Process-resolved yields must be regenerated using one of the explicitly
numeric columns:

- `run2_candidate_physical_weight`;
- `primary_projection_physical_weight_inclusive`;
- `primary_projection_physical_weight_mhh_category`;
- `direct_projection_physical_weight`.

## 7. Training weights versus physical evaluation weights

The current c7t--c7y models use `development_hierarchical_weight` for
training and nested development, while physical yields use the signed
projection weights defined above.

For the final full-statistics rerun, the primary classification-loss
contract is intended to be physical-composition-aware and
class-stabilized. For active classification head \(h\),

\[
\widetilde w_i^{(h)}
=
\begin{cases}
\dfrac{|w_i^{\mathrm{phys}}|}
{2\sum_{j\in\mathrm{signal},h}|w_j^{\mathrm{phys}}|},
& i\in\mathrm{signal},\\[1.2em]
\dfrac{|w_i^{\mathrm{phys}}|}
{2\sum_{j\in\mathrm{background},h}|w_j^{\mathrm{phys}}|},
& i\in\mathrm{background}.
\end{cases}
\]

Fold-training weights are then rescaled to mean one for numerical
stability. This preserves physical composition within each active class
while preventing the total Run-2 background normalization from dominating
the loss.

The planned model heads are:

- inclusive signal-versus-background classification;
- nonresonant specialist: signal versus QCD and nonresonant-like
  backgrounds;
- resonant/top specialist: signal versus top-rich, top-associated, and
  single-Higgs backgrounds;
- assignment loss: uniquely matchable signal events with normalized
  absolute physical signal weights.

All quoted physical yields, weighted distributions, category
optimization, significance values, and likelihood results must use signed
physical weights, never the stabilized training-loss weights.

## 8. Step 0D status and unresolved source mapping

Step 0D completed with

    RESULT=PN_C7_PHYSICAL_WEIGHT_AND_TRAIN_COVERAGE_AUDIT_PASS
    NEXT=FREEZE_FULL_SPLIT_PHYSICAL_WEIGHT_CONTRACT

The saved Step 0D output has SHA-256

    96854910dbf054614b537a54cc2c6cdd392f409455b4523cf356199a5e9520a2

The audit established the presence and numerical consistency of the current
train physical-weight columns. It did not complete exact source-member
reconciliation because the split manifest identifies members with
`group_id`, while the c7q materialization registry has no directly matching
`group_id` field.

| Inventory | Members | Generated events |
|---|---:|---:|
| Full train manifest | 464 | 3,799,873 |
| c7q materialized registry | 441 | 3,569,873 |
| Difference requiring crosswalk | 23 | 230,000 |

The 23-member difference may consist entirely of zero-candidate sources,
but this is not yet proven. The next gate must map the two registries
through authoritative source identities and establish whether any unmatched
member has nonzero exactly-3b or >=4b candidate rows.

## 9. Remaining analysis gates

1. Resolve the 464-versus-441 source-member crosswalk.
2. Prove whether all 23 unmatched members have zero candidate rows.
3. Regenerate correct process-resolved physical yields.
4. Freeze the full-split physical-weight and training-loss contract.
5. Freeze model architectures, feature sets, specialist definitions, and
   category-optimization rules.
6. Materialize validation physical tables after the development lock.
7. Keep test sealed until the complete analysis lock.
8. Rerun all models under the common full-statistics contract.
9. Construct category-dependent multijet transfer response and covariance.
10. Perform the final profile-likelihood evaluation.

## 10. Draft manuscript wording

> The simulated dataset comprises 5.0 million background and 0.2 million
> signal events. Source members are assigned as indivisible groups to
> training, validation, and test samples containing 3.800, 1.003, and
> 0.398 million generated events, respectively. All generated events enter
> the acceptance accounting, while machine-learning models are trained only
> on events satisfying the frozen reconstruction and candidate
> requirements. Expected yields are normalized to 138 fb\(^{-1}\) at
> 13 TeV using signed generator weights and process-specific cross-section
> coefficients. The dominant multijet contribution in the >=4b signal
> population is obtained from an exactly-3b control template through a
> frozen transfer model, with direct >=4b multijet simulation reserved for
> closure tests. Model development uses source-group out-of-fold
> predictions on the training sample; validation and test samples are
> evaluated only after predeclared analysis locks.

This wording must be updated after source-member reconciliation, validation
materialization, and the final likelihood are complete.
