# Expanded HH4b cut baseline

This permanent runner establishes the candidate-level cut baseline for
the expanded 5M-background/200k-signal source population.

It opens only the frozen train cache. Validation remains closed until
the BDT models and train-derived thresholds are frozen.

## Mass-plane geometry

The implemented variable is

\[
R_{HH}
=
\sqrt{
(m_{H_1}-125~\mathrm{GeV})^2+
(m_{H_2}-120~\mathrm{GeV})^2
}.
\]

The runner reports:

- reference signal region: \(R_{HH}<30\);
- reference control region: \(30\le R_{HH}<55\);
- optimized nominal cut: \(R_{HH}<34\);
- rounded alternatives: \(R_{HH}<31.5\) and \(R_{HH}<35.5\).

This is a CMS-aligned mass-plane benchmark on the current Delphes
candidate reconstruction, not a reproduction of the full CMS
background-estimation and statistical procedure.

## Interpretation

The weighting is hierarchical development balancing. It is not
cross-section, luminosity, or physical-yield normalization.

The output distinguishes:

- rejection fraction: \(1-\epsilon_B\);
- inverse background efficiency: \(1/\epsilon_B\).

## Usage

```bash
python scripts/analysis/run_hh4b_expanded_cut_baseline.py \
  --config configs/baselines/hh4b_expanded_cut_bdt_protocol_v2.json \
  --output-dir outputs/agent_runs/hh4b_expanded_cut_baseline_train_20260728_v1 \
  --source-commit <FULL_COMMIT_SHA>
```
