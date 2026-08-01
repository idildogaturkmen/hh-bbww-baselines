# Independent HH evaluation plan

## Trigger

Begin this workflow only after receiving an exact complete ROOT-file path or
base directory from the dataset contact.

## Stage A — path and schema canary

Use one remote file and no persistent outputs.

Verify:

- XRootD access;
- highest valid ROOT tree cycle;
- required 87-branch intersection;
- `pass_selection`;
- `pass_4j3b_selection`;
- particle-vector consistency;
- particle multiplicity and fraction above 256;
- finite and positive logarithm inputs.

## Stage B — released-model canary

Select 64 events uniformly across the file.

Run:

- exact 19-feature preprocessing;
- first-256-particle truncation;
- all three ONNX models in isolated processes;
- ensemble averaging.

Require:

- output shape `[N, 138]`;
- finite values;
- softmax normalization within `1e-4`;
- no persistent output;
- repository unchanged.

## Stage C — spread pilot

Process at least 1,000 uniformly or randomly distributed selected events.

Record:

- mean signal, QCD, and ttbar probabilities;
- fractions above fixed signal thresholds;
- per-model and ensemble agreement;
- signal-grid marginals;
- top mass-grid cells;
- truncation rates;
- 4j3b efficiency.

Do not interpret the first N sequential events as an unbiased sample.

## Stage D — official DCB reconstruction

Construct a temporary `Events` tree containing:

- `pass_selection`;
- `pass_4j3b_selection`;
- `score_0` through `score_137`.

Run the official DCB fitter at the released threshold.

Report separately:

- primary optimizer success;
- physical primary peak in 40–200 GeV;
- valid non-sentinel secondary peak;
- physical secondary peak;
- coarse HH localization near `(mH, mH)`;
- out-of-range and sentinel rates.

## Stage E — independent discrimination

Evaluate the independent HH sample against independent QCD and ttbar.

Required outputs:

- HH versus QCD ROC;
- HH versus ttbar ROC;
- HH versus combined-background ROC;
- fixed operating points;
- event-count and process-macro metrics;
- bootstrap uncertainty where sample size permits.

No training sample may be used as the signal evaluation sample.

## Stage F — weighting

Remain unweighted until the two `gen_weight` entries are authoritatively
defined.

Once confirmed, preserve both:

- unweighted model-performance metrics;
- physically weighted yield/significance metrics.

## Stage G — Track A transfer

After unbiased released-model evaluation is complete:

1. construct the same particle representation for Track A;
2. evaluate the released model without retraining;
3. compare with cut, BDT, DNN, LBN, and SPA-Net baselines;
4. test embedding transfer or initialization;
5. evaluate domain robustness and fixed-mass performance.

## Persistent storage rule

Do not transfer the full external dataset to persistent EOS until
`eosquota` visibly reports 5 TB.
