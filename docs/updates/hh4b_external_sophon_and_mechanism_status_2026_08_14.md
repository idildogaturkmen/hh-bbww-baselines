# HH4b External Sophon and Mechanism-Benchmark Status

Date: 2026-08-14

Scope: durable status record reconstructed from the on-disk run records. This
note contains no CMS collision-data artifacts and no internal CMS material. It
distinguishes completed validation from active or incomplete computation.

## Executive status

| Workstream | Status at this snapshot | Paper role |
|---|---|---|
| External Sophon E0/E1/E2 ablation | Complete and holdout-adjudicated | Current reportable controlled ablation |
| Local mechanism benchmark | Active: DNN 15/15, event transformer 6/15 | Supporting architecture study; no pooled result yet |
| External-data adapter and truth labels | Pilot validation complete | Enables official SPA-Net study |
| Paper comparison framework | Complete as schemas and model registry | Reporting infrastructure, not performance results |
| Official SPA-Net data integration | Loader and schema closure complete | Strong integration milestone |
| Official SPA-Net integration canary | Complete: direct checks pass; CPU CLI attempts timed out | Implementation ready; GPU execution preflight remains |
| ParT-enhanced SPA-Net | Blocked pending an authentic, compatible frozen representation | Planned primary representation comparison |
| JP-JEPA-enhanced SPA-Net | Blocked pending authentic code, checkpoint, preprocessing, and permission | Optional; not deadline-critical |

## Local mechanism benchmark

Run identifier: `track_a_mechanism_benchmark_20260814_v1`.

This is a development-only comparison of a dense neural network with the
custom `CMSInspiredFiveJetPairwiseEventTransformer` on the local Delphes
population. It is not SPA-Net, not a CMS analysis, and does not use CMS
collision data.

### Completed work

- All 15 DNN fold/seed jobs completed successfully.
- The original bounded phase completed 2 of 15 event-transformer jobs:
  folds 0 and 1 for seed 20260811.
- A first continuation driver contained a mechanical bug: it read rows marked
  `COMPLETE` as well as `MISSING`. It began a duplicate fold-0 job and was
  stopped before that job wrote an output.
- The original fold-0 and fold-1 artifacts were verified untouched using file
  times, the previously recorded runtime fields, and the duplicate attempt's
  empty job log.
- The corrected `run_continuation_v2.sh` strips CRLF line endings, explicitly
  filters for `status == MISSING`, and passed a dry run with exactly 13 jobs,
  no duplicates, and fold 2 / seed 20260811 as its first job.

### Active-computation snapshot

The corrected continuation subsequently completed, with return code zero:

- fold 2, seed 20260811;
- fold 3, seed 20260811;
- fold 4, seed 20260811.

The continuation then completed fold 0 for seed 20260812 with return code zero.
Therefore 6 of 15 event-transformer jobs were complete at this snapshot. The
active job was fold 1, seed 20260812. The corrected driver and training process
were both alive.

No pooled prediction, AUC, rejection, ranking, or model-comparison result may
be reported until all 15 event-transformer jobs are valid. The current state
is successful computation in progress, not a finished benchmark.

## External Sophon study: completed preparation

The external study uses the public Yang-Li jet-free HH->4b simulation route.
The existing paper-facing E0/E1/E2 result remains the only completed model
performance result from this route.

The subsequent preparation established:

- a frozen file-disjoint development and `holdout_Q` split;
- a deterministic five-jet assignment/classification adapter;
- 1,008,376 pilot events: 421,921 signal, 470,455 QCD, and 116,000 ttbar;
- 17/17 pilot leakage and integrity checks passing;
- an event-level four-unique-jet matchable fraction of 47.5%-48.9%;
- a separate per-quark reconstruction/matching rate of 82.55%, with the
  denominators explicitly distinguished;
- a correctly named custom SPA-Net-inspired pairwise set transformer retained
  only as an engineering canary, never as the paper's SPA-Net baseline;
- a 10-model paper registry, comparability matrix, result-table schemas,
  figure specifications, and a machine-checkable result validator.

The paper registry keeps the external Sophon comparison separate from the
supporting local-Delphes study. Existing results from the two populations must
not be placed in one ranked table.

## Official SPA-Net integration

Run identifier: `phase4S_official_spanet_integration_canary_20260814_v1`.

### Environment and implementation

- Official repository: `Alexanders101/SPANet`.
- Pinned commit: `46c68051fbc6178b932b8eec562c792886769511`.
- Package version: 2.2.0.
- License: BSD-3-Clause.
- A separate Python 3.9 environment was created; the frozen E0/E1/E2 and
  custom-canary environments were not modified.
- The official declared dependencies were installed. Runtime-discovered
  dependencies `rich`, `tensorboard`, and `psutil` were also required by the
  official package and recorded.

### Completed official-loader closure

The eight already-authorized pilot NPZ files were converted to file-disjoint
official SPA-Net HDF5 train and validation files. The official
`JetReconstructionDataset` loader reported:

| Split | Rows | Signal rows | Assignment-valid rows |
|---|---:|---:|---:|
| train | 652,934 | 280,992 | 137,236 |
| validation | 355,442 | 140,929 | 68,906 |
| total | 1,008,376 | 421,921 | 206,142 |

All HDF5 input, target, classification, and official-loader row counts agreed.
The official assignment-valid mask exactly matched the source NPZ mask, event
labels matched exactly, train and validation source files were disjoint, and
the combined row count matched the prior leakage audit.

The two-Higgs event configuration was loaded through the official
`EventInfo` parser. It represented both within-Higgs daughter exchange and
the exchange of the two identical Higgs resonances.

### Canary closure

- Loader messages were redirected away from the payload and the official-loader
  verification was regenerated as clean standalone JSON.
- The loss-gated single-head definition, strict assignment-only definition,
  and two-head definition were frozen and distinguished by exact parameter
  counts.
- Direct official-API forward, loss, backward, and gradient checks completed
  for all three configurations in approximately 40--50 seconds each. All
  losses and gradients were finite.
- Both bounded one-epoch official CLI attempts ended at their explicit
  600-second CPU limits: the single-head attempt during backward and the
  two-head attempt during forward. Their DataLoader-worker `Terminated`
  tracebacks were timeout-shutdown effects, not model failures.
- The model registry now classifies the official single-head and two-head
  SPA-Net entries as `TRAINING_NEEDED`, not blocked by missing implementation.
- Phase-4Q, Phase-4R, and Phase-4S provenance and SHA-256 manifests were
  regenerated and verified.

Consequently, the official SPA-Net implementation and data interface are
validated for development training. The CPU CLI runs provide no performance
result and show that scale-out should use a separately validated GPU or batch
execution path rather than a longer interactive CPU run.

## Sealed resources

At this snapshot:

- `holdout_A` remains spent only for the completed E0/E1/E2 adjudication;
- `holdout_B` remains unopened;
- `holdout_Q` remains unopened;
- `inference_qcd_1` and `inference_qcd_2` remain unopened;
- no final signal-evaluation resource was opened;
- no CMS collision data was used.

## Next actions

1. Let the corrected local event-transformer continuation run to completion.
   Pool and evaluate only after all 15 event-transformer jobs are valid.
2. Preserve both official SPA-Net CLI timeout attempts as operational evidence;
   do not classify either as a model failure or silently overwrite its logs.
3. Validate a bounded GPU/batch execution path on the existing authorized
   pilot before any full-corpus materialization or development training.
4. Resolve the authentic frozen-ParT checkpoint, preprocessing, and feature
   compatibility gate for the planned embedding comparison.
5. Estimate full-corpus storage and wall time from measured pilot throughput,
   then freeze the scale-out contract before materializing additional files.
6. Keep every holdout and inference resource sealed until a separate
   pre-registered evaluation.

## Paper interpretation

The official-loader closure is a substantive reproducibility result: the
external pilot can be represented without row, label, mask, or assignment
drift in the official SPA-Net data interface. It is not yet evidence that
SPA-Net improves assignment or event classification. The paper must wait for
the completed common-protocol model comparison before making that claim.
