# SPA-Net H→bb Assignment Runs

## Task

H→bb jet assignment inside HH→bbWW events using the SPA-Net event-assignment format.

## Dataset

- Training/validation file: data/spanet_hbb/spanet_hbb_trainval.h5
- Test file: data/spanet_hbb/spanet_hbb_test.h5
- Test events: 1969
- Max jets per event: 20
- Target: unordered H→bb pair, stored as TARGETS/h/b1 and TARGETS/h/b2

## Event file

The final SPA-Net event file uses explicit target-to-input syntax:

- b1: Jets
- b2: Jets

and includes the b1/b2 symmetry permutation.

## First SPA-Net run

Run directory:

outputs/spanet_runs/hbb_first_run_gpu/version_0

Best checkpoint loaded by spanet.test:

epoch=47-step=1104-validation_average_jet_accuracy=0.522.ckpt

Held-out test result:

| Jet category | Event/H purity |
|---|---:|
| == 2 jets | 1.000 |
| == 3 jets | 0.742 |
| >= 4 jets | 0.529 |
| Full | 0.546 |

## SPA-Net v2 run

Run directory:

outputs/spanet_runs/hbb_run_v2_gpu/version_0

Best checkpoint loaded by spanet.test:

epoch=35-step=1656-validation_average_jet_accuracy=0.536.ckpt

Held-out test result:

| Jet category | Event/H purity |
|---|---:|
| == 2 jets | 1.000 |
| == 3 jets | 0.767 |
| >= 4 jets | 0.541 |
| Full | 0.558 |

## Same-test-file top-2 b-tag sanity baseline

Using the SPA-Net test HDF5 file directly:

- Events: 1969
- Top-2 b-tag pair accuracy: 0.4926

## Interpretation

SPA-Net is training and evaluating correctly. The explicit event-file syntax is valid, and SPA-Net v2 improves over a simple top-2 b-tag baseline on the same test file. However, SPA-Net v2 remains substantially below the corrected pair-level BDT and Pair-DNN baselines, which were around 0.698–0.700 truth-pair accuracy.

The main weakness remains the >=4 jet category, which contains most test events. Increasing the SPA-Net model size from the first run to v2 improved full H purity only from 0.546 to 0.558, suggesting that the current bottleneck is likely the input representation or task formulation rather than only model capacity.
