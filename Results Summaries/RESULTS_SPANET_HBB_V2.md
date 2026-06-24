# SPA-Net H→bb v2 Run

## Setup

- Task: H→bb jet assignment inside HH→bbWW events
- Event file: explicit H→bb event syntax with b1 and b2 assigned to Jets
- Train+validation file: data/spanet_hbb/spanet_hbb_trainval.h5
- Test file: data/spanet_hbb/spanet_hbb_test.h5
- Run directory: outputs/spanet_runs/hbb_run_v2_gpu/version_0
- Training: 100 epochs, GPU, larger SPA-Net configuration

## Test result

SPA-Net loaded checkpoint:

epoch=35-step=1656-validation_average_jet_accuracy=0.536.ckpt

Official SPA-Net test output:

| Jet category | Event/H purity |
|---|---:|
| == 2 jets | 1.000 |
| == 3 jets | 0.767 |
| >= 4 jets | 0.541 |
| Full | 0.558 |

## Interpretation

The v2 SPA-Net run improved only slightly over the first SPA-Net run, from about 0.546 to 0.558 full H purity. This remains below the corrected pair-level BDT and Pair-DNN baselines, which are around 0.698–0.700. The main weakness remains the high-jet-multiplicity category, where most test events lie.

This suggests that simply increasing the SPA-Net model size is not enough. The next step should be to improve or diagnose the input representation rather than continuing blind architecture tuning.
