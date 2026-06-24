# SPA-Net H→bb First Run

The first full SPA-Net H→bb assignment run trained successfully on the GPU.

## Setup

- Task: H→bb jet assignment inside HH→bbWW events
- Input dataset: data/spanet_hbb/spanet_hbb_trainval.h5
- Test dataset: data/spanet_hbb/spanet_hbb_test.h5
- Train+validation events: 8004
- Test events: 1969
- Max jets: 20
- Features: log_pt, eta, sin_phi, cos_phi, log_mass, btag, btag_phys, response_corr, log_corrected_pt, log_corrected_mass
- Training: 50 epochs, GPU, first untuned SPA-Net config

## Test result

Official SPA-Net test output:

| Jet category | Event/H purity |
|---|---:|
| == 2 jets | 1.000 |
| == 3 jets | 0.742 |
| >= 4 jets | 0.529 |
| Full | 0.546 |

## Interpretation

The first SPA-Net setup works technically: the model trains, saves checkpoints, and evaluates on the held-out test set. However, the first-run performance is below the corrected BDT and corrected Pair-DNN baselines, which are around 0.698–0.700 H→bb truth-pair accuracy.

The performance drop is mostly in events with four or more jets, where the combinatorial assignment problem is harder. The BDT and Pair-DNN baselines used explicit pair-level features such as corrected m_bb and |m_bb - 125|, while this first SPA-Net configuration only used per-jet features. The next step is therefore to improve the SPA-Net input/features and/or configuration before making a final comparison.
