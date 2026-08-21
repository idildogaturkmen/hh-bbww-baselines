# TABLE_SPANET_2M_10M_SCALING

**DEVELOPMENT ONLY -- not a final/governing result, not independent Stage-C inference, not physically normalized.**

Same architecture, same seed=0, same batch_size, same 50-epoch budget, same frozen 400k validation population, same checkpoint criterion (max validation_average_jet_accuracy). Only training-population size differs.

| Metric | 2M seed-0 | 10M seed-0 |
|---|---|---|
| Primary best epoch | 49 | 47 |
| validation_average_jet_accuracy (primary) | 0.491228 | 0.492835 |
| Secondary best val/loss/total_loss | 0.510649 | 0.491777 |
| Total runtime (s) | 8577.31 | 31727.16 |
| Events/s (train-only) | 15499.29 | 16848.25 |
| Peak GPU reserved (bytes) | 1,614,807,040 | 1,614,807,040 |
| Training population | 2,000,000 | 10,000,000 |
| AUC all-background (classification, separate eval) | 0.9693 | 0.9693 |
| AUC QCD | 0.9682 | 0.9678 |
| AUC ttbar | 0.9754 | 0.9775 |

2M->10M does NOT produce a clear event-classification AUC improvement (all-background/QCD AUC essentially flat to slightly down; ttbar AUC up ~0.002) despite 10M's slightly higher validation_average_jet_accuracy (a JET-ASSIGNMENT metric, not a classification metric).
