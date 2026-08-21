# TABLE_MODEL_AUC_COMPARISON

**DEVELOPMENT ONLY -- not a final/governing result, not independent Stage-C inference, not physically normalized.**

Cohort: 400,000 events (signal=193,358, QCD=174,485, ttbar=32,157), identical across all four models.

| Model | Training population | Feature/input definition | Seed | Training budget | Checkpoint criterion | AUC all-bg | AUC QCD | AUC ttbar |
|---|---|---|---|---|---|---|---|---|
| BDT-K | 8,377,425 train | 52 kinematic features (jet 4-vectors / derived kinematic observables; no flavor-tag information). | 0 | 6000 boosting rounds (hard cap); early-stopping patience=50 rounds configured but NOT triggered -- terminated on the round cap, not the early-stop rule. | best_iteration by validation logloss (best_iteration=5998); model did not early-stop, effectively plateaued in its final ~15 rounds per HARVEY_SUMMARY. | 0.8366 | 0.8271 | 0.8882 |
| BDT-KF | 8,377,425 train | 82 features: K's 52 kinematic features plus 3 Sophon AK4 flavor-tag probability channels (probB/probC/probL) per jet slot. | 0 | 6000 boosting rounds (hard cap); early-stopping patience=50 rounds configured but NOT triggered -- terminated on the round cap, not the early-stop rule. | best_iteration by validation logloss (best_iteration=5999); model did not early-stop, effectively plateaued in its final ~15 rounds per HARVEY_SUMMARY. | 0.9682 | 0.9661 | 0.9793 |
| SPA-Net 2M (primary) | 2,000,000 train | SPA-Net official-schema per-jet kinematic input tensors (up to 10 AK4 jets), joint assignment+classification architecture (hidden_dim=32, transformer_dim=128, 8 encoder layers). | 0 | 50 epochs, batch_size=2048 | PRIMARY = maximum validation_average_jet_accuracy (best epoch 49, score 0.491227924823761). | 0.9693 | 0.9682 | 0.9754 |
| SPA-Net 10M (primary) | 10,000,000 train | Identical to SPANET_2M -- same SPA-Net official-schema per-jet kinematic input tensors and architecture, only training-population size differs. | 0 | 50 epochs, batch_size=2048 | PRIMARY = maximum validation_average_jet_accuracy (best epoch 47, score 0.4928354024887085). | 0.9693 | 0.9678 | 0.9775 |
