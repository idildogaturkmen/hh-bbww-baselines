# DNN-v3 qcdplus mass-aware and topology-only baselines

This trains ordinary dense neural networks using the same samples and train/test split as BDT-v3 qcdplus.

Two feature sets are evaluated:
- mass_aware: same features as mass-aware BDT-v3 qcdplus
- topology_only: direct mass variables removed

Outputs include AUC summaries, rectangle scans, category yields, background composition, scored test events, and saved PyTorch models.
