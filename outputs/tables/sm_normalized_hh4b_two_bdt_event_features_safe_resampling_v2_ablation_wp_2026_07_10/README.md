# Safe event-feature two-BDT resampling study

This study adds Poisson bootstrap resampling of selected test events to estimate MC-statistical uncertainty in the two-BDT working points.

Number of train/test seeds: 20
Poisson bootstraps per seed per working point: 200

Working points:
[(0.775, 0.5, 'previous_v2_loose_mid'), (0.8, 0.5, 'previous_v2_loose_best'), (0.825, 0.5, 'ablation_best_median'), (0.85, 0.5, 'ablation_tighter_qcd_same_top'), (0.825, 0.475, 'ablation_qcd825_looser_top'), (0.85, 0.475, 'qcd850_looser_top'), (0.825, 0.525, 'previous_nominal_balanced')]

This is complementary to multiseed training stability. Multiseed variation probes training/split stability, while Poisson bootstrap resampling probes finite-MC statistical fluctuations in the selected regions.
