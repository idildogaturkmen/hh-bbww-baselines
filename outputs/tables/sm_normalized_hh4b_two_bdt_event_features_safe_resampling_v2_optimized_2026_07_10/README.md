# Safe event-feature two-BDT resampling study

This study adds Poisson bootstrap resampling of selected test events to estimate MC-statistical uncertainty in the two-BDT working points.

Number of train/test seeds: 20
Poisson bootstraps per seed per working point: 200

Working points:
[(0.7, 0.5, 'v2_single_seed_best'), (0.725, 0.5, 'v2_single_seed_next'), (0.775, 0.5, 'v2_loose_mid'), (0.8, 0.5, 'v2_loose_best_sqrtB'), (0.825, 0.525, 'v2_nominal_balanced'), (0.825, 0.55, 'v2_multiseed_median_best'), (0.85, 0.55, 'v2_tighter_qcd'), (0.875, 0.75, 'v2_higher_purity')]

This is complementary to multiseed training stability. Multiseed variation probes training/split stability, while Poisson bootstrap resampling probes finite-MC statistical fluctuations in the selected regions.
