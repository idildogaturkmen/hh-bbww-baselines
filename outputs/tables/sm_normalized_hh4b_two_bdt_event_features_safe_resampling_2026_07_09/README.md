# Safe event-feature two-BDT resampling study

This study adds Poisson bootstrap resampling of selected test events to estimate MC-statistical uncertainty in the two-BDT working points.

Number of train/test seeds: 20
Poisson bootstraps per seed per working point: 200

Working points:
[(0.8, 0.5, 'loose_best_sqrtB'), (0.825, 0.525, 'nominal_balanced'), (0.875, 0.75, 'higher_purity'), (0.85, 0.75, 'intermediate')]

This is complementary to multiseed training stability. Multiseed variation probes training/split stability, while Poisson bootstrap resampling probes finite-MC statistical fluctuations in the selected regions.
