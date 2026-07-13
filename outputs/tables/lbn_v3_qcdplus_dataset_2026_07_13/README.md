# LBN-v3 qcdplus dataset

This dataset is built from the same BDT-v3 qcdplus candidate parquets used for the BDT and DNN baselines.

Main array:
- `X_p4`: shape `(N, 4, 4)`, containing the four selected candidate jets in `[E, px, py, pz]` format.

Auxiliary arrays:
- `X_aux_topology`: topology-only scalar features.
- `X_aux_mass_aware`: mass-aware scalar features.

Labels and masks:
- `y`: signal label.
- `is_qcd`: QCD-background mask.
- `is_top`: top-background mask.
- `train_mask` and `test_mask`: deterministic split using the same random seed and test fraction as the BDT/DNN studies.

Interpretation:
This dataset is intended for an LBN-DNN baseline. Since LBN uses four-vectors, it should be treated as physics-structured and mass-aware by construction.
