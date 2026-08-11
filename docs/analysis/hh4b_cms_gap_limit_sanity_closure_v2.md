# HH4b CMS-gap independent limit sanity closure v2

For the fixed train-only `R_HH(125,125)<34` count, `S=184.72444`, `B=48,653,342.978`, and `S/B≪1`. The independent small-signal approximation is

`Z ≈ S/sqrt(B) = 0.02648305`, hence median 95% CLs `mu95 ≈ 1.96/Z = 74.0096`.

The frozen likelihood implementation gives 74.0152; the relative difference is 0.0075%. The previously frozen exact Asimov significance is 0.02648303. This closes without modifying the inference code.

For independent channels, likelihood information adds: `Z²≈Σ S_i²/B_i`. Exact3tag gives 0.0199215, ge4tag gives 0.0394927, and the quadrature result is `Z=0.0442328`, implying `mu95≈44.3110`. The implemented value is 44.3276 (0.0374% higher), as expected from retaining finite `S/B` terms. There is no discrepancy.

`CMS_GAP_VALIDATION_PAYLOADS_OPENED=0`; `CMS_GAP_TEST_PAYLOADS_OPENED=0`.
