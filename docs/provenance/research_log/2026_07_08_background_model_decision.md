# Background model decision, 2026-07-08

Based on the weighted region-yield table, QCD bbbb remains the dominant HH4b background in all tested candidate regions, including R_HH < 30, R_HH < 55, high-mHH regions, and combined R_HH + mHH selections.

Current nominal background model:
- QCD bbbb HT-sliced 10k/slice
- inclusive ttbar 100k
- Zbbbb 100k

Cross-checks and diagnostics:
- inclusive QCD bbbb 100k is retained as a closure/reference sample for the HT-sliced QCD model.
- ttbb 50k is retained as a diagnostic top+heavy-flavor sample.

The ttbb sample has much higher HH4b-candidate efficiency than inclusive ttbar, confirming that top+extra-heavy-flavor events are more HH4b-like. However, ttbb is not added directly to the inclusive ttbar normalization because inclusive ttbar can already contain extra heavy flavor from showering. A rigorous combination would require a truth-level split/veto or matched top+heavy-flavor sample definition.

Next analysis step:
- Build combined signal-vs-background classifiers using weighted backgrounds.
- Compare cut-based HH mass-plane selections against BDT/DNN-style baselines.
- Use QCD HT-sliced as the primary QCD model and inclusive QCD as a closure cross-check.
