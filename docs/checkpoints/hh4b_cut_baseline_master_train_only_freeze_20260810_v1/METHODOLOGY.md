# Frozen HH→4b cut-baseline methodology

- The primary train-side generalization estimate is pooled five-fold nested outer-OOF performance.
- The median deployment cut evaluated on train is a fixed diagnostic, not the primary generalization estimate.
- The historical comparator is fixed at $R_{HH}(125,125)<34$ and was not optimized here.
- The 1,000-replica selection-stability bootstrap is diagnostic and cannot change the nominal cut.
- The 2,000-replica paired source-group metric bootstrap evaluates frozen selections and does not rerun selection optimization.
- Expected yields use $\sqrt{s}=13$ TeV and 138 fb$^{-1}$ equivalent (138000 pb$^{-1}$ internally).
- Results are Delphes simulation projections and are not an official CMS measurement.
- Validation may be evaluated once only with the frozen nominal cut after a separate authorization checkpoint.
- Test remains sealed for the later frozen cross-model comparison.
