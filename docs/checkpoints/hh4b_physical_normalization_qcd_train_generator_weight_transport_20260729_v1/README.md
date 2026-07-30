# HH4b hard-QCD train generator-weight transport freeze

This checkpoint freezes the completed PN-c4i train-only transport of nominal Pythia generator weights into immutable candidate-level sidecars for variable-weight hard-QCD shards and a compact constant-weight registry for uniform-positive shards.

- Source commit: `60fa40399f6205bd2e5d8cda3d243e287d65a5e8`
- Source scale-out: `/uscms/homes/i/iturkmen/pn_c4i_qcd_train_generator_weight_transport_scaleout_20260729_v1`
- Train shards: 186 (127 variable, 59 uniform)
- Variable candidate sidecar rows: 3
- Uniform candidate rows covered by constants: 53
- Zero-candidate variable shards: 124
- Validation payloads opened: 0
- Final-evaluation payloads opened: 0
- Physical luminosity weights or yields calculated: 0

The sidecar quantity `generator_cross_section_contribution_pb` is the already-frozen direct hard-QCD stitched projection coefficient multiplied by the nominal generator weight. Direct hard-QCD remains secondary closure/projection only; the intended primary multijet treatment remains the lower-b-tag control-region transfer.

## Next gate

Build the process reference-cross-section and normalization provenance registry for the HH signals and non-QCD backgrounds. No luminosity-normalized candidate weight is authorized until generator definitions, reference cross sections, decay/filter conventions, signed generator-weight denominators, and overlap policies are frozen.
