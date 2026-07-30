# HH4b QCD train generator-weight transport freeze

PN-c4i completed checksum-verified, train-only nominal generator-weight transport for all 186 physical hard-QCD train shards. The transport population contains 127 variable-positive shards and 59 uniform-positive shards. Variable shards use the unique Delphes `Event/Event.Weight` branch with event-count, sum-of-weights, squared-sum, minimum, maximum, and train-candidate event-index closure. Uniform shards are represented by an exact constant generator weight of one and require no redundant payload read.

PN-c4j freezes the consolidated train candidate sidecars, uniform-weight registry, transport registry, ordered target inventory, fragment checksum manifest, and a compact 186-row shard closure registry. This checkpointing step opens no event or candidate payload and calculates no luminosity-normalized weight or yield.

The direct stitched hard-QCD contribution remains secondary closure/projection only. It must not be summed with overlapping `qcd_bbbb_general` or `qcd_bbbb_iht400to600` samples. The intended primary CMS-style multijet treatment remains the lower-b-tag control-region transfer.

The next gate is to build the process reference-cross-section and normalization provenance registry for the HH signals and non-QCD backgrounds. Physical candidate weights remain unauthorized until the exact process definitions, reviewed cross sections, branching-fraction and filter conventions, generator-weight denominators, and overlap-removal policies are frozen.
