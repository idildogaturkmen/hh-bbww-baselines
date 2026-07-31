# Track B IHEP pileup external-blocker checkpoint

Date: 2026-07-31

## Purpose

Record the final local readiness state for the controlled
same-generator-event comparison between:

1. the Track A raw-EFlow reconstruction; and
2. the released IHEP PUPPI particle representation.

## Signal source

The preserved Track A signal source is:

`/uscms_data/d3/iturkmen/hh4b_delphes/hepmc/ggf_hh4b_ak4ak8_10k_pythia8.hepmc`

The file:

- is readable;
- contains 10,000 generated HH events;
- has valid HepMC3 ASCII structure;
- contains the start marker, event records, and end marker.

## Pinned released configuration

Public repository:

`pku-hep-group/jetfree-hh4b`

Pinned commit:

`e8387f7e49a7f27c65b2f4a1f75cacf01cd45392`

Pinned Delphes card:

`delphes/cards/delphes_card_CMS_JetClassII_lite.tcl`

The card actively constructs:

`RunPUPPI/PuppiParticles`

and writes it as:

`ParticleFlowCandidate`

The card requires:

- `PileUpFile MinBias_100k.pileup`
- `MeanPileUp 50`
- `ZVertexSpread 0.25`
- `TVertexSpread 800E-12`

## Delphes runtime

The local DelphesHepMC3 executable was validated under:

- LCG 106
- GCC 13.1.0
- ROOT 6.32.02
- Delphes 3.5.1pre09-compatible runtime

The following dependencies resolve correctly:

- `libstdc++.so.6`
- `libtbb.so.12`
- `libvdt.so`
- ROOT libraries

Available C++ symbol versions include:

- `GLIBCXX_3.4.30`
- `GLIBCXX_3.4.31`

Result:

`DELPHES_LCG_RUNTIME_READY`

## Local pileup conversion capability

The following converters are available and have resolved runtimes:

- `hepmc2pileup`
- `root2pileup`
- `stdhep2pileup`

The standard Delphes converter card is available:

`cards/converter_card.tcl`

Therefore:

- an authoritative minimum-bias HepMC sample can be converted locally;
- a suitable authoritative ROOT sample can be converted locally;
- the final authoritative `.pileup` file can be used directly.

## Missing assets

No local or CVMFS copy was found for:

`MinBias_100k.pileup`

No matching minimum-bias HepMC or ROOT source was found.

The following authoritative generation inputs were also not found:

- `generatePileUp.cmnd`
- minimum-bias Pythia process settings
- event-generation seed
- exact event-generation software contract
- working local `DelphesPythia8` generation path for this contract

## Result

`EXACT_IHEP_PILEUP_STILL_EXTERNAL`

The exact matched-IHEP-card signal canary is externally blocked,
not locally or technically blocked.

## Resume conditions

The matched-card signal canary may resume when any one of the
following is provided:

1. the exact `MinBias_100k.pileup` file and checksum;
2. the authoritative minimum-bias HepMC source and checksum;
3. a complete authoritative reproduction contract including:
   - generator and version;
   - process settings;
   - collision energy;
   - number of events;
   - random seed;
   - conversion procedure;
   - expected output checksum or validation quantities.

## Scientific interpretation

The existing Track A released-model canary remains a
raw-EFlow-to-PUPPI-model domain-transfer stress test.

Its high-mass signal localization must not yet be interpreted as
an intrinsic failure of the released model on Track A because the
official PUPPI particle representation has not been reproduced.

An arbitrary generic pileup file must not be substituted for the
exact controlled comparison.
