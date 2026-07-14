# HH4b SPA-Net leading-8 dataset

Dataset tag: `qcdplus_btag4_v1`

Inputs:
- X_jets: shape (events, 8, 5), features = pt, eta, phi, mass, btag
- jet_mask: real/padded jet mask
- y: signal/background label
- assignment: shape (events, 2, 2), jet indices for H1/H2 daughters, -1 if missing
- assignment_mask: 1 if all four H→bb truth daughters are matched to selected jets
- weight_pb: event physics weight = xsec / generated events

Selection:
- jets with pt > 30.0 GeV and |eta| < 2.5
- at least 4 selected jets
- at least 4 selected b-tagged jets
- keep up to leading 8 selected jets by pt
- truth matching ΔR < 0.4

Notes:
- Jet flavor is not included as an input feature.
- Background events have assignment labels set to -1 and assignment_mask = 0.
- Signal events without complete truth assignment are kept for classification but masked for assignment loss.
