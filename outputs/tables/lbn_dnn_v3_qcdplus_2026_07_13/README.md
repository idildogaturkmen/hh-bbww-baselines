# LBN-DNN v3 qcdplus baseline

This directory contains lightweight LBN-style DNN results for the HH→4b Delphes analysis.

Modes:
- `lbn_p4_only`: four candidate jet four-vectors only.
- `lbn_p4_plus_topology`: four-vectors plus topology-only auxiliary features.

The LBN layer learns non-negative combinations of the four candidate jet four-vectors and computes Lorentz features before a dense classifier.

This is a physics-structured neural-network baseline between the plain DNN and SPA-Net.
