# HH4b controlled standard LHE-weight canary

The metadata audit found unweighted-event evidence for all 47 standard campaigns and no
negative-weight text evidence, but that is not sufficient to authorize `Ngen` as the
physical-normalization denominator.

PN-c3b performs the first explicitly authorized generator-payload access. It selects six
standard process strata:

- `vbf_hh4b`;
- `ttbar_inclusive`;
- `qcd_bbbb_general`;
- `tchannel_top`;
- `tth_hbb`;
- `wwz_zbb`.

For each process it chooses the smallest frozen representative bundle containing exactly
one `.lhe.gz` source member. The complete bundle is transferred to temporary non-workspace
scratch, checked against its frozen size and SHA-256, and removed after the audit.

Only the LHE member is opened. The parser records the LHE `IDWTUP` convention and all
nominal `XWGTUP` values. It classifies each representative as uniform positive, variable
positive, signed, zero-containing, or invalid.

Uniform-positive behavior is canary evidence only. This gate does not generalize the result
to all campaigns and does not authorize `Ngen`, a signed sum of weights, a cross section,
or a physical yield.
