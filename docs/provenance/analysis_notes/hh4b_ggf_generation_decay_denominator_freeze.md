# HH4b ggF generation, decay, and denominator freeze

PN-c4c established that both ggF campaigns use one exact production payload. Its process
card imports the HEFT model and generates

```text
p p > h h, (h > b b~), (h > b b~)
```

at 13 TeV through runtime beam overrides. There is no explicit s-channel resonance.

PN-c4d closes the remaining symbolic conventions without assigning a numerical cross
section:

- the generator kinematics are an LO HEFT ML-training approximation;
- both Higgs bosons are forced to decay to `bb`;
- a future inclusive `pp -> HH` reference cross section must therefore be multiplied by
  `BR(H -> bb)^2` exactly once;
- the 100 source shards contain 1000 unweighted events each;
- the full-campaign denominators are 20,000 and 80,000;
- train, validation, and evaluation shards must not each be renormalized independently.

The denominator convention is frozen, but numerical branching fractions, cross sections,
physical weights, and yields remain unauthorized. No event payload or candidate content is
opened.
