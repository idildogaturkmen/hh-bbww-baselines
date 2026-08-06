# Multivariate cut implementation clarification

The base 54-family contract is scientifically suitable as an interpretable,
CMS-inspired cut benchmark. This checkpoint does not change its candidate
families or primary operating point.

It clarifies the required true nested procedure:

1. Optimize each family separately inside each inner-training split.
2. Apply that split's thresholds once to its inner-held-out fold.
3. Choose the family from pooled cross-fitted inner-held-out predictions.
4. Refit only the chosen family on all outer-development folds.
5. Apply it once to the untouched outer fold.

Optimizing one threshold vector and evaluating it on the same pooled
outer-development rows is explicitly forbidden.

The bounded canary uses one outer fold and six sentinel families only to validate
the implementation. Its winner cannot alter the production contract and cannot
be used as a paper physics result.

The cut study is CMS-inspired rather than CMS-approved. Final plots must state
`Delphes simulation` and `train-source-group OOF`. Final nuisance-aware
sensitivity is deferred to a later likelihood contract.
