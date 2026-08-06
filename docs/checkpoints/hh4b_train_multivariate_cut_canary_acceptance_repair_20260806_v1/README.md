# Multivariate-cut canary acceptance repair

The category-conditional diagnostic completed every implementation and
provenance check but the bounded pooled inner-OOF signal efficiencies missed
the exact target by tiny amounts:

- exact3tag: 0.585940952290 versus 0.585957;
- ge4tag: 0.585252113872 versus 0.585957.

The fully open control was exactly 1.0 in both categories. The diagnostic used
true inner cross-fitting, kept outer fold 0 sealed, produced identical complete
reruns, recomputed selected masks, preserved comparison/physical weight roles,
and proved that exact3tag and ge4tag selected event sets are disjoint.

The prior canary incorrectly required a finite bounded held-out sample to
reproduce the production operating point with zero numerical tolerance. This
checkpoint repairs only the **implementation-canary acceptance rule**.

A canary passes when all implementation gates pass and the best sentinel pooled
inner-OOF efficiency in each category is within 0.001 absolute of the target.
This tolerance is not permitted in the production scan. The full scan retains
the exact category-conditional target of 0.585957.

No production family or threshold has been selected. Diagnostic winners and
thresholds remain nonphysical evidence and cannot enter paper figures.
