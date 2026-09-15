# Harvey-facing global BDT background-suppression brief

## MEASURED FACTS

All performance values below use the immutable five-fold nested outer-OOF TRAIN predictions. ROC quantities are unweighted event counts; physical yields use signed Run-2 evaluation weights.

### Global mass-plane-blind BDT

target epsilon_S | achieved epsilon_S | epsilon_B | rejection | physical S | physical B | QCD raw rows | QCD Neff
---:|---:|---:|---:|---:|---:|---:|---:
0.6 | 0.6 | 0.194606 | 5.14 | 436.2 | 8.564e+08 | 4948 | 1.23e+03
0.585957 | 0.585957 | 0.183893 | 5.44 | 425.1 | 7.867e+08 | 4596 | 1.14e+03
0.5 | 0.5 | 0.124211 | 8.05 | 356.7 | 5.147e+08 | 3029 | 749
0.4 | 0.4 | 0.0762672 | 13.1 | 283.1 | 2.364e+08 | 1625 | 393
0.25 | 0.25 | 0.034556 | 28.9 | 170 | 7.218e+07 | 505 | 113
0.1 | 0.1 | 0.00771414 | 130 | 58.95 | 1.605e+07 | 139 | 31.6

### Global mass-aware BDT

target epsilon_S | achieved epsilon_S | epsilon_B | rejection | physical S | physical B | QCD raw rows | QCD Neff
---:|---:|---:|---:|---:|---:|---:|---:
0.6 | 0.6 | 0.191614 | 5.22 | 436.3 | 8.508e+08 | 4994 | 1.19e+03
0.585957 | 0.585957 | 0.18096 | 5.53 | 425.3 | 7.733e+08 | 4583 | 1.09e+03
0.5 | 0.5 | 0.125248 | 7.98 | 357.5 | 4.638e+08 | 2836 | 671
0.4 | 0.4 | 0.0757219 | 13.2 | 281.4 | 2.296e+08 | 1624 | 336
0.25 | 0.25 | 0.0306391 | 32.6 | 173.6 | 6.128e+07 | 449 | 90.4
0.1 | 0.1 | 0.00600992 | 166 | 66.07 | 7.984e+06 | 47 | 14

Zero surviving QCD simulation rows, where encountered, must not be interpreted as zero physical QCD.

## CMS LITERATURE REFERENCE

The contextual HIG-24-010 resolved SR4b-conditioned guide points are approximately rejection 100 at epsilon_S~0.40 and 1000 at epsilon_S~0.10. They are not apples-to-apples with this broad global classifier and were not used for tuning.

## DIAGNOSIS

`CLASSIFIER_AND_QCD_LIMITING`

The current representation lacks continuous b-tag scores, the fifth b-tag-ranked jet, its four-vector, and five-jet alternative combinations. High-score physical interpretation is additionally limited wherever QCD Neff falls below 100 or only O(1-10) QCD rows survive.

## UNRESOLVED HYPOTHESES

A richer global representation may recover additional rejection, but this cannot be established from the current OOF products. Sparse weighted QCD tails may also obscure the true physical rejection. Broad-preselection differences remain entangled with the non-apples-to-apples CMS comparison.

## NEXT EXPERIMENT

Perform a bounded TRAIN-only re-extraction canary that materializes the five leading b-tag-ranked jet four-vectors, alternative pair kinematics, and any genuinely continuous tag discriminator available upstream. Preserve five-fold source-group nested CV and sealed validation/test. Do not submit a large extraction or new scientific training campaign until schema closure and QCD-tail support are demonstrated.
