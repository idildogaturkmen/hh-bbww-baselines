# Full 5.2M workflow-role contract

This checkpoint assigns every one of the 630 source members and all
5,200,000 generated events to an explicit train, validation, or test
workflow role.

## Generated-event universe

- Background events: 5,000,000
- Signal events: 200,000
- Total events: 5,200,000
- Source members: 630
- Split mechanism: source-grouped and disjoint

All generated events participate in source, split, acceptance, and
normalization accounting. Only events that pass the frozen reconstruction
and candidate requirements become ML rows.

## Train populations

The primary physical train projection contains 31,225 rows, including
576 transferred hard-QCD template rows.

The overlapping `qcd_bbbb_general` and
`qcd_bbbb_iht400to600` train samples contribute 22,457 auxiliary
classification candidates.

The planned full train fit population is therefore:

    31,225 + 22,457 = 53,682 rows

The auxiliary rows have no physical-yield or likelihood role.

## Validation and test

Primary validation and test remain sealed.

Physically authorized sources may enter the predeclared primary
validation and test evaluations at their respective gates.

Overlapping qcd-bbbb validation and test sources remain excluded from
primary physical metrics. They may be reported only as separately labeled
post-lock QCD stress tests and may not affect model selection, thresholds,
categories, yields, significance, or likelihood construction.

## Outputs

- `source_workflow_roles.tsv`: role of every source member;
- `split_role_summary.tsv`: split and workflow-role totals;
- `process_role_summary.tsv`: process-resolved totals;
- `train_projection_audit.tsv`: current and planned train tables;
- `summary.json`: machine-readable contract.
