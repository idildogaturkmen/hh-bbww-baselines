# HH4b remaining standard-group weight-convention audit

PN-c3d froze signed ROOT-weight denominators for the three train triboson campaigns. The
remaining standard model consists of 20 card-equivalent process groups and 41
process-campaign normalization records spanning 15 unique production campaign labels.
Several production campaign labels intentionally contain more than one physics process.

PN-c3e selects the smallest resolved train ROOT bundle for each remaining process group,
verifies the frozen bundle size and SHA-256, extracts only the mapped ROOT source, and reads
only the nominal event-record weight branch.

A group may use the generated-event count as an effective denominator only when its
representative ROOT weights are finite, nonzero, positive, and constant. In that case the
common constant cancels:

\[
\frac{c}{N_{\mathrm{gen}}c} = \frac{1}{N_{\mathrm{gen}}}.
\]

The effective numerator is recorded as one and each campaign retains its own manifest
generated-event denominator. Cross-section sharing is not authorized.

Any signed or variable group remains fail-closed and must receive complete source-event
weight sidecars before normalization.

No candidate, validation-candidate, or evaluation-candidate content is opened, and no
external cross section, physical weight, yield, model, or threshold is calculated.
