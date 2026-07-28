# HH4b ttbar seven-member reconstruction-only completion

This checkpoint validates the seven canonical-72 recovery products
created from retained Delphes ROOT files from full-chain cluster
`59896839` on `lpcschedd5.fnal.gov`.

The seven ROOT inputs contain exactly 70,000 Delphes events. Their
reconstruction produces 291 canonical candidate rows across seven
members.

Every candidate file has the exact canonical 72-column names, order,
and Arrow types. All numeric values are finite. There are no duplicate
`(sample, event)` keys within any member or across the combined
seven-member population. Candidate, receipt, final-status, and member
checksum identities are recorded in the accompanying tables.

MadGraph, Pythia, and Delphes were not rerun. No scheduler action
occurred, no sealed test member was opened, and physical normalization
remains out of scope.

The next gate is to commit this completion checkpoint and rerun the
27-member ttbar canonical-readiness audit.
