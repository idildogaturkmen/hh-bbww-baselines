# HH4b ttbar seven-member reconstruction-only recovery

This checkpoint freezes a local exact-image reconstruction-only recovery
for the seven retained Delphes ROOT products from full-chain cluster
`59896839` on `lpcschedd5.fnal.gov`.

All seven ROOT files are immutable inputs with exactly 10,000 Delphes
entries each. MadGraph, Pythia, and Delphes will not be rerun.

The recovery uses the tested Python 3.9 package archive and corrected
payload inside the RHEL9 image. It explicitly binds CVMFS, starts from a
clean container environment, sources LCG 106 to match the failed worker
context, and then removes `PYTHONHOME` and replaces `PYTHONPATH` before
calling `/usr/bin/python3`.

The shard001 canary produced 45 canonical-72 rows in two byte-identical
runs. Its candidate SHA-256 is frozen as an execution-time regression
gate.

Seven sequential local runs are described. None has been executed.
No scheduler action occurred, no sealed test member was opened, and
physical normalization remains out of scope.
