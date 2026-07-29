# HH4b controlled standard LHE-weight canary

This checkpoint opens only six explicitly selected `source/unweighted_events.lhe.gz` members, one from each physics stratum: VBF HH signal, inclusive ttbar, standard QCD bbbb, t-channel single top, ttH(H->bb), and WWZ->Zbb.

Each containing EOS bundle is verified against its frozen byte size and SHA-256 before the LHE member is streamed. ROOT, Parquet, candidate, validation-candidate, and evaluation-candidate content is not opened.

The canary records IDWTUP and every nominal XWGTUP value in each selected LHE. Uniform-positive behavior supports only a candidate Ngen convention. It does not authorize generalization to other campaigns or authorize a normalization denominator.
