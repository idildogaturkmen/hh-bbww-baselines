# HH4b denominator-evidence and source-recovery audit

This checkpoint inventories all denominator, event-count, weight-strategy, negative-weight, unweighted-event, and QCD sampling evidence already present in the 547 checksum-frozen provenance files. It also searches three approved LPC text-file roots for exact references to the six nonstandard ggF HH and importance-sampled hard-QCD campaign names.

The search excludes ROOT, Parquet, HepMC, LHE, compressed files, archives, shell histories, credentials, virtual environments, caches, and other binary files. No event payload is opened.

All evidence remains candidate evidence. Run-card `event_norm` and representative `nevents` are not sufficient by themselves to authorize a campaign denominator. The next gate must either recover a trusted aggregate sum of nominal generator weights or perform a controlled generator-level weight audit proving a uniform positive nominal weight convention.
