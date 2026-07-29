# Legacy ttbar generation-provenance search

The frozen source registry contains one legacy local ttbar candidate campaign identified by

`ttbar_100k_shard000_hh4b_candidates.parquet`.

Its original generator cards, logs, process definition, and normalization denominator are
not part of the EOS production-bundle provenance archive.

This gate searches only approved local roots for exact source-identity anchors. It:

- stats filesystem entries;
- reads only small non-payload text files;
- excludes shell histories, credentials, caches, virtual environments, Git internals, and
  event payload formats;
- records exact filename and content references;
- classifies directly associated provenance candidates;
- never assigns a generic ttbar card to the legacy sample automatically.

The candidate Parquet is not opened. ROOT, Parquet, HepMC, LHE, and archive payloads are
not read.

Any located artifact remains unapproved until its source identity and content are manually
reviewed. If exact provenance cannot be recovered, the rigorous normalization decision is
to regenerate, replace, or exclude the legacy campaign rather than invent a denominator.
