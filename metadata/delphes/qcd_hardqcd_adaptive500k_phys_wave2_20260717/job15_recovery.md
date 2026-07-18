# Wave 2 job 15 recovery

Original job:

`59823913.15`

Physics identity:

- bin: 0, pTHat 50-75 GeV
- shard: 9788
- seed: 1209788
- events: 10,000
- split: train

## Failure

Pythia generation and Delphes completed successfully. The event
summary was written. The HH4b reconstruction correctly produced a
verified empty candidate Parquet because no event contained four
selected b-tagged jets.

The Python reconstruction process then received SIGABRT during
interpreter/library teardown. The wrapper therefore stopped at the
reconstruction stage before bundling and EOS stage-out.

No final EOS bundle exists.

## Recovery policy

One retry is authorized using exactly the same campaign, pTHat bin,
shard ID, seed, event count, and split.

The retry uses an operational-only empty-output exit fix. It does not
change event generation, detector simulation, selection, candidate
reconstruction, weighting, or schema.

The retry replaces the missing shard and contributes no additional
statistical events.
