# Five-source exactly-3b reconstruction canary

This checkpoint records the bounded five-source train canary authorized by
PN-c7n v5.

The canary spans:

- a signal bundle with legacy Adler-32 plus exact byte-size evidence;
- an ordinary signed-weight background bundle;
- an ordinary uniform-weight background bundle;
- a variable-weight hard-QCD bundle;
- an authoritative local legacy ttbar ROOT.

For every source, this checkpoint proves:

- the live source checksum and byte size;
- unique resolution of the frozen ROOT archive member, allowing only audited path aliases such as leading directories;
- nested reconstruction-archive extraction for the frozen ggF layout;
- the exact frozen three-b output schema;
- exactly three selected tagged jets plus the highest-pt selected untagged
  promoted jet;
- unique in-range ROOT event identities;
- zero overlap with the corresponding four-b candidate events;
- exact generator-weight transport for every selected three-b event;
- full-member signed sidecar closure against every ROOT `Event.Weight`, with the existing four-b events verified as a subset of that 10,000-event map;
- EOS output size and Adler-32 closure;
- complete removal of temporary archives, extracted ROOTs, candidate copies,
  and local output files.

The five three-b Parquets and five generator-weight sidecars are stored under
`/store/user/iturkmen/hh4b_delphes/run2_13tev/frozen_v2/threeb_control/train_canary_20260731_v5`.

No Run-2 physical weights or selected yields are calculated here. Validation
and test remain sealed. A successful canary authorizes full train three-b
reconstruction only.
