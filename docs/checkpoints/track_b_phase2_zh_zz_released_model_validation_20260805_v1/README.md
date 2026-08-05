# Track B Phase 2: released-model validation for ZH and ZZ

**Checkpoint date:** 2026-08-05 UTC
**Branch:** `delphes-hh4b-production`
**Audited implementation head before this documentation commit:** `1d552cd51b8285f7df277ab77fd7e966f25ce3db`

## Scope

This checkpoint records the process-isolated SophonHH released-model validation completed for the ZH and ZZ control samples. The scientific products remain node-local under `/tmp`; no ROOT, NPZ, NPY, ONNX, or other binary inference products are committed here.

The validated released-model interface is:

- `pf_features`: `[N, 19, 256]`
- `pf_vectors`: `[N, 4, 256]`
- `pf_mask`: `[N, 1, 256]`
- output: `[N, 138]`
- signal classes: `0` through `135`
- QCD class: `136`
- ttbar class: `137`

Each model was run in a fresh model-only Python process with one CPU ONNX Runtime session. Preprocessing exited before model loading. Tensor batches were loaded sequentially to prevent the memory-overlap failure observed in the earlier all-in-one execution.

## Frozen implementation

| Item | Frozen identifier |
|---|---|
| execution backend SHA-256 | `91df61ffb13cab62fbd3755ed1dd55db535ed5f8cfa960322f1d0096a4067bd8` |
| ROOT contract Git blob | `6a434f477b7f06cd269c6c54d76c7ee5cf834b62` |
| score contract Git blob | `a4a67b87b43fd913cff86b625722d4ebaf14b928` |
| model 0 SHA-256 | `7b7de7ba1bfbbf00117b251661320afabe166c74c8687e3673d97590a3745786` |
| model 1 SHA-256 | `5514875d359fd4d6516e3367eb3264e6b4fea21f82755fc45851d06f788d429f` |
| model 2 SHA-256 | `2d02df9702bfa13f8c20b467087121abab2556fa654e12084b73cee968f3566e` |

## ZH validation

### Deterministic 1,000-event product

| Quantity | Value |
|---|---:|
| mean signal probability | `0.6663492197556656` |
| mean QCD probability | `0.2698487119757853` |
| mean ttbar probability | `0.06380206758255048` |
| events with `p_signal > 0.997` | `187 / 1000` |
| three-model top-1 agreement | `0.714` |
| process composition | `-1: 53`, `16: 947` |
| events truncated at 256 particles | `3` |
| ensemble score SHA-256 | `b75e6d2fc697cc66d0c74b1d567d3ceb17dc35982d78de531dad090bb61df336` |
| Events ROOT SHA-256 | `b5cce7ce283fae2ccf3903c6f87f9672433c2b3d20fde1d80c3b220c49f2aa11` |
| ensemble receipt SHA-256 | `ad15412bd0b43fc8d36006d753873230515b4b3d9d660db0782d76f7cff79a9f` |

### Reconstructed 64-event control

| Quantity | Value |
|---|---:|
| mean signal probability | `0.7492179485104496` |
| mean QCD probability | `0.15964379027815845` |
| mean ttbar probability | `0.09113825794225949` |
| events with `p_signal > 0.997` | `15 / 64` |
| three-model top-1 agreement | `0.703125` |
| selected-entry map SHA-256 | `3323201d4dbf171866948af56f66e9af0aa73da62e7e9bf39f8efef177130198` |
| ensemble score SHA-256 | `e425cc6c4b144dae1bf0fdc61aa17a26fcd8d74e4e01c2e5a846c91be5962fd0` |
| Events ROOT SHA-256 | `50a9c6d16124b3105f0e4bc9f43153c9b650f912ef61760f79d4d323d929bbb3` |
| comparison receipt SHA-256 | `1c3823ac4fc1cf874314824492dd75e1a5d5b35dafee12fe32bb1d2daf374f39` |

The frozen 64-event ZH summary was reproduced within the required tolerance. The 64-event and 1,000-event aggregate differences are deterministic subset-size and composition effects, not a released-model reproduction failure.

## ZZ validation

### Source and deterministic 1,000-event map

| Quantity | Value |
|---|---:|
| source SHA-256 | `c312e7adbc2652e91bd7556325565f2b638da8bcde69ef304a94501603d42d8f` |
| tree entries | `200000` |
| eligible entries | `8674` |
| selected-entry map SHA-256 | `4e829afc849a11c96826f034d5bb5840241cc3a4d0cc078d08c80afa3d8599c9` |
| first / last selected entry | `3 / 199990` |
| process composition | `12: 1000` |
| events above 256 particles | `4` |

### Deterministic 1,000-event product

| Quantity | Value |
|---|---:|
| mean signal probability | `0.445673965550993` |
| mean QCD probability | `0.4464766624782877` |
| mean ttbar probability | `0.10784936993610404` |
| events with `p_signal > 0.997` | `67 / 1000` |
| three-model top-1 agreement | `0.835` |
| process composition | `12: 1000` |
| events truncated at 256 particles | `4` |
| ensemble score SHA-256 | `90f9790715a3c81aeedf10d6d187f227ef6e75cb4e30f607a7475d077c16863c` |
| Events ROOT SHA-256 | `4e6abf76da4bf80b68b850da6a32c6f09a42e665bf2cf76301c307996ea9c708` |
| ensemble receipt SHA-256 | `1604769de2e6abe19ebe60cccfc812e01be0d2289a327265fe9ea967524e9edc` |

### Reconstructed 64-event control

| Quantity | Value |
|---|---:|
| mean signal probability | `0.5334414797241307` |
| mean QCD probability | `0.3906304861281349` |
| mean ttbar probability | `0.07592803724258967` |
| events with `p_signal > 0.997` | `7 / 64` |
| three-model top-1 agreement | `0.734375` |
| selected-entry map SHA-256 | `0152a036c61a6cd46f1ef9d622f1c7e37cbe9ba8319197361117056ea58171d9` |
| ensemble score SHA-256 | `78e1478c6c84057e9f914d986748d1b7de16f670675b2b320b0ae82e7d9cd8ef` |
| Events ROOT SHA-256 | `fee7c54109b783952fb17cf47c846b90e30c76d02361a7a18c548a537a095047` |
| comparison receipt SHA-256 | `8e183cf900abeadb092fabfcdf2523e3fc172b183312ba94367066384c1117e3` |

The frozen 64-event ZZ summary was reproduced within the required tolerance. The prior node-local ZZ64 ROOT checksum was not retained, so byte identity was not claimed; the score summaries, process composition, score contract, and DCB-compatible ROOT roundtrip were the scientific gates.

## Validation status

- ZH 64-event control gate: **closed**
- ZH 1,000-event released-model product: **validated**
- ZZ 64-event control gate: **closed**
- ZZ 1,000-event released-model product: **validated**
- repository, d3, and EOS were not modified by the inference runs
- no new cgroup OOM kill was observed
- no Phase 3 work is authorized by this checkpoint

## Next Phase 2 action

Recover and preflight the deterministic 1,000-event QCD control using the frozen source-entry map. The QCD validation must quantify class-136 dominance and the high-signal tail before proceeding to the ttbar control and the final four-control comparison.
