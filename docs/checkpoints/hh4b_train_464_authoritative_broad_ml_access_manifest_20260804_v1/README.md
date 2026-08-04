# Authoritative 464-source broad-ML train access manifest

This checkpoint freezes the passed train-source access manifest at source
commit `8c5bc84d59a860ebca6896cae62c3e7df8f59cf3`.

## Closed train population

- 464 train sources
- 3,799,873 generated train events
- 441 primary physical sources
- 23 train-only auxiliary `qcd_bbbb` sources
- 442 remote-bundle source rows
- 22 local direct-ROOT source rows
- 87 primary signal bundle sources, including 7 VBF sources

All 441 primary sources have authoritative extraction contracts.

The 23 auxiliary `qcd_bbbb` sources remain classification-only and require
representative archive-member and checksum validation in the seven-class
broad-extractor canary.

Validation and test payloads were not opened, and no model was trained while
building this checkpoint.

## Next gate

Run the seven-class representative broad-extractor canary before any
464-source train extraction is submitted.
