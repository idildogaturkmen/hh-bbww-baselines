# COLLIDE-1M Background Inventory

Expanded local COLLIDE-1M subset after PVC expansion to 300Gi.

## Disk / file summary

- Local raw parquet directory: `outputs/collide_selected_backgrounds`
- Number of parquet files: 156
- Raw size: about 208 GB
- Workspace after download: 300G total, 213G used, 88G available

## Process group summary

| process group | n samples | n files | total rows |
|---|---:|---:|---:|
| HH_bbWW signal | 1 | 2 | 17,967 |
| HH other | 4 | 8 | 77,876 |
| ttbar / ttV / ttH / tttt | 7 | 33 | 320,909 |
| DY/Z+jets | 5 | 15 | 149,592 |
| W+jets | 2 | 6 | 59,859 |
| diboson | 9 | 27 | 267,033 |
| triboson | 1 | 2 | 19,787 |
| single Higgs | 15 | 45 | 409,451 |
| QCD | 2 | 6 | 51,484 |
| gamma | 2 | 6 | 42,134 |
| minbias | 1 | 3 | 30,000 |
| upsilon | 1 | 3 | 30,000 |

## Notes

This is an expanded, balanced all-category subset, not the full COLLIDE-1M dataset.

No dedicated single-top sample was found in the inspected COLLIDE-1M sample list. The `ttW_incl` single-top warning from the inventory script is a regex false positive; `ttW_incl` is top-pair plus W, not single top.

For HH→bbWW, `HH_bbWW` is the signal. Other HH modes should be treated separately as `HH_other` background/contamination.
