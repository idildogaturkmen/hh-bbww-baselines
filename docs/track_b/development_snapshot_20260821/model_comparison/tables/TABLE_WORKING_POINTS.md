# TABLE_WORKING_POINTS

**DEVELOPMENT ONLY -- not a final/governing result, not independent Stage-C inference, not physically normalized.**

R = 1/epsB (background rejection). Finite-support warnings (Poisson-limited or zero-survivor operating points) are listed per row where applicable -- treat those R values as unreliable or undefined, not precise estimates.

| Model | target epsS | achieved epsS | R all-bg | R QCD | R ttbar | n_sig pass | n_bg pass | n_QCD pass | n_ttbar pass | finite-support warning |
|---|---|---|---|---|---|---|---|---|---|---|
| BDT-K | 0.60 | 0.6000 | 7.3 | 7.1 | 8.2 | 116,015 | 28,354 | 24,447 | 3,907 |  |
| BDT-K | 0.50 | 0.5000 | 10.8 | 10.7 | 11.0 | 96,679 | 19,177 | 16,254 | 2,923 |  |
| BDT-K | 0.40 | 0.4000 | 16.8 | 17.1 | 15.3 | 77,343 | 12,292 | 10,192 | 2,100 |  |
| BDT-K | 0.25 | 0.2500 | 39.1 | 41.3 | 30.4 | 48,340 | 5,282 | 4,225 | 1,057 |  |
| BDT-K | 0.20 | 0.2000 | 57.0 | 61.0 | 41.9 | 38,672 | 3,628 | 2,861 | 767 |  |
| BDT-K | 0.10 | 0.1000 | 163.6 | 181.8 | 106.1 | 19,336 | 1,263 | 960 | 303 |  |
| BDT-KF | 0.60 | 0.6000 | 121.8 | 114.0 | 193.7 | 116,015 | 1,697 | 1,531 | 166 |  |
| BDT-KF | 0.50 | 0.5000 | 225.3 | 211.8 | 345.8 | 96,679 | 917 | 824 | 93 |  |
| BDT-KF | 0.40 | 0.4000 | 468.6 | 438.4 | 747.8 | 77,343 | 441 | 398 | 43 |  |
| BDT-KF | 0.25 | 0.2500 | 1,497.4 | 1,363.2 | 3,215.7 | 48,340 | 138 | 128 | 10 |  |
| BDT-KF | 0.20 | 0.2000 | 2,431.1 | 2,208.7 | 5,359.5 | 38,672 | 85 | 79 | 6 | ttbar: only 6 survivors at epsS=0.2 - efficiency/rejection is low-statistics (Poisson-limited), not a precise estimate |
| BDT-KF | 0.10 | 0.1000 | 11,480.1 | 9,693.6 | undefined | 19,336 | 18 | 18 | 0 | ttbar: zero survivors at epsS=0.1 (threshold=0.997453) - efficiency=0 exactly, rejection undefined (only a lower bound RB > 32157 is supported) |
| SPA-Net 2M (primary) | 0.60 | 0.6000 | 135.9 | 129.1 | 191.4 | 116,015 | 1,520 | 1,352 | 168 |  |
| SPA-Net 2M (primary) | 0.50 | 0.5000 | 275.2 | 262.4 | 373.9 | 96,679 | 751 | 665 | 86 |  |
| SPA-Net 2M (primary) | 0.40 | 0.4000 | 574.0 | 548.7 | 765.6 | 77,344 | 360 | 318 | 42 |  |
| SPA-Net 2M (primary) | 0.25 | 0.2500 | 2,246.1 | 2,127.9 | 3,215.7 | 48,340 | 92 | 82 | 10 |  |
| SPA-Net 2M (primary) | 0.20 | 0.2000 | 4,132.8 | 3,965.6 | 5,359.5 | 38,672 | 50 | 44 | 6 | ttbar: only 6 survivors at epsS=0.2 - efficiency/rejection is low-statistics (Poisson-limited), not a precise estimate |
| SPA-Net 2M (primary) | 0.10 | 0.1000 | 18,785.6 | 17,448.5 | 32,157.0 | 19,337 | 11 | 10 | 1 | ttbar: only 1 survivors at epsS=0.1 - efficiency/rejection is low-statistics (Poisson-limited), not a precise estimate |
| SPA-Net 10M (primary) | 0.60 | 0.6000 | 132.8 | 124.5 | 207.5 | 116,015 | 1,556 | 1,401 | 155 |  |
| SPA-Net 10M (primary) | 0.50 | 0.5000 | 260.3 | 245.4 | 387.4 | 96,679 | 794 | 711 | 83 |  |
| SPA-Net 10M (primary) | 0.40 | 0.4000 | 510.2 | 476.7 | 824.5 | 77,343 | 405 | 366 | 39 |  |
| SPA-Net 10M (primary) | 0.25 | 0.2500 | 1,895.8 | 1,762.5 | 3,215.7 | 48,341 | 109 | 99 | 10 | ttbar: only 10 survivors at epsS=0.25 - efficiency/rejection is low-statistics (Poisson-limited), not a precise estimate |
| SPA-Net 10M (primary) | 0.20 | 0.2000 | 4,132.8 | 3,712.4 | 10,719.0 | 38,672 | 50 | 47 | 3 | ttbar: only 3 survivors at epsS=0.2 - efficiency/rejection is low-statistics (Poisson-limited), not a precise estimate |
| SPA-Net 10M (primary) | 0.10 | 0.1000 | 29,520.3 | 24,926.4 | undefined | 19,336 | 7 | 7 | 0 | ttbar: zero survivors at epsS=0.1 (threshold=0.998676) - efficiency=0 exactly, rejection undefined (only a lower bound RB > 32157 is supported) |
