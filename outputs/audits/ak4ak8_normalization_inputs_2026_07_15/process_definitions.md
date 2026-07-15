# AK4/AK8 normalization-input audit

## Delphes card

- Path: `/uscms_data/d3/iturkmen/repos/hh-bbww-baselines/cards/delphes/delphes_card_CMS_lpc_ak4ak8.tcl`
- Exists: `True`
- SHA256: `b7905672263984680d2d92fa043d9881517cf108fb1dc0f72ceaa34c5ee805d7`

## MG5 process definitions

### ggF_HH_4b: `HH4b_ggf_heft`

**Process definition**

```text
import model /uscms_data/d3/iturkmen/hh4b_delphes/mg5_models/heft | define j = g u c d s u~ c~ d~ s~ | define l+ = e+ mu+ | define l- = e- mu- | define vl = ve vm vt | define vl~ = ve~ vm~ vt~ | define p = g | generate p p > h h, (h > b b~), (h > b b~) | output /uscms_data/d3/iturkmen/hh4b_delphes/mg5/HH4b_ggf_heft -f
```

**Important run-card parameters**

```json
{"cut_decays": "False", "drbb": "0.0", "ebeam1": "6500.0", "ebeam2": "6500.0", "etab": "-1.0", "iseed": "0", "lhaid": "230000", "maxjetflavor": "4", "mmbb": "0.0", "nevents": "10000", "pdlabel": "nn23lo1", "ptb": "0.0"}
```

### ggF_HH_4b: `HH4b_ggf_heft_smoke`

**Process definition**

```text
import model /uscms_data/d3/iturkmen/hh4b_delphes/mg5_models/heft | define j = g u c d s u~ c~ d~ s~ | define l+ = e+ mu+ | define l- = e- mu- | define vl = ve vm vt | define vl~ = ve~ vm~ vt~ | define p = g | generate p p > h h, (h > b b~), (h > b b~) | output /uscms_data/d3/iturkmen/hh4b_delphes/mg5/HH4b_ggf_heft_smoke -f
```

**Important run-card parameters**

```json
{"cut_decays": "False", "drbb": "0.4", "drbj": "0.4", "ebeam1": "6500.0", "ebeam2": "6500.0", "etab": "2.7", "etaj": "5.0", "iseed": "0", "lhaid": "230000", "maxjetflavor": "4", "mmbb": "0.0", "nevents": "10000", "pdlabel": "nn23lo1", "ptb": "25.0", "ptj": "20.0"}
```

### ggF_HH_4b: `HH4b_ggf_loop_smoke`

**Process definition**

```text
import model loop_sm | define j = g u c d s u~ c~ d~ s~ | define l+ = e+ mu+ | define l- = e- mu- | define vl = ve vm vt | define vl~ = ve~ vm~ vt~ | define p = g | generate p p > h h [virt=QCD] | output /uscms_data/d3/iturkmen/hh4b_delphes/mg5/HH4b_ggf_loop_smoke -f
```

**Important run-card parameters**

```json
{"ebeam1": "6500.0", "ebeam2": "6500.0", "iseed": "0", "lhaid": "230000", "maxjetflavor": "4", "nevents": "10000", "pdlabel": "nn23lo1"}
```

### VBF_HH_4b: `HH4b_smoke_vbf`

**Process definition**

```text
import model sm | define l+ = e+ mu+ | define l- = e- mu- | define vl = ve vm vt | define vl~ = ve~ vm~ vt~ | define p = g u c d s u~ c~ d~ s~ | define j = g u c d s u~ c~ d~ s~ | generate p p > h h j j QCD=0, (h > b b~), (h > b b~) | output /uscms_data/d3/iturkmen/hh4b_delphes/mg5/HH4b_smoke_vbf -f
```

**Important run-card parameters**

```json
{"cut_decays": "False", "drbb": "0.4", "drbj": "0.4", "drjj": "0.4", "ebeam1": "6500.0", "ebeam2": "6500.0", "etab": "2.7", "etaj": "5.0", "iseed": "0", "lhaid": "230000", "maxjetflavor": "4", "mmbb": "0.0", "mmjj": "0.0", "nevents": "10000", "pdlabel": "nn23lo1", "ptb": "25.0", "ptj": "20.0"}
```

### QCD_bbbb: `QCD_bbbb_presel_smoke`

**Process definition**

```text
import model sm | define j = g u c d s u~ c~ d~ s~ | define l+ = e+ mu+ | define l- = e- mu- | define vl = ve vm vt | define vl~ = ve~ vm~ vt~ | define p = g u c d s u~ c~ d~ s~ | generate p p > b b~ b b~ QED=0 | output /uscms_data/d3/iturkmen/hh4b_delphes/mg5/QCD_bbbb_presel_smoke \
```

**Important run-card parameters**

```json
{"drbb": "0.4", "drbj": "0.4", "ebeam1": "6500.0", "ebeam2": "6500.0", "etab": "2.7", "etaj": "5.0", "iseed": "0", "lhaid": "230000", "maxjetflavor": "4", "mmbb": "0.0", "nevents": "10000", "pdlabel": "nn23lo1", "ptb": "25.0", "ptj": "20.0"}
```

### QCD_bbbb: `QCD_bbbb_smoke`

**Process definition**

```text
import model sm | define j = g u c d s u~ c~ d~ s~ | define l+ = e+ mu+ | define l- = e- mu- | define vl = ve vm vt | define vl~ = ve~ vm~ vt~ | define p = g u c d s u~ c~ d~ s~ | generate p p > b b~ b b~ | output /uscms_data/d3/iturkmen/hh4b_delphes/mg5/QCD_bbbb_smoke -f
```

**Important run-card parameters**

```json
{"drbb": "0.0", "ebeam1": "6500.0", "ebeam2": "6500.0", "etab": "-1.0", "iseed": "0", "lhaid": "230000", "maxjetflavor": "4", "mmbb": "0.0", "nevents": "20", "pdlabel": "nn23lo1", "ptb": "0.0"}
```

### ttbar: `TTbar_smoke`

**Process definition**

```text
import model sm | define p = g u c d s u~ c~ d~ s~ | define j = g u c d s u~ c~ d~ s~ | define l+ = e+ mu+ | define l- = e- mu- | define vl = ve vm vt | define vl~ = ve~ vm~ vt~ | define p = g u c d s b u~ c~ d~ s~ b~ | generate p p > t t~ | output /uscms_data/d3/iturkmen/hh4b_delphes/mg5/TTbar_smoke -f
```

**Important run-card parameters**

```json
{"drbb": "0.4", "drbj": "0.4", "ebeam1": "6500.0", "ebeam2": "6500.0", "etab": "2.7", "etaj": "5.0", "iseed": "0", "lhaid": "230000", "maxjetflavor": "5", "nevents": "10000", "pdlabel": "nn23lo1", "ptb": "25.0", "ptj": "20.0"}
```

### other: `TTbb_smoke`

**Process definition**

```text
import model sm | define j = g u c d s u~ c~ d~ s~ | define l+ = e+ mu+ | define l- = e- mu- | define vl = ve vm vt | define vl~ = ve~ vm~ vt~ | define p = g u c d s u~ c~ d~ s~ | generate p p > t t~ b b~ QED=0 | output /uscms_data/d3/iturkmen/hh4b_delphes/mg5/TTbb_smoke -f
```

**Important run-card parameters**

```json
{"drbb": "0.4", "drbj": "0.4", "ebeam1": "6500.0", "ebeam2": "6500.0", "etab": "2.7", "etaj": "5.0", "iseed": "0", "lhaid": "230000", "maxjetflavor": "4", "mmbb": "0.0", "nevents": "5000", "pdlabel": "nn23lo1", "ptb": "25.0", "ptj": "20.0"}
```

### ZH_4b: `ZH4b_smoke`

**Process definition**

```text
import model sm | define j = g u c d s u~ c~ d~ s~ | define l+ = e+ mu+ | define l- = e- mu- | define vl = ve vm vt | define vl~ = ve~ vm~ vt~ | define p = g u c d s u~ c~ d~ s~ | generate p p > z h, (z > b b~), (h > b b~) | output /uscms_data/d3/iturkmen/hh4b_delphes/mg5/ZH4b_smoke -f
```

**Important run-card parameters**

```json
{"cut_decays": "False", "drbb": "0.4", "drbj": "0.4", "ebeam1": "6500.0", "ebeam2": "6500.0", "etab": "2.7", "etaj": "5.0", "iseed": "0", "lhaid": "230000", "maxjetflavor": "4", "mmbb": "0.0", "nevents": "10000", "pdlabel": "nn23lo1", "ptb": "25.0", "ptj": "20.0"}
```

### ZZ_4b: `ZZ4b_smoke`

**Process definition**

```text
import model sm | define j = g u c d s u~ c~ d~ s~ | define l+ = e+ mu+ | define l- = e- mu- | define vl = ve vm vt | define vl~ = ve~ vm~ vt~ | define p = g u c d s u~ c~ d~ s~ | generate p p > z z, (z > b b~), (z > b b~) | output /uscms_data/d3/iturkmen/hh4b_delphes/mg5/ZZ4b_smoke -f
```

**Important run-card parameters**

```json
{"cut_decays": "False", "drbb": "0.4", "drbj": "0.4", "ebeam1": "6500.0", "ebeam2": "6500.0", "etab": "2.7", "etaj": "5.0", "iseed": "0", "lhaid": "230000", "maxjetflavor": "4", "mmbb": "0.0", "nevents": "10000", "pdlabel": "nn23lo1", "ptb": "25.0", "ptj": "20.0"}
```

### other: `Zbb_smoke`

**Process definition**

```text
import model sm | define j = g u c d s u~ c~ d~ s~ | define l+ = e+ mu+ | define l- = e- mu- | define vl = ve vm vt | define vl~ = ve~ vm~ vt~ | define p = g u c d s u~ c~ d~ s~ | generate p p > z b b~, z > b b~ | output /uscms_data/d3/iturkmen/hh4b_delphes/mg5/Zbb_smoke -f
```

**Important run-card parameters**

```json
{"cut_decays": "False", "drbb": "0.4", "ebeam1": "6500.0", "ebeam2": "6500.0", "etab": "2.7", "etaj": "5.0", "iseed": "0", "lhaid": "230000", "maxjetflavor": "4", "mmbb": "0.0", "nevents": "1000", "pdlabel": "nn23lo1", "ptb": "25.0", "ptj": "20.0"}
```

### Zbbbb: `Zbbbb_presel_smoke`

**Process definition**

```text
import model sm | define j = g u c d s u~ c~ d~ s~ | define l+ = e+ mu+ | define l- = e- mu- | define vl = ve vm vt | define vl~ = ve~ vm~ vt~ | define p = g u c d s u~ c~ d~ s~ | generate p p > z b b~, z > b b~ | output /uscms_data/d3/iturkmen/hh4b_delphes/mg5/Zbbbb_presel_smoke -f
```

**Important run-card parameters**

```json
{"cut_decays": "False", "drbb": "0.4", "drbj": "0.4", "ebeam1": "6500.0", "ebeam2": "6500.0", "etab": "2.7", "etaj": "5.0", "iseed": "0", "lhaid": "230000", "maxjetflavor": "4", "mmbb": "0.0", "nevents": "10000", "pdlabel": "nn23lo1", "ptb": "25.0", "ptj": "20.0"}
```

