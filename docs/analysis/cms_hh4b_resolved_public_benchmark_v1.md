# CMS HH→4b resolved public benchmark v1

Primary official source: [CMS HIG-24-010 publication page](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIG-24-010/index.html).

| fact | value | direct comparison | caveat |
|---|---:|---|---|
| topology | resolved HH→4b | closest public topology | CMS uses collision data, full detector reconstruction, control regions, and a multibin likelihood |
| collision energy | 13 TeV | yes | Run 2 only |
| luminosity | 138 fb⁻¹ | yes | same luminosity used in the Delphes projection |
| expected 95% CL upper limit on inclusive μHH | 5.9 | primary benchmark | official CMS expected limit; not reproduced by this repository |
| observed 95% CL upper limit on inclusive μHH | 10.0 | context only | data-dependent and not comparable to a background-only Delphes expectation |
| Run-2+Run-3 combined expected limit | 2.8 | no | includes 13.6 TeV Run 3 and merged-topology information |

The repository result must be called a **Delphes simulation expected-limit diagnostic**, not an official CMS result. Its likelihood omits experimental systematics and cannot reproduce CMS's data-driven QCD model.
