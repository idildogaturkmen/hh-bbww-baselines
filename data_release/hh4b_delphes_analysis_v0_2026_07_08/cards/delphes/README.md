# Delphes detector card

The HH4b production pipeline uses:

`cards/delphes/delphes_card_CMS_lpc.tcl`

This is a CMS-like Delphes detector card stored in the analysis repository. The Delphes executable is run through `DelphesHepMC3`.

The file `delphes_card_CMS.tcl`, if present, is the default Delphes CMS card copied for reference, but the production scripts point to `delphes_card_CMS_lpc.tcl`.
