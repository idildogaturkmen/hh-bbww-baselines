# Delphes realism and JetClass-II comparison note

The current HH4b production uses a CMS-like Delphes fast-simulation card:

`cards/delphes/delphes_card_CMS_lpc.tcl`

This is a jet-level Delphes baseline using reconstructed jets and Delphes b-tagging information.

A recent jet-free HH→4b study uses a more advanced JetClass-II Delphes configuration with:
- modified CMS Delphes card,
- d0 and dz impact-parameter smearing,
- pileup with average 50 interactions,
- tuned PUPPI pileup mitigation,
- particle-level inputs,
- custom Sophon/SophonAK4 taggers.

That setup is more realistic for particle-level event taggers, especially because impact-parameter and pileup-mitigated particle features enter the neural-network inputs.

For this SURF timeline, I will not regenerate all samples with a JetClass-II-style card. Instead, I will clearly frame the present work as a CMS-like Delphes jet-level baseline. The current goal is to build a rigorous and reproducible baseline for:
- resolved HH→4b reconstruction,
- QCD bbbb importance sampling,
- top/heavy-flavor background diagnostics,
- comparison of jet-level ML baselines.

A future extension would be to save slim particle/constituent-level parquets from Delphes ROOT files and test particle-aware embeddings or taggers inspired by JetClass-II, Sophon, ParT, GloParT, PHAT-JeT, or JP-JEPA.
