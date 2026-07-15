# CMS Run-2-inspired 3b/4b region tables

These tables implement the mass-region and 3b/4b structure of CMS
HIG-20-005 using AK4 Delphes objects.

Important limitations:

- Delphes Jet.BTag may be binary and is not a calibrated continuous DeepJet
  discriminator.
- The trigger and lepton selections are proxies.
- No CMS collision data are used.
- These tables support a simulation closure study; they are not themselves
  a CMS data-driven background measurement.
