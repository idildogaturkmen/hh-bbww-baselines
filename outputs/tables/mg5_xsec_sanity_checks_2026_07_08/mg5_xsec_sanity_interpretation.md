# MG5 cross-section sanity checks

These checks were run to answer whether MG5 cross sections are globally near 1 or whether the HH signal normalization issue is process-specific.

Main conclusions:

1. MG5 is not globally producing cross sections near 1.
   - Z→ee gives 1388.7 pb.
   - ggF single-Higgs stable in HEFT gives 17.621 pb.

2. The HEFT model/card effective H→bb ratio is:
   - H→bb / H stable = 0.6618

3. The ggF HH decay handling is internally consistent:
   - ggF HH→4b / ggF HH stable = 0.4366
   - This is close to BR(H→bb)^2 using the MG5 effective BR: 0.4379

4. The ggF production normalization remains the issue:
   - stable ggF HH in this HEFT setup is 2.278 fb, much lower than the official SM ggF HH reference.

5. The VBF decay-chain check is suspicious:
   - VBF HHjj→4b / VBF HHjj stable = 0.6796
   - This is closer to one H→bb branching ratio than BR(H→bb)^2, so the VBF process/decay definition needs another check.

Current recommendation:
- Use MG5 samples for shapes and efficiencies.
- Confirm with Harvey whether final signal normalization should be external official SM HH cross sections × BR(H→bb)^2.
