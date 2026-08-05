# Track B Phase 2 mass-grid provenance checkpoint

## Decision

**The exact released-analysis mapping of signal outputs 0–135 to the
exchange-symmetric 16 x 16 mass-cell grid is proven. The independent
training-head output ordering is not publicly documented and requires
producer confirmation.**

- Repository head at staging: `025eff23be0d214aa88a616c2ab8b57b3f4e6e06`
- Published DCB robustness checkpoint ancestor:
  `8858a43d9bc25d5e873ae42dc849bb59a41616ea`
- Official upstream source commit:
  `e8387f7e49a7f27c65b2f4a1f75cacf01cd45392`
- Released DCB source SHA-256:
  `c4814da1cf2b53e33c9c717a213224d179271216300cf955a201893eb18b05e6`

## Proven

- The released model has 138 output nodes.
- Outputs 0–135 are consumed as 136 signal-grid values by the released DCB
  analysis.
- The released grid is 16 x 16 with cell edges 40, 50, ..., 200 GeV.
- The released DCB coordinates are the cell centers 45, 55, ..., 195 GeV.
- Output indices advance in row-major mathematical upper-triangle order:
  outer `i = 0..15`, inner `j = i..15`.
- The grid is symmetrized as `(H + H.T) / 2`.
- The complete 136-row mapping is preserved under
  `evidence/released_output_index_to_mass_cell_mapping.tsv`.
- All three released ONNX models expose one `softmax` output of shape
  `[N, 138]`.

## Not proven

The public repository does not contain the original Weaver training-label
configuration, and the ONNX files contain no custom class-label metadata.
Therefore the released analysis code proves how the 136 signal scores are
interpreted, but it does not independently prove that the unavailable
training pipeline assigned output nodes in exactly the same order.

The paper's lower-triangle wording and the released code's `i <= j`
upper-triangle implementation describe the same unordered cell set under
exchange symmetry. Exact output-node sequencing still requires the producer's
ordered label list or confirmation.

The identities of zero-based background outputs 136 and 137 should also be
confirmed explicitly.

## Terminology policy

Permitted now:

- released-model 136-cell signal-grid response;
- DCB-fitted continuous mass-coordinate estimate derived from the released
  score grid;
- localization in the released mass-cell coordinate system.

Avoid until producer confirmation and calibration closure:

- independently verified training-label index mapping;
- calibrated reconstructed Higgs-boson masses;
- unbiased mass resolution;
- event-by-event true-mass reconstruction.

The values 45, 55, ..., 195 GeV are cell centers used by the released DCB
histogram. They are not by themselves proof of generator point-mass
hypotheses at those values.

## Git-safe source serialization

The source-run artifact manifest preserves the exact 545-byte upstream
function copy with SHA-256 `229b4bcf13a7af05a6a54920a364c98b5835bd9eb3ae1b96aea8913dcac2f067`. That source contained four
otherwise blank lines with four trailing ASCII spaces each.

For a clean Git whitespace gate, the checkpoint copy removes only those 16
space bytes. Its Git-safe SHA-256 is `1b141eb94a74ff93467b90da4d0fc789366dc3dc494e867824cdb8d903f1cebf`. Python AST identity
and exact execution equivalence on the full 136-score / 16-bin mapping probe
are proven in
`evidence/extract_full_histogram_data_git_serialization_receipt.json`.

No executable token, line ending, output value, bin center, or mapping rule
changed.

## Next external action

Send `author_clarification_draft.txt` to the model/data producer and request:

1. the exact ordered list for the 136 signal output nodes;
2. confirmation of the lower/upper triangle convention;
3. confirmation that zero-based outputs 136 and 137 correspond to QCD and
   ttbar;
4. ideally, the original Weaver data configuration or ordered 138-label list.

An empirical training-signal argmax-versus-generator-mass audit can be used as
a supporting cross-check, but not as a substitute for authoritative training
configuration provenance.

## Evidence

- `evidence/summary.json`: final provenance decision and terminology policy.
- `evidence/released_mapping_proof.json`: executable mapping proof.
- `evidence/released_output_index_to_mass_cell_mapping.tsv`: all 136 mappings.
- `evidence/extract_full_histogram_data_source.py`: Git-safe copy of the
  exact released mapping function, normalized only by removing trailing
  spaces from four otherwise blank lines.
- `evidence/extract_full_histogram_data_git_serialization_receipt.json`:
  original and normalized hashes plus AST and execution-equivalence proof.
- `evidence/onnx_model_metadata.json`: three 138-output ONNX models with no
  label metadata.
- `evidence/paper_online_verification.json`: paper-v2 geometry verification.
- `evidence/relevant_source_query_summary.tsv`: public-source query census.
- `source_40J4A_R4_artifact_manifest.json`: hashes for the complete 16-artifact
  source run, including bulky files intentionally not copied into this
  checkpoint.
