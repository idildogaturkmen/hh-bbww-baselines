# AI-assisted Delphes HH4b workflow

This branch uses a human-in-the-loop, AI-assisted workflow for Delphes-based HH→4b studies.

## Goal

Use AI coding/reasoning tools to accelerate repetitive analysis tasks while keeping physics validation under human control.

## Current scope

- Generate controlled Delphes HH→4b signal and background samples.
- Version-control generator cards, Delphes cards, sample manifests, and analysis scripts.
- Convert Delphes ROOT outputs to parquet.
- Validate object selection, b-tagging, truth matching, event weights, and sample normalization.
- Compare cut-based, BDT/DNN, SPA-Net, and later particle-aware approaches.

## Human-in-the-loop validation checklist

Before trusting any result, check:

1. The generator process is physically correct.
2. The center-of-mass energy, Higgs decay mode, and sample cuts are documented.
3. Cross sections come from generator logs or trusted references, not placeholders.
4. Delphes card and b-tagging assumptions are version-controlled.
5. Event counts, sum of weights, and filter efficiencies are saved per shard.
6. ROOT-to-parquet conversion preserves jets, b-tags, leptons, MET, and GenParticle information.
7. Truth matching is validated on event displays or printed examples.
8. Signal/background comparisons use the same object selection.
9. Absolute significance is not quoted until cross sections and systematics are defined.
10. AI-generated code is reviewed, tested, and committed only after manual validation.

## Tool roles

- Copilot: code completion and boilerplate inside VS Code.
- ChatGPT/Claude/Codex-style tools: planning, debugging, review, validation checklists.
- Human analyst: physics definitions, sample strategy, validation, interpretation.
