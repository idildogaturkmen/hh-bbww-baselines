# FINAL EXECUTION PACKET — postproduction → join → preprocessing → SPA2M+ParT training

**Built:** 2026-09-09, this session, by re-auditing every script in this
package for actual executability (not just reading docstrings) and by
dry-running every step that can be dry-run without touching the live
exact_2M production or launching training. **The live EAF worker1
process was never inspected, touched, or interfered with.** No join, no
preprocessing, no SPA-Net training, no additional-8M/10M ParT production,
no JP-JEPA, no holdout_B, no Stage C was run this session.

**Verdict: `READY_TO_EXECUTE_AFTER_880_220`**. Steps 1–8 (everything through
both training launches) have no known blocker beyond production reaching
880/880 train + 220/220 val. **Update, 2026-09-10 (Track F matched-
evaluation exporter implementation session): G5 (Step 9's one real
blocker) is RESOLVED** — `code/export_model_eval_events.py` now exists in
the authoritative evaluation package
(`track_f_evaluation_readiness_protocol_20260909_v1`), was run for real
against the frozen CONTROL checkpoint (not synthetic, not TEST/ParT2M —
see Step 9 below), and Step 9's evaluation now targets the current,
amended evaluator (`code/evaluate_multi_model.py`, PREREGISTRATION.md
Amendment 2026-09-10), not the superseded `evaluate_matched_2m.py`. Step 9
itself still cannot run to completion until Step 7 produces a TEST
checkpoint — that dependency was never G5 and is unchanged.

---

## 0. What was actually verified this session (not just read)

| # | Finding | How verified |
|---|---|---|
| F1 | `verify_shard_checksums.py`'s core hash-check logic works end-to-end against **real** production data | Ran `check_one_shard()` directly against `train_0000` on EOS: receipt+H5 fetched via `xrdcp`, hashes matched, `pass=true` |
| F2 | `aggregate_embedding_audit.py`'s window-disjointness proof + H5-schema checks (`JOIN_KEY/native_hdf5_row_index`, `JOIN_KEY/jet_slot`, `EMBEDDINGS/part_full` shape `(n,128)` float32) work against real shards | Ran with `--partial-ok --max-shards 3` against real `train_0000..0002`: `AGGREGATE_PASS=true`, all schema/window checks passed |
| F3 | Every hardcoded SHA256 in every script (ParT checkpoint, native train/val H5, manifest, native-control checkpoint) matches the actual file on disk | Independently re-hashed all of them this session; all match exactly |
| F4 | `SPANET_REPO` pinned-commit + clean-working-tree preflight checks in `launch_spa2m_part.py` pass right now | `git rev-parse HEAD` = `debbdc999...`, `git status --porcelain` empty |
| F5 | `launch_spa2m_part.py` reproduces `launch_2M_seed0_training.py`'s hyperparameters **exactly** (seed, batch size, epochs, architecture dict, optimizer dict, loss scales, balance flags, `partial_events`, `dataset_limit`, dataloader workers, trainer construction args, checkpoint monitor/mode/save policy, the weight-loading monkeypatch, the secondary-val-loss subclass) except the two documented, intentional differences (HDF5 paths, event YAML) | Line-by-line diff against the real native launcher (`launch_2M_seed0_training.py`), not the docstring's claim |
| F6 | ParT is genuinely frozen/precomputed in the training path | `launch_spa2m_part.py` never loads a ParT model or runs a ParT forward pass — it only ever reads pre-built, standardized static embedding columns from the augmented H5 |
| F7 | The join is identity-based and fail-closed | Read `streaming_native_part_join.py`: asserts every `(row,slot)` falls in its own shard's declared window, asserts no within-shard duplicate keys, hard-fails if a ParT embedding lands on a native-MASK "not real" slot; writes into the output at the *identity-derived* `[row_lo:row_hi)` slice, never by list order. **Then proved this executes correctly** by building a synthetic 12-event/2-shard population with the exact real schema and running the join for real — 100% coverage, correct output shape `(12,10,135)` |
| F8 | `verify_join_coverage.py`'s 100%-coverage / re-opened-mask / `mask⇒covered` gate works | Ran for real against the synthetic joined output: all 10 checks `true`, `OVERALL_JOIN_VERIFICATION_PASS=true` |
| F9 | Train-only statistics cannot leak validation | `compute_train_embedding_stats.py` takes only `--joined-train-h5` — it never opens, references, or accepts a val path (plus a filename guard). `build_augmented_input.py` applies the **same** frozen train-derived means/stds unchanged to both `--split train` and `--split val` — there is no separate val-stats code path anywhere in the package. Structurally impossible to leak, not just "shouldn't happen." |
| F10 | The active-dimension threshold logic is numerically correct | Built a synthetic population with a known ground truth (30 near-constant dims, 98 real-variance dims). `compute_train_embedding_stats.py`'s data-driven gap threshold (0.00105) and its `1e-3` threshold scan both recovered **exactly** the true 98 active dims (`active_dims == list(range(30,128))` → `True`) |
| F11 | `build_augmented_input.py` produces correct output for both variants | Ran both `--variant active` (→ 105-wide = 7+98) and `--variant all128` (→135-wide) on the synthetic joined H5: correct shapes, `all_finite=true`, masked slots exactly `0.0`, standardized real-jet columns z-scored correctly, `TARGETS`/`CLASSIFICATIONS` copied verbatim from the native H5, named `(N,10)` per-key datasets matching the real `trackb_hh4b.yaml` template's `INPUTS/SEQUENTIAL/Source` structure |
| F12 | `bootstrap_utils.py`'s `roc_auc`/`paired_bootstrap_delta`/`mcnemar_test`/`exact_poisson_count_interval` run correctly | Smoke-tested all four with synthetic arrays; `exact_poisson_count_interval` correctly fell back to its documented normal-approximation path (see gap G3 below) |
| F13 | Native H5 checksums hardcoded in `run_full_join.sh` are still correct | Re-hashed both `production_2M_{train,val}.h5` (676 MB / 135 MB) — exact match |
| F14 | All 8 Python scripts `py_compile` clean; all 3 shell scripts `bash -n` clean | Ran this session, including the new wrapper (§0, gap G4) |

## Gaps / caveats found (none block Steps 1–8; G3 doesn't block anything; G5 is RESOLVED — see 2026-09-10 update below)

- **G1 — "routine" postflight is not actually fast.** `run_postflight_gate.sh`'s Step 2 (`aggregate_embedding_audit.py`) is invoked **without** `--partial-ok`/`--max-shards`, so it unconditionally re-downloads, re-hashes, and re-opens **every one of the 1,100 manifest shards** (≈7–8 GB, ~2,200 `xrdcp` calls) regardless of the `CHECKSUM_SAMPLE_N` setting that only affects Step 4. `POSTFLIGHT_RUNBOOK.md`'s description of the no-arg call as "fast" is stale/misleading — both the "routine" and "`CHECKSUM_SAMPLE_N=full`" invocations pay this same full-corpus cost via Step 2; they only differ in Step 4's own sample size. Budget **30–90+ minutes** for either call, not "fast." Not a defect — the design intentionally makes Step 2 a full completion proof — just a documentation correction.
- **G2 — `JOIN_AND_PREPROCESS_RUNBOOK.md` cites a `receipt.json`'s `dry_run_verification` section that does not exist anywhere in this package.** The prior session's synthetic-H5 dry-run evidence was apparently never written to disk (or was lost). **This session regenerated that evidence for real** (F7–F11 above); the runbook's narrative claim is now actually substantiated, but the missing file itself was not recreated (no script in this package writes one — the claim lived only in prose). Not a blocker, but worth knowing the original citation was dangling.
- **G3 — `scipy` is broken in this LPC shell's default `/usr/bin/python3`** (`numpy 1.23.5` from `/usr/lib64` is missing a working `numpy._typing`, and the user-local `scipy 1.13.1` needs it → `ModuleNotFoundError` on `from scipy import stats`). `bootstrap_utils.py`'s `exact_poisson_count_interval()` already degrades gracefully to its documented normal-approximation fallback in this environment (confirmed, F12) — not fatal, nothing else in this package imports scipy. Only matters for Step 9: if you want the *exact* Garwood CI on sparse-tail rejection working points (not the fallback), run that step in an environment with working scipy/numpy, or fix this one first (e.g. `pip install --user -U numpy scipy` in a scratch venv).
- **G4 — No EAF GPU wrapper existed for `launch_spa2m_part.py`** (unlike the native launcher, which has `run_2M_seed0_training.sh`). **Built this session**: `scripts/train/run_spa2m_part_training.sh`, a direct generalization of the native wrapper's proven pixi-env resolution and `nvidia-smi -L` GPU/MIG discovery-and-validation logic (never a stale UUID, never a placeholder, refuses to guess among multiple visible devices), parameterized for `--variant`. `bash -n` clean; **not run** (no GPU on this node).
- **G5 (the one real blocker, Step 9 only) — no inference/scoring script exists anywhere in the project that produces the `.npz` files `evaluate_matched_2m.py` requires.** That script's own docstring says so explicitly ("this script does not launch inference"). The closest precedent, `evaluate_classification.py` (native-control classification eval, already run and on disk), only extracts the classification signal score + reconstructed process label via a CPU forward pass — it does **not** extract per-event jet-pair assignment correctness, reconstructed Higgs masses, or `event_id`, all of which `evaluate_matched_2m.py`'s documented input schema requires for both CONTROL and TEST. **A new script must be written** (reusing `evaluate_classification.py`'s CPU-forward-pass / `DataLoader(drop_last=False)` / process-label-reconstruction pattern as a starting point, extended to also pull `outputs.assignments`, compute `higgs_mass_{1,2}` from the predicted pairing, and tag `event_id=native_hdf5_row_index`) before Step 9 can actually run. This does not block Steps 1–8; there is naturally time to write it while training runs.

  **RESOLVED 2026-09-10** (Track F matched-evaluation exporter implementation session; history preserved above, not deleted). `code/export_model_eval_events.py` was written in the authoritative evaluation package
  (`/uscms_data/d3/iturkmen/hh4b_delphes/track_f_evaluation_readiness_protocol_20260909_v1/code/`), targeting the CURRENT, amended schema (`schemas/model_eval_events.schema.json`, `pred_b1..pred_b4`/`truth_b1..truth_b4` — NOT the `assignment_correct`/`higgs1`/`higgs2` booleans `evaluate_matched_2m.py`'s own docstring, quoted above, still describes; that predecessor script and its docstring are superseded, not edited). It reuses `evaluate_classification.py`'s exact architecture/options block and `_patched_load_assignments` weight-loading patch (parameterized, not hardcoded), `model.predict(sources).assignments` (the same call `part2_spanet_assignment.py` already uses successfully against this exact checkpoint) for predicted jet-pair indices, and `four_vec()`/`pair_mass()` (copied verbatim from `part3_event_listing_and_truth.py`/`part6_pairing_accuracy_and_mass.py`) for Higgs-mass reconstruction. Truth (`TARGETS/h1/{b1,b2}`, `TARGETS/h2/{b3,b4}`) and raw kinematics (`INPUTS/Source/{pt,eta,phi,mass,MASK}`) are read directly via h5py, independent of the model's own tensor pipeline — deliberately, to stay structurally immune to this project's own documented in-place-tensor-mutation bug precedent (`RUNNER_PROVENANCE_RECEIPT.json`, `track_b_harvey_tail_characterization_20260825_v1`). **Run for real** against the frozen CONTROL checkpoint (read-only inference on an already-public checkpoint; does not touch or reveal the unseen ParT2M/TEST result) — see the evaluation package's `STATUS.md`/`receipt.json` for the full cross-check results (predicted assignments, truth, and pairing-accuracy metrics all matched existing frozen artifacts exactly). Exported: `exports/control_2m_native_eval.npz`. **Step 9 below now points at this exporter and at `code/evaluate_multi_model.py` (not `evaluate_matched_2m.py`, which `PREREGISTRATION.md` Amendment 2026-09-10 superseded).**
- **Minor/cosmetic:** `NATIVE_CONTROL_CHECKPOINT_SHA256` is defined in `launch_spa2m_part.py` but never referenced anywhere in that file (dead constant; harmless — the real check happens implicitly since the CONTROL checkpoint itself is never touched by this launcher). Also: `launch_spa2m_part.py` does not replicate the native launcher's post-fit checkpoint-reload/`state_dict`-key integrity check (`reload_ok`, `missing_keys`, `unexpected_keys`) — a reduced-rigor omission, not a hyperparameter difference, and not required by the "reproduce hyperparameters exactly" instruction; flagging for awareness only.
- **Resources:** disk is not a concern (17 TB free on `/uscms_data/d3`; the full pipeline through both training-input builds needs on the order of 45–50 GB of scratch: ~10 GB local shard sync + ~13 GB joined H5s + ~23 GB for both augmented-input variants × both splits). RAM: `compute_train_embedding_stats.py` peaks at **~4.6–6 GB RSS** at full 2M-train scale (the single `(9,044,239, 128)` float32 real-jet array is the dominant allocation) — this LPC worktree shell currently reports only ~8.5 GB free; run Step 5 on a node/batch slot with more comfortable headroom, not a busy shared interactive shell. Training itself needs the EAF GPU/pixi environment (`torch`/`pytorch_lightning`/the pinned `spanet` clone are **not** installed in this shell's system `python3` at all — expected, training was never meant to run here).

---

## Ordered execution commands

All paths below are absolute. `PKG` is this package's root. Run Steps 1–3
from any LPC/EAF login node with `xrdfs`/`xrdcp`/Kerberos+X.509 access.
Run Steps 4–6 from a node with real disk/RAM headroom (not necessarily
GPU). Run Steps 7–8 **only** from an EAF GPU session. Step 9 is currently
blocked on G5 above.

```bash
PKG=/uscms_data/d3/iturkmen/hh4b_delphes/track_f_postproduction_pipeline_20260908_v1
```

### Step 1 — Final census (cheap, repeat as often as you like while waiting)

```bash
CENSUS_SCRIPT=/uscms_data/d3/iturkmen/hh4b_delphes/track_b_part_postflight_2m_20260907_v1/census/census_2m.py
EOS_ROOT=root://cmseos.fnal.gov//store/user/iturkmen/hh4b_delphes/track_b_part_full_embedding_production_exact_2M_20260824_v1/shards
CKPT_SHA256=61e752f80d7c237d4b18b97705df416a8518dd9e3d5a78a8bbdeebadd787fec0

TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR="$PKG/logs/census_${TS}"
mkdir -p "$OUT_DIR"
xrdfs cmseos.fnal.gov ls "$EOS_ROOT/train" > "$OUT_DIR/train_listing.txt"
xrdfs cmseos.fnal.gov ls "$EOS_ROOT/val"   > "$OUT_DIR/val_listing.txt"
python3 "$CENSUS_SCRIPT" "$OUT_DIR/train_listing.txt" "$OUT_DIR/val_listing.txt" \
    "$CKPT_SHA256" "$OUT_DIR/census.json"
```

**Expected output:** `TRAIN: 880/880 ... gate_pass=True`, `VAL: 220/220 ...
gate_pass=True`, `CHECKPOINT: pass=True`, `OVERALL_CENSUS_PASS = True`,
exit code 0.

**STOP condition:** if `OVERALL_CENSUS_PASS` is not exactly `True` (any
`h5_only`/`receipt_only`/`missing_both`/`duplicate_or_collision_ids`/
`unexpected_ids_outside_expected_range` nonzero, or `complete_pairs` <
expected, or checkpoint hash mismatch) — **stop, do not proceed to Step
2**, wait and re-run Step 1 later. This step is read-only and safe to
re-run unlimited times.

### Step 2 — Routine postflight gate

```bash
bash "$PKG/scripts/postflight/run_postflight_gate.sh"
```

**Expected output:** a fresh `$PKG/logs/postflight_<UTC>/` directory
containing `train_listing.txt`, `val_listing.txt`, `census.json`,
`aggregate_audit.json`, `manifest_cross_check.json`, `checksum_train.json`,
`checksum_val.json`, `GATE_RESULT.json`, `GATE_RESULT.txt`. Per **G1
above, this is not fast** — budget 30–90+ minutes; Step 2/4 of the gate
(`aggregate_embedding_audit.py`) always re-downloads and re-verifies all
1,100 shards' content regardless of this being the "routine" call.

**STOP condition:** proceed only if the command's own exit code is `0`
**and** `GATE_RESULT.json` contains `"OVERALL_POSTFLIGHT_PASS": true`.
Anything else (including 3-of-4 sub-checks passing) — stop, investigate,
do not join. Record this run's `$PKG/logs/postflight_<UTC>/` path.

### Step 3 — Full checksum postflight (immediately before the join, not routinely)

```bash
CHECKSUM_SAMPLE_N=full bash "$PKG/scripts/postflight/run_postflight_gate.sh"
```

**Expected output:** same artifact set as Step 2, but `checksum_train.json`
/`checksum_val.json` cover **every** shard (not a 20-sample spot check) —
same real cost as Step 2 either way (G1). This is the rigorous,
immediately-pre-freeze run; use **this** run's output directory (not
Step 2's) as the gate input to Step 4.

**STOP condition:** identical to Step 2 — `OVERALL_POSTFLIGHT_PASS: true`
and exit 0, or stop.

```bash
GATE_DIR="<the $PKG/logs/postflight_<UTC>/ path this Step 3 run printed>"
```

### Step 4 — Identity-safe join

```bash
JOIN_TS=$(date -u +%Y%m%dT%H%M%SZ)
bash "$PKG/scripts/join/run_full_join.sh" "$GATE_DIR" "$PKG/logs/join_${JOIN_TS}"
JOIN_DIR="$PKG/logs/join_${JOIN_TS}"
```

**Expected output:** `$JOIN_DIR/joined_train.h5` (`joined_features` shape
`(2000000,10,135)` float32 = 7 native + 128 ParT, plus `mask`,
`part_coverage_mask`), `$JOIN_DIR/joined_val.h5` (`(400000,10,135)`),
`JOIN_RECEIPT_train.json`, `JOIN_RECEIPT_val.json`,
`JOIN_RECEIPT_COMBINED.json`. Refuses to run at all if `$GATE_DIR` didn't
pass (hard `exit 11`) or if either native H5's SHA256 has drifted (hard
`exit 13`). Needs ~10 GB (local shard sync) + ~13 GB (joined outputs) of
scratch disk; not RAM-heavy (shard-streaming design, proven at 5/20/50/150
bounded scale already, and now also proven correct end-to-end on a
synthetic population this session, F7–F8).

**STOP condition:** proceed only if the command's own exit code is `0`
**and** `JOIN_RECEIPT_COMBINED.json` has `"OVERALL_JOIN_PASS": true`. If
any per-split `OVERALL_JOIN_VERIFICATION_PASS` is false — stop; do not
compute statistics or train on a partially- or incorrectly-joined file.

### Step 5 — TRAIN-ONLY embedding statistics (never val)

```bash
python3 "$PKG/scripts/preprocess/compute_train_embedding_stats.py" \
    --joined-train-h5 "$JOIN_DIR/joined_train.h5" \
    --out "$JOIN_DIR/PART_TRAIN_STATS_FROZEN.json"
```

**Expected output:** `n_train_events=2000000 n_real_jets_train=9044239
all_finite=True`, a `data_driven_gap_threshold_recommended_for_review`,
and an explicit scan across `{1e-1,1e-2,5e-3,1e-3,1e-4,1e-5}`. Peak RSS
~4.6–6 GB (see resource note above — pick a node with headroom).

**STOP condition — do not skip this judgment call:** read the printed
threshold scan and the data-driven gap threshold. **Only if** the full
2,000,000-train-event population's gap sits close to the prior
provisional `1e-3`/20-active-dim result should you reuse `1e-3` in Step
6 — if the full-population gap lands somewhere materially different,
justify and use *that* value instead. Silently copying `1e-3` without
looking at this file's output is exactly the "silently canonize" failure
mode this step exists to prevent. Record your chosen threshold as
`ACTIVE_THRESHOLD` below.

```bash
ACTIVE_THRESHOLD="<value you justified from Step 5's scan, e.g. 1e-3>"
```

### Step 6 — Build both model-input variants (train + val, both from the SAME frozen train stats)

```bash
YAML_TEMPLATE=/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4S_official_spanet_integration_canary_20260814_v1/event_config/trackb_hh4b.yaml
NATIVE_H5_TRAIN=/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4AF_spanet_partial_events_population_correction_20260819_v1/production_hdf5_build_v1/work/production_2M_train.h5
NATIVE_H5_VAL=/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4AF_spanet_partial_events_population_correction_20260819_v1/production_hdf5_build_v1/work/production_2M_val.h5

# Variant A: primary resource-aware active-dimension input
for SPLIT in train val; do
  NATIVE_VAR="NATIVE_H5_${SPLIT^^}"
  python3 "$PKG/scripts/preprocess/build_augmented_input.py" \
      --joined-h5 "$JOIN_DIR/joined_${SPLIT}.h5" \
      --native-h5 "${!NATIVE_VAR}" \
      --split "${SPLIT}" \
      --train-stats "$JOIN_DIR/PART_TRAIN_STATS_FROZEN.json" \
      --variant active --active-threshold "$ACTIVE_THRESHOLD" \
      --event-yaml-template "$YAML_TEMPLATE" \
      --out-h5 "$JOIN_DIR/spa2m_part_active_${SPLIT}.h5" \
      --out-yaml "$JOIN_DIR/part_augmented_active_hh4b.yaml" \
      --out-receipt "$JOIN_DIR/BUILD_RECEIPT_active_${SPLIT}.json"
done

# Variant B: all-128 sensitivity/control input
for SPLIT in train val; do
  NATIVE_VAR="NATIVE_H5_${SPLIT^^}"
  python3 "$PKG/scripts/preprocess/build_augmented_input.py" \
      --joined-h5 "$JOIN_DIR/joined_${SPLIT}.h5" \
      --native-h5 "${!NATIVE_VAR}" \
      --split "${SPLIT}" \
      --train-stats "$JOIN_DIR/PART_TRAIN_STATS_FROZEN.json" \
      --variant all128 --eps 1e-3 \
      --event-yaml-template "$YAML_TEMPLATE" \
      --out-h5 "$JOIN_DIR/spa2m_part_all128_${SPLIT}.h5" \
      --out-yaml "$JOIN_DIR/part_augmented_all128_hh4b.yaml" \
      --out-receipt "$JOIN_DIR/BUILD_RECEIPT_all128_${SPLIT}.json"
done
```

**Expected output:** 4 H5 files (`spa2m_part_{active,all128}_{train,val}.h5`),
2 event YAMLs, 4 build receipts. `active` variant width = `7 +
n_retained` (verify `n_retained_part_dims` in the receipt matches what
Step 5 justified); `all128` width = `135` exactly. ~23 GB additional
scratch disk across all 4 files.

**STOP condition:** open each `BUILD_RECEIPT_*.json` and confirm
`"all_finite": true` and the expected `n_retained_part_dims` **before**
launching any training. A non-finite value or an unexpected dimension
count here means stop and re-derive Step 5, do not train on it.

### Step 7 — Primary SPA2M+ParT training (active variant)

**Manual gate, by design:** `launch_spa2m_part.py` hardcodes
`AUTHORIZED_TO_RUN = False`. A human must open that file and change it to
`True` after personally reviewing this packet and the Step 6 receipts —
this session does not and will not do that edit, and this command will
refuse with exit code 2 until it's done. Run only on an **EAF GPU
session**, never LPC.

```bash
bash "$PKG/scripts/train/run_spa2m_part_training.sh" active \
    "$JOIN_DIR/spa2m_part_active_train.h5" \
    "$JOIN_DIR/spa2m_part_active_val.h5" \
    "$JOIN_DIR/part_augmented_active_hh4b.yaml" \
    "$JOIN_DIR/BUILD_RECEIPT_active_train.json" \
    spanet_2M_part_active_seed0
```

**Expected output:** `$PKG/scripts/train/work/training_spanet_2M_part_active_seed0_result.json`
plus TensorBoard logs and primary/secondary/`last` checkpoints under
`$PKG/scripts/train/spanet_2M_part_active_seed0/`. Expect wall time on
the order of the native run's measured `total_fit_wall_s≈8577s` (~2.4h)
at the same batch size/epoch budget, modulo the wider input.

**STOP condition:** confirm `exit_status: "COMPLETED"` (not `"FAILED"`),
`n_train_events: 2000000`, `n_val_events: 400000`,
`len(epoch_history) == 50`, and `primary_checkpoint_sha256` is non-null
before treating this as a usable checkpoint for Step 9.

### Step 8 — All-128 sensitivity/control training

Same manual `AUTHORIZED_TO_RUN` gate as Step 7 (already flipped once
Step 7 ran).

```bash
bash "$PKG/scripts/train/run_spa2m_part_training.sh" all128 \
    "$JOIN_DIR/spa2m_part_all128_train.h5" \
    "$JOIN_DIR/spa2m_part_all128_val.h5" \
    "$JOIN_DIR/part_augmented_all128_hh4b.yaml" \
    "$JOIN_DIR/BUILD_RECEIPT_all128_train.json" \
    spanet_2M_part_all128_seed0
```

**Expected output / STOP condition:** identical structure and checks to
Step 7, under `run_name=spanet_2M_part_all128_seed0`.

### Step 9 — Matched evaluation (**G5 RESOLVED 2026-09-10 — see history in "Gaps / caveats" above**)

The `evaluate_matched_2m.py`-based version of this step (quoted in full in
the G5 bullet above, preserved for history, not deleted) is **superseded**:
that script's own schema (`assignment_correct`, `higgs1_assignment_correct`,
`higgs2_assignment_correct`) was found, in a pre-result audit, to silently
assume a canonical H1-vs-H2 labeling that does not exist (the two Higgs
candidates are an explicit `PERMUTATIONS.EVENT: [h1,h2]` symmetry) —
`PREREGISTRATION.md` Amendment 2026-09-10 in the authoritative evaluation
package corrected this before any ParT2M result existed. **Use the current
exporter and evaluator below, not `evaluate_matched_2m.py`.**

`EVAL_PKG=/uscms_data/d3/iturkmen/hh4b_delphes/track_f_evaluation_readiness_protocol_20260909_v1`

**9a. Export CONTROL** (already done, real, this session — read-only
inference against the frozen, already-public native SPA2M checkpoint; does
not touch or reveal TEST/ParT2M):

```bash
PYBIN=/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4U_eaf_official_spanet_gpu_canary_20260815_v1/.pixi/envs/default/bin/python3
TRACK_B=/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812
PHASE4AF="$TRACK_B/phase4AF_spanet_partial_events_population_correction_20260819_v1"

"$PYBIN" "$EVAL_PKG/code/export_model_eval_events.py" \
    --arm control \
    --checkpoint "$PHASE4AF/spanet_2M_seed0_production_training_v1/spanet_2M_seed0/version_0/checkpoints/primary-epoch=49-step=48800-validation_average_jet_accuracy=0.4912.ckpt" \
    --checkpoint-sha256 dc39cf76f0d8e40d07228f240fe58b1179e8134cd4f96c5c786a7d528d31ea8d \
    --input-h5 "$PHASE4AF/production_hdf5_build_v1/work/production_2M_val.h5" \
    --input-h5-sha256 3c94bf8300d1dc3324e2cf25f9b6c618dffb94819cb3a297a43ac06320d5c2a2 \
    --hdf5-train "$PHASE4AF/production_hdf5_build_v1/work/production_2M_train.h5" \
    --hdf5-train-sha256 fe9e4d8fa501e24c9585fa18f59caafe8c7631337f8e0428094fa3229e09fe76 \
    --event-yaml "$TRACK_B/phase4S_official_spanet_integration_canary_20260814_v1/event_config/trackb_hh4b.yaml" \
    --event-yaml-sha256 278885c34c53aee73ceece060d4f464e5195780f3e8b77594d8a2e4b6fd60ed3 \
    --spanet-repo "$TRACK_B/phase4AE_spanet_10jet_literature_aligned_training_preflight_20260818_v1/gpu_canary_10jet_trainonlyweighted_v23exact_r2/spanet_repo_v23exact" \
    --spanet-pinned-commit debbdc999bfb785eb110a36c5fd3eff211ebf234 \
    --val-manifest-tsv "$PHASE4AF/production_population_contract_gate_v23exact_r2_nested_training_scaling_20260819_v1/work/manifest_validation_400k.tsv" \
    --out-npz "$PKG/exports/control_2m_native_eval.npz"
```

Already run: 400,000/400,000 events scored (~2,300 events/s, ~174s wall),
process composition `signal=193358 qcd=174485 ttbar=32157` (exact match to
`FROZEN_FACTS.json`), checkpoint loaded with zero missing/unexpected keys,
all hard validations passed. Cross-checked against existing frozen
artifacts (see `$EVAL_PKG/STATUS.md` for full numbers): truth
(`truth_targets_400k.npz`) exact match on all 400,000 events; classification
AUC matches `classification_evaluation_result.json` to ~1e-10 (floating-
point-noise level); predicted assignments match
`spanet2m_signal_pairing_sample.npz`'s 10,000-event sample on 99.11% of
rows exactly, with the 0.89% residual attributable to CPU floating-point
non-determinism on near-tied assignment decisions (confirmed same
torch 2.8.0+cu128/numpy 2.0.2 in both environments — not a version
mismatch), producing a symmetric, non-systematic ±0.0004-0.0005 shift in
aggregate pairing-accuracy metrics, not a directional bias.

**9b. Export TEST** (**NOT run — SPA2M+ParT does not exist yet**; exact
future command, once Step 7 completes):

```bash
# Read the actual checkpoint path/hash from Step 7's own result JSON once
# it exists -- do not guess the filename (it encodes epoch/step/metric
# values only known after training, exactly like CONTROL's own filename).
TEST_RESULT_JSON="$PKG/scripts/train/work/training_spanet_2M_part_active_seed0_result.json"
# TEST_CKPT=<primary checkpoint path recorded in $TEST_RESULT_JSON>
# TEST_CKPT_SHA256=<primary_checkpoint_sha256 recorded in $TEST_RESULT_JSON>
# TEST_EVENT_YAML_SHA256=$(sha256sum "$JOIN_DIR/part_augmented_active_hh4b.yaml" | cut -d' ' -f1)

"$PYBIN" "$EVAL_PKG/code/export_model_eval_events.py" \
    --arm test \
    --checkpoint "$TEST_CKPT" --checkpoint-sha256 "$TEST_CKPT_SHA256" \
    --input-h5 "$JOIN_DIR/spa2m_part_active_val.h5" \
    --input-h5-sha256 "$(sha256sum "$JOIN_DIR/spa2m_part_active_val.h5" | cut -d' ' -f1)" \
    --hdf5-train "$JOIN_DIR/spa2m_part_active_train.h5" \
    --hdf5-train-sha256 "$(sha256sum "$JOIN_DIR/spa2m_part_active_train.h5" | cut -d' ' -f1)" \
    --event-yaml "$JOIN_DIR/part_augmented_active_hh4b.yaml" --event-yaml-sha256 "$TEST_EVENT_YAML_SHA256" \
    --spanet-repo "$TRACK_B/phase4AE_spanet_10jet_literature_aligned_training_preflight_20260818_v1/gpu_canary_10jet_trainonlyweighted_v23exact_r2/spanet_repo_v23exact" \
    --spanet-pinned-commit debbdc999bfb785eb110a36c5fd3eff211ebf234 \
    --val-manifest-tsv "$PHASE4AF/production_population_contract_gate_v23exact_r2_nested_training_scaling_20260819_v1/work/manifest_validation_400k.tsv" \
    --out-npz "$PKG/exports/test_2m_part_active_eval.npz"
```

Same CONTROL/TEST exporter path, same event ordering/identity semantics
(both proven via the same dataloader-vs-raw-HDF5 label identity check,
independently, per invocation) — nothing in the exporter differs between
arms except which checkpoint/H5/YAML it is pointed at.

**9c. Run the current evaluator** (`code/evaluate_multi_model.py`, not
`evaluate_matched_2m.py`) once both exports exist:

```bash
python3 "$EVAL_PKG/code/evaluate_multi_model.py" \
    --control-npz "$PKG/exports/control_2m_native_eval.npz" \
    --test-npz    "$PKG/exports/test_2m_part_active_eval.npz" \
    --control-training-result "$PHASE4AF/spanet_2M_seed0_production_training_v1/work/training_2M_seed0_result.json" \
    --test-training-result "$TEST_RESULT_JSON" \
    --build-receipt "$JOIN_DIR/BUILD_RECEIPT_active_val.json" \
    --native10m-aggregate-json /uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4AI_spanet_10M_scaling_seed0_20260821_v1/classification_evaluation_10M/work/classification_evaluation_10M_result.json \
    --seed 0 --n-boot 10000 \
    --out-json "$JOIN_DIR/comparison_result_seed0.json"
```

**Expected output:** `comparison_result_seed0.json`
(`schemas/comparison_result.schema.json`) with AUC (all-bg/QCD/ttbar),
paired-bootstrap ΔAUC, rejection-at-fixed-efficiency (predeclared
`PREREGISTRATION.md` εS points), the symmetry-safe McNemar reconstruction
comparison, jet-multiplicity-stratified AUC, resource comparison, and
single-seed story flags. Labeled
`MATCHED_DEVELOPMENT_VALIDATION_RESULT_NOT_FINAL_TEST_PERFORMANCE` by the
script itself — repeat this labeling in any downstream write-up (per
`FROZEN_FACTS.json:data_policy.labeling_rule`).

**STOP condition:** the script itself hard-fails (exit 2/3) if the
control/test `event_id` or `process` arrays don't match after sorting —
treat that as a real identity bug, not a warning, and do not report
anything from that run. If additional seeds are called for
(`PREREGISTRATION.md` section 7.1), repeat 9b/9c per seed and run
`decide_go_no_go.py` across all of them before any GO/NO-GO conclusion.
If G3 (scipy broken in this LPC shell's default `/usr/bin/python3`) is
still unresolved in whatever environment runs `evaluate_multi_model.py`,
note in the writeup that sparse-tail Poisson/binomial intervals used the
normal-approximation fallback, not the exact method (the pixi environment
used for 9a/9b above has a working scipy, confirmed this session — prefer
running 9c there too if exact intervals matter).

---

## Final verdict

**`READY_TO_EXECUTE_AFTER_880_220`** for Steps 1–8 (postproduction census
→ postflight → identity-safe join → train-only statistics → both
augmented-input variants → both training launches). Nothing in that path
depends on anything except production reaching 880/880 train + 220/220
val — every script involved was read for actual executability against the
real schema, every hardcoded hash/path was independently re-verified
against real files this session, and every step that could be dry-run
without touching production or launching training was dry-run for real
(not just reasoned about) and passed, including an exact ground-truth
match on the active-dimension detection logic.

**G5 is RESOLVED (2026-09-10, history preserved above, not deleted).** The
CONTROL/TEST `.npz` inference/scoring script now exists
(`track_f_evaluation_readiness_protocol_20260909_v1/code/
export_model_eval_events.py`), targets the current, amended schema and
evaluator (not the superseded `evaluate_matched_2m.py`), and has been **run
for real** against the frozen CONTROL checkpoint — 400,000/400,000 events,
all hard validations passed, cross-checked against existing frozen
artifacts (truth exact on all 400k events; classification AUC to ~1e-10;
predicted assignments 99.11% exact on a 10k-event sample, residual
explained by CPU floating-point non-determinism, not a version mismatch or
logic error). **Step 9's remaining dependency is unchanged and was never
part of G5**: it still cannot run to completion until Step 7 produces a
real TEST checkpoint (no training was launched by this exporter work, and
none is authorized here). `EXPORTER_READY_FOR_CONTROL_AND_FUTURE_TEST` —
see `track_f_evaluation_readiness_protocol_20260909_v1/STATUS.md` for the
full accounting.
