# Next experiments — updated by the ZERO20 result

None of the experiments below are authorized or running. This page ranks
them by how much they would actually teach us, given what
[`zero20_width_control.md`](zero20_width_control.md) already ruled out.
**Ranking before ZERO20 existed is not repeated here uncritically — ZERO20
changes which follow-up is most informative**, and that is the point of
this page.

## What ZERO20 already answered

Input-embedding width (7→27) is **not** the mechanism — ZERO20 reproduced
none of the harm despite sharing the wider architecture exactly. The
remaining candidate mechanisms are both about the *content* SPA-Net
receives when those 20 columns are non-zero: (a) which 20 dimensions were
selected, and (b) how those dimensions are preprocessed and interact with
optimization once training starts.

## Ranked follow-ups

### 1. All-128 sensitivity run (highest priority now)

Same architecture family, same 2M/400k cohort, but **all 128** frozen ParT
dimensions instead of the 20 selected by training-set variance, using
train-only standardization for all 128 (eps-clamped, same convention as
the active20 build). Before ZERO20 existed, an all-128 result would have
been hard to interpret — a null result could mean "width is the problem"
or "128 dimensions of frozen content don't help" ambiguously. **Now that
width is ruled out**, an all-128 result cleanly tests the
feature-selection hypothesis alone:

- If all-128 **recovers** toward native performance → the variance-based
  20-dimension selection specifically discarded the useful signal (some
  of the more discriminative raw dimensions were confirmed, in the
  original active20 diagnosis, to have been dropped by the variance
  filter) — the fix is a better selection rule, not abandoning frozen ParT
  features.
- If all-128 **still shows harm** despite having every available
  dimension → the problem is not which dimensions were dropped, and
  points more strongly at preprocessing/optimization interaction (below).

Existing joined-embedding artifacts already contain all 128 dimensions
per jet for the full 2M/400k cohort — no new embedding extraction would be
required, only a new standardization + training run. Estimated cost:
comparable single-GPU wall time to the ZERO20 run itself (each of these
runs has taken under two hours of fit time on an A100-class MIG slice);
the standardization/data-build step is the main additional overhead.

### 2. Usefulness-selected (not variance-selected) 20-dimension run

Cheaper and more targeted than all-128: re-select 20 dimensions by a
train-only *discriminative-power* criterion (e.g. signal-vs-background or
correct-vs-incorrect-jet-assignment separation, computed without touching
the validation cohort) instead of population variance, holding the input
width at 27 exactly as before. This isolates the feature-selection
hypothesis with the smallest possible architecture change — a positive
result here would be a stronger, more direct confirmation than all-128
that selection method (not just dimension count) was the issue.

### 3. Preprocessing / optimization-interaction probe

If neither of the above recovers performance, the remaining candidate is
that standardized frozen-embedding columns interact badly with this
training recipe regardless of which dimensions are chosen — e.g. a
learning-rate or initialization interaction specific to the newly-added
input block. A cheap first probe: a short learning-rate or warm-up
sensitivity scan on the ParT-active20 configuration specifically (not a
full retrain), checking whether the harm is sensitive to optimizer
settings that do not touch model capacity or feature content at all.

### 4. JP-JEPA integration — proceed, but not by copying the active20
recipe unchanged

The planned JP-JEPA representation-learning integration was designed to
follow the same "frozen embedding → variance-based dimension selection →
z-score → append to SPA-Net input" pattern documented for ParT. Given
ZERO20 rules out *width* as a harm mechanism but leaves the
*selection/preprocessing* questions open, JP-JEPA integration should
**not** reuse the variance-based selection step unchanged until follow-ups
1–2 above have been run for ParT — otherwise it risks reproducing whatever
selection-related harm mechanism those experiments are designed to find,
for a different embedding, for no new information. Reusing the join-key
and cohort-construction machinery (which ZERO20's own audit found no fault
with) remains appropriate.

## What is explicitly not proposed

- Repeating ZERO20 itself (its result is final and does not need a second
  seed to change the qualitative conclusion — the paired 95% CIs above are
  already far from the practical-effect floor in both directions).
- Abandoning frozen pretrained jet representations as a direction — the
  evidence points at a fixable integration detail, not a fundamental
  incompatibility.
- Any specific numeric authorization (GPU-hours, launch commands) — this
  page is a ranking, not a launch plan.
