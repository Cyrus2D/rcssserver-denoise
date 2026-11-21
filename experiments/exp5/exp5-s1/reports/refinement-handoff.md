# Refinement handoff: position forecasts during visual gaps

## Candidate designated before new-game scoring

Practical objective: near50_risk. Its exact frozen model, selector, threshold,
blend and cap settings are in [the frozen policy](refinement-frozen-policies.json).
Compare it with the separate mean-focused and hard-cap policies in
[the report](report.html#refinement-final). Do not add gains from different stages.

- Mean candidate: continued GRU128, followed by a severity-weighted boosted selector.
- Practical candidate: WM-centered residual GRU128, harm-penalty1 fine-tuning,
  followed by a severity-weighted boosted selector.
- Stricter empirical harm budgets: harm-penalty3 GRU128 plus selector.
- Fixed safety ceilings are additional, explicitly separate policies.

## Simple engineering alternatives

[Exact decision trees](refinement-pruned-trees.json) and the additional section17
comparison retain two choices: continued GRU128 + seven-node tree for mean error;
WM-centered GRU128 + eleven-node tree for a practical risk tradeoff. These
additional tree checks reuse the confirmation batch exploratorily; fitting and
threshold selection use only the original validation games. They do not replace
the preregistered primary policy comparison. Equal-decision subtree pruning
preserves every decision and matched500k calibration rows per export. A compact tree is fixed to its exported threshold; regenerate it if that threshold changes.

## Inputs, outputs and coordinates

Every neural candidate receives the original51 observation features in the exact
manifest order, a categorical uniform-number embedding, and32 current/past-cycle
steps. It outputs the current target position as displacement from its last
confirmed named-sighting anchor. It does not predict velocity, body or actions.

The residual candidate adds the existing WM offset to the original point readout.
Its zero-initialized head starts by preserving WM. Copy prepare.py normalization,
masks and coordinate transforms exactly; do not mix absolute and anchor-relative
positions. Runtime selector features are defined in refinement_select.features.
The selector sees the raw model proposal, before any optional safety ceiling.

A score such as0.8 from the weighted classifier is not80% positional confidence
or80% probability of beating WM. The weighting encodes sizes of gains and losses.

## Eligibility and reset behavior to preserve in a future integration

- This experiment uses unquantized visual measurements, ordinary finite adaptive
  views, no hearing payloads and no distance fading. It is not the Exp3 quantization
  denoiser and is not validated for other observation configurations.
- Known-number target, valid WM position, confirmed named-sighting history, age1–30.
- Never replace a posCount0 WM update. Preserve fresh confirmed observations.
- Reset history on PlayOn boundaries, stopped cycles and time discontinuities.
- The model was trained with a rolling32-step window and zero initial hidden state
  for each inference, with left padding after resets. An indefinitely carried GRU
  hidden state is a different estimator and must not be substituted silently.
- Keep the unmodified WM as the model input/reference stream. Feeding corrected
  outputs back into the features changes the training distribution.
- Truth, true distance and anonymous-target truth association are evaluation
  labels only. The practical objective uses estimated observer-target distance.
- Additional caps limit movement from WM; they bound added current distance error,
  not final error or action quality.

## Evidence and limits

The original20 test games were already inspected and are exploratory. Ten new
full-length baseline games use fresh fixed seeds, with frozen policies and no
threshold retuning. All22 observers per game finalized, hearing payload count is
zero, fullstate is reference-only, and the original12 key server settings match.
New-game results confirm trajectories from the same two baseline teams; agents
do not use the correction to make decisions in these games.

A future live integration must separately check state consistency, action-cache
routing, decision time and closed-loop feedback. Offline error reduction is not
a measured match-win benefit.

## Artifacts

Checkpoints, selector pickles, raw predictions and TensorBoard logs:
logs/exp5-s1/refinement. Independent games:
logs/exp5-s1/refinement-fresh; their evaluation-only dataset:
logs/exp5-s1/refinement-fresh-dataset. Small result snapshots and source/checkpoint
fingerprints are retained beside this report. The reproducible entry point is
code/run_refinement.py. New-game results must not become another tuning set.
