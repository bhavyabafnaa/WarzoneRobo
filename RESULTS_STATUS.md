# Results validity status

## Current boundary

The pre-fix CSV and HTML result tables were removed from the corrected branch;
they remain recoverable in Git history for traceability. Other generated
artifacts committed before the experiment-validity correction are not valid
evidence for the paper and must not be cited as corrected results. This
includes the existing files under `figures/`, `videos/` and `checkpoints/`.

## Why the earlier run is invalidated

The corrected audit found that the previous pipeline:

1. min-max normalized one ICM transition at a time, producing zero curiosity;
2. passed the policy optimizer to the ICM loss, leaving ICM weights unchanged;
3. stored `next_state` against the action selected from `state`;
4. treated deterministic planner actions as though PPO had sampled them;
5. included planner shaping in the reported environment reward;
6. trained the value function against immediate normalized rewards instead of returns;
7. used process-randomized hashes for pseudo-count buckets;
8. allowed a stale benchmark value to be recorded when RND was disabled; and
9. did not use matched deterministic benchmark rollouts.

Because these issues affect both method identity and measured outcomes, the old
means, standard deviations and p-values are provisional historical values—not
results from the corrected algorithm.

## Corrected five-seed rerun

The corrected six-variant matrix over seeds 0–4 has been completed. PPO + ICM
+ Planner was numerically highest in training reward, but its paired advantage
over PPO was small and not statistically significant. Held-out success did not
improve, and all greedy policies selected the same constant action. Therefore
the corrected run supports neither hybrid superiority nor a useful generalized
policy.

## Confirmatory regeneration

The next experiment is declared in `configs/confirmatory.yaml`:

```bash
python train.py --config configs/confirmatory.yaml --validate-only
python train.py --config configs/confirmatory.yaml
```

It retains pure survival as the task and introduces random safe starts, 30/10/30
disjoint train/validation/test maps, 40 matched training seeds, 500 episodes,
and 20 matched stochastic rollouts per held-out map. Greedy rollouts are
secondary diagnostics.

The primary outcome is stochastic held-out survival rate. The declared hybrid
must exceed every other variant by at least 0.05 absolute survival probability,
with paired significance after Holm correction. Reward comparisons and the
ICM-by-planner interaction are secondary analyses. A pre-training map gate
rejects constant-action-solvable maps and map sets on which a safe non-learning
heuristic cannot survive.

Only the newly generated per-seed records and summaries may be transferred to
the manuscript. The 40-seed outcomes must not be used to tune this declared
protocol; any later changes require a newly versioned protocol and independent
test maps.
