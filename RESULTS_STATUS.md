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

## Required regeneration

Run the corrected six-variant matrix over seeds 0–4:

```bash
python train.py --config configs/default.yaml --seeds 0 1 2 3 4
```

Only the newly generated per-seed records and summaries may be transferred to
the manuscript. The primary confirmatory comparison should be declared before
inspection; for the current paper that comparison is PPO versus PPO + ICM +
Planner. Other variant comparisons should be described as ablations unless a
multiple-comparison procedure is selected.
