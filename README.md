# WarzoneRobo

WarzoneRobo contains research code exploring reinforcement learning (RL) techniques for navigating grid based strategy maps. The project compares vanilla policy gradients with techniques that improve exploration or planning.

Unlike a typical navigation task with a goal location, the agent's objective here is long-term survival. Each step incurs cost and risk penalties based on the current maps, so policies must balance movement with staying alive.

## Project goals
* Train an agent using Proximal Policy Optimization (PPO).
* Augment the agent with Intrinsic Curiosity Module (ICM) and Random Network Distillation (RND) to encourage exploration.
* Combine RL with a symbolic planner to reduce search space and guide decision making.

## Getting started
The repository includes a Jupyter notebook named `demo.ipynb` that showcases the environment and training utilities. You can run the notebook interactively or execute its cells as a script.

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter and open the notebook
jupyter notebook demo.ipynb
```

Alternatively, you can run the notebook from the command line using `jupyter nbconvert`:

```bash
jupyter nbconvert --to notebook --execute demo.ipynb --output output.ipynb
```

### Command line training
You can also train the models directly with `train.py`. Hyperparameters can be
supplied via command line flags or a YAML configuration file:

```bash
python train.py --grid_size 8 --num_episodes 200
```

or

```bash
python train.py --config configs/default.yaml
```
To store generated figures run with `--plot-dir`:
```bash
python train.py --config configs/default.yaml --plot-dir figures
```

The repository includes `configs/default.yaml` as a starting configuration.
Duplicate and modify this file to experiment with different training settings.
Additional environment options can also be specified in the YAML:

```yaml
grid_size: 8
num_episodes: 200
dynamic_risk: true      # enemies increase risk over time
dynamic_cost: true     # cost near mines decays and rises dynamically
add_noise: true         # perturb loaded maps on reset
```

### Generating benchmark tables
After training, `train.py` evaluates each agent on the exported benchmark maps.
The metrics are saved to `results/benchmark_results.csv` and, when the output
path ends with `.html` or `.tex`, an additional formatted table is produced.

```bash
python train.py --num_episodes 200
# CSV and HTML tables are written to the `results/` folder
```

Use `--stat-test` to choose the statistical test applied to the aggregated
results. Available options are `paired`, `welch`, `mannwhitney` and `anova`. The
default is a paired t-test against the PPO baseline. P-values below `0.05`
are marked with `*` in the table and those below `0.01` with `**`.

## Components
* **PPO** – The main reinforcement learning algorithm used to learn policies from environment interaction.
* **ICM** – Adds intrinsic rewards based on prediction error of the agent's dynamics model to promote exploring unseen states.
* **RND** – Provides exploration bonuses by comparing a fixed random network with a trained predictor network.
* **Planner** – A symbolic planner computes heuristic paths that the agent can follow, helping integrate classical planning with learned policies.
* A decaying planner bonus starts high and linearly decreases each episode so the agent transitions from planner guidance to independent action.

The demo notebook experiments with different combinations of these components to evaluate their effect on success rate and exploration.

## Environment features
The grid world includes optional *dynamic risk* and *dynamic cost* modes. When enabled, hazard locations and traversal costs change over time so agents cannot memorize a single map. A small `survival_reward` of `0.05` is given each step. Benchmark maps can be exported with `export_benchmark_maps` and loaded later for evaluation. The environment's `render()` method returns RGB frames so that `render_episode_video` can produce GIFs of agent behavior.

## Success metric
An episode is marked as a **success** only when the agent survives for all
`max_steps` without hitting a mine or colliding with an enemy. Training logs store
these binary flags and plots display the running success rate next to the reward
curve. Result tables also report average success across seeds so improvements in
reward cannot mask agents that simply die early.

## Running Experiments
Train all models from a configuration file:
```bash
python train.py --config configs/default.yaml
```
Checkpoints are saved under `checkpoints/`, episode videos under `videos/`, and result tables under `results/`. GIF files use the active setting as a prefix, for example `baseline_ppo_only_0.gif`. Hyperparameters such as planner weights (`cost_weight`, `risk_weight`, etc.) can be edited in the YAML file or passed as command-line flags. The `seed` value in `configs/default.yaml` initializes both NumPy and PyTorch and turns on deterministic CuDNN settings so runs are reproducible.
Use `--plot-dir figures/` to save training plots such as reward curves and heatmaps. The directory is created automatically.

Specify `--initial-beta` and `--final-beta` to linearly decay the curiosity
weight. The value decreases until two thirds of the episodes have completed,
then stays at the final level. For example:

```bash
python train.py --initial-beta 0.2 --final-beta 0.05
```

To measure the effect of curiosity you can disable the ICM module:

```bash
python train.py --initial-beta 0.2 --final-beta 0.05 --disable_icm
```

To repeat an experiment with multiple random seeds you can loop over the `--seed` argument:

```bash
for s in 0 1 2 3 4; do
    python train.py --config configs/default.yaml --seed $s
done
```

## Running Tests
Execute the unit tests with:
```bash
pytest -q
```
The full suite runs in well under a minute on a CPU.

## Reproducing Our Results
The provided Dockerfile allows you to recreate the exact environment used for
our experiments. Run the following commands:

```bash
git clone <repo-url>
cd WarzoneRobo
docker build -t warzonerobo .
docker run --rm warzonerobo python train.py --config configs/default.yaml
```

Checkpoints are saved under `checkpoints/` and benchmark tables under
`results/` within the repository.
