import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .icm import ICMModule
from .planner import SymbolicPlanner
from .pseudocount import PseudoCountExploration
from .rnd import RNDModule


class PPOPolicy(nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.actor(x), self.critic(x)

    def act(self, x: torch.Tensor):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value


def compute_gae(
    rewards: list[float],
    values: list[float],
    gamma: float = 0.99,
    lam: float = 0.95,
):
    """Compute generalized advantages for one completed episode."""
    advantages: list[float] = []
    gae = 0.0
    bootstrap_values = values + [0.0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * bootstrap_values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages


def _linear_schedule(start: float, end: float, progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return float(start + (end - start) * progress)


def train_agent(
    env,
    policy: PPOPolicy,
    icm: ICMModule,
    planner: SymbolicPlanner,
    optimizer_policy: torch.optim.Optimizer,
    optimizer_icm: torch.optim.Optimizer | None,
    use_icm=True,
    use_planner=True,
    num_episodes: int = 500,
    beta: float = 0.1,
    final_beta: float | None = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    planner_weights: dict | None = None,
    rnd: RNDModule | None = None,
    pseudo: PseudoCountExploration | None = None,
    seed: int = 42,
    add_noise: bool = False,
    logger=None,
    initial_bonus: float = 0.5,
    final_planner_bonus: float = 0.1,
    reset_env: bool = True,
    training_map_path: str = "maps/map_00.npz",
    training_map_paths: list[str] | None = None,
    episode_records: list[dict] | None = None,
):
    """Train PPO with optional intrinsic reward and risk-gated planning.

    The planner is advisory: PPO always samples the executed action.  When
    local risk is elevated, the planner recommends an action and the learning
    reward receives a decaying bonus if the PPO action agrees with that
    recommendation and reaches a lower-risk cell.  This preserves an on-policy
    PPO update while still providing explicit planner guidance.

    Environment reward is logged without intrinsic or planner shaping.  The
    latter bonuses affect only the learning return, so reward comparisons
    between variants remain meaningful.
    """
    if use_icm is True and optimizer_icm is None:
        raise ValueError("ICM training requires an optimizer over ICM parameters")
    if use_icm == "rnd" and (rnd is None or optimizer_icm is None):
        raise ValueError("RND training requires a module and predictor optimizer")
    if use_icm == "pseudo" and pseudo is None:
        raise ValueError("Pseudo-count training requires a pseudo-count module")

    reward_log = []
    paths_log = []
    planner_usage = []
    ppo_usage = []
    intrinsic_rewards = []
    extrinsic_rewards = []
    step_counts = []
    success_flags = []
    planner_usage_rate = []

    initial_bonus = max(0.0, float(initial_bonus))
    final_planner_bonus = max(0.0, float(final_planner_bonus))

    if training_map_paths is None:
        os.makedirs(os.path.dirname(training_map_path) or ".", exist_ok=True)
        if reset_env:
            env.reset(seed=seed)
            env.save_map(training_map_path)
        episode_maps = [training_map_path]
    else:
        if not training_map_paths:
            raise ValueError("training_map_paths must contain at least one map")
        missing_maps = [path for path in training_map_paths if not os.path.exists(path)]
        if missing_maps:
            raise FileNotFoundError(
                "Training maps do not exist: " + ", ".join(missing_maps)
            )
        episode_maps = list(training_map_paths)

    for episode in range(num_episodes):
        # Rotate through the complete generated training set.  The seed-based
        # offset and episode RNG are deterministic, so every variant sees the
        # same map and enemy-trajectory sequence for a given run seed.
        episode_seed = seed * 1_000_003 + episode
        episode_map = episode_maps[(episode + seed) % len(episode_maps)]
        torch.manual_seed(episode_seed)
        obs, reset_info = env.reset(
            seed=episode_seed,
            load_map_path=episode_map,
            add_noise=add_noise,
        )

        done = False
        total_ext_reward = 0.0
        step_count = 0
        info = {}
        terrain_decay = max(0.1, 1 - (episode / 1000))
        obs_buf = []
        action_buf = []
        logprob_buf = []
        value_buf = []
        reward_buf = []
        agent_path = []
        planner_consultations = 0
        ppo_decisions = 0

        if planner_weights is not None:
            planner = SymbolicPlanner(
                env.cost_map,
                env.risk_map,
                env.np_random,
                cost_weight=planner_weights.get("cost_weight", 2.0),
                risk_weight=planner_weights.get("risk_weight", 3.0),
                revisit_penalty=planner_weights.get("revisit_penalty", 1.0),
            )
        else:
            planner.cost_map = env.cost_map
            planner.risk_map = env.risk_map
            planner.np_random = env.np_random
            planner.grid_size = env.grid_size
            planner.reset()

        intrinsic_log = 0.0
        visit_count = np.zeros((env.grid_size, env.grid_size))
        episode_progress = episode / max(num_episodes - 1, 1)
        planner_bonus = _linear_schedule(
            initial_bonus,
            final_planner_bonus,
            episode_progress,
        )

        if final_beta is not None:
            decay_end = max(1, int(num_episodes * 2 / 3))
            beta_progress = min(episode, decay_end) / decay_end
            beta_value = _linear_schedule(beta, final_beta, beta_progress)
        else:
            beta_value = float(beta)

        while not done:
            state = np.asarray(obs, dtype=np.float32).copy()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            risk = env.risk_map[env.agent_pos[0]][env.agent_pos[1]]

            if risk > 0.7:
                planner.risk_weight = 5.0
                planner.cost_weight = 1.0
                use_planner_now = use_planner
            elif risk > 0.5:
                planner.risk_weight = 3.0
                planner.cost_weight = 2.0
                use_planner_now = use_planner
            elif risk > 0.2:
                planner.risk_weight = 1.0
                planner.cost_weight = 3.0
                use_planner_now = use_planner
            else:
                use_planner_now = False

            planner_action = None
            if use_planner_now:
                planner_action = int(planner.get_safe_subgoal(env.agent_pos))
                planner_consultations += 1

            # PPO remains the behavior policy in every state.  The planner
            # changes the learning reward, never the executed action.
            action, old_logprob, value = policy.act(state_tensor)
            old_logprob = old_logprob.detach().reshape(())
            ppo_decisions += 1

            next_obs, ext_reward, done, _, info = env.step(
                action,
                terrain_decay=terrain_decay,
            )
            x, y = env.agent_pos
            visit_count[x][y] += 1
            count_reward = 1.0 / np.sqrt(visit_count[x][y])

            planner_shaping = 0.0
            if (
                planner_action is not None
                and action == planner_action
                and env.risk_map[x][y] < 0.5
            ):
                planner_shaping = planner_bonus

            next_tensor = torch.tensor(
                next_obs,
                dtype=torch.float32,
            ).unsqueeze(0)
            action_tensor = torch.tensor([action])

            if use_icm == "count":
                intrinsic_reward = float(count_reward)
            elif use_icm == "rnd":
                r_int, prediction, target = rnd(state_tensor)
                intrinsic_reward = float(r_int.item())
                rnd_loss = F.mse_loss(prediction, target.detach())
                optimizer_icm.zero_grad()
                rnd_loss.backward()
                optimizer_icm.step()
            elif use_icm == "pseudo":
                intrinsic_reward = float(pseudo.bonus(state))
            elif use_icm is True:
                curiosity, forward_loss, inverse_loss = icm(
                    state_tensor,
                    next_tensor,
                    action_tensor,
                )
                intrinsic_reward = float(curiosity.item())
                icm_loss = forward_loss + inverse_loss
                optimizer_icm.zero_grad()
                icm_loss.backward()
                optimizer_icm.step()
            else:
                intrinsic_reward = 0.0

            learning_reward = (
                float(ext_reward) + planner_shaping + beta_value * intrinsic_reward
            )
            intrinsic_log += intrinsic_reward
            total_ext_reward += float(ext_reward)
            step_count += 1
            agent_path.append(tuple(env.agent_pos))

            # Store the state at which the action was selected.  The previous
            # implementation stored next_obs here, shifting every transition.
            obs_buf.append(state)
            action_buf.append(action)
            logprob_buf.append(old_logprob)
            value_buf.append(float(value.item()))
            reward_buf.append(learning_reward)
            obs = next_obs

        advantages = np.asarray(
            compute_gae(
                reward_buf,
                value_buf,
                gamma=gamma,
                lam=gae_lambda,
            ),
            dtype=np.float32,
        )
        returns = advantages + np.asarray(value_buf, dtype=np.float32)

        advantage_tensor = torch.tensor(advantages, dtype=torch.float32)
        advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (
            advantage_tensor.std(unbiased=False) + 1e-8
        )

        obs_tensor = torch.tensor(np.stack(obs_buf), dtype=torch.float32)
        action_tensor = torch.tensor(action_buf, dtype=torch.long)
        old_logprob_tensor = torch.stack(logprob_buf).detach()
        return_tensor = torch.tensor(returns, dtype=torch.float32)

        logits, new_values = policy(obs_tensor)
        distribution = torch.distributions.Categorical(logits=logits)
        new_logprob = distribution.log_prob(action_tensor)

        ratio = torch.exp(new_logprob - old_logprob_tensor)
        surrogate_1 = ratio * advantage_tensor
        surrogate_2 = (
            torch.clamp(
                ratio,
                1 - clip_epsilon,
                1 + clip_epsilon,
            )
            * advantage_tensor
        )
        policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
        entropy_loss = -entropy_coef * distribution.entropy().mean()

        value_loss = F.mse_loss(new_values.squeeze(-1), return_tensor)
        total_loss = policy_loss + value_coef * value_loss + entropy_loss
        optimizer_policy.zero_grad()
        total_loss.backward()
        optimizer_policy.step()

        if episode % 50 == 0:
            paths_log.append(agent_path)

        reward_log.append(total_ext_reward)
        planner_usage.append(planner_consultations)
        ppo_usage.append(ppo_decisions)
        intrinsic_rewards.append(intrinsic_log)
        extrinsic_rewards.append(total_ext_reward)
        step_counts.append(step_count)
        survived = step_count >= env.max_steps and not info.get("dead", False)
        success_flags.append(1 if survived else 0)
        planner_percent = (
            planner_consultations / ppo_decisions if ppo_decisions else 0.0
        )
        planner_usage_rate.append(planner_percent)
        if episode_records is not None:
            episode_records.append(
                {
                    "Episode": episode + 1,
                    "Training Map": episode_map,
                    "Training Map Index": (episode + seed) % len(episode_maps),
                    "Environment Seed": episode_seed,
                    "Start Row": reset_info["start_position"][0],
                    "Start Column": reset_info["start_position"][1],
                    "Environment Reward": total_ext_reward,
                    "Intrinsic Reward": intrinsic_log,
                    "Steps": step_count,
                    "PPO Decisions": ppo_decisions,
                    "Planner Consultations": planner_consultations,
                    "Planner Consultation Rate": planner_percent,
                    "Success": success_flags[-1],
                }
            )

        success_rate = float(np.mean(success_flags))
        if logger is not None:
            if hasattr(logger, "add_scalar"):
                logger.add_scalar("reward/extrinsic", total_ext_reward, episode)
                logger.add_scalar("reward/intrinsic", intrinsic_log, episode)
                logger.add_scalar("success_rate", success_rate, episode)
                logger.add_scalar("planner_usage", planner_percent, episode)
            elif hasattr(logger, "log"):
                logger.log(
                    {
                        "extrinsic_reward": total_ext_reward,
                        "intrinsic_reward": intrinsic_log,
                        "success_rate": success_rate,
                        "planner_usage": planner_percent,
                        "episode": episode,
                    }
                )

        print(
            f"Episode {episode + 1:03d}/{num_episodes} | "
            f"Reward: {total_ext_reward:.2f} | "
            f"Steps: {step_count} | "
            f"PPO: {ppo_decisions} | "
            f"Planner consulted: {planner_consultations} | "
            f"Success: {success_rate * 100:.1f}%"
        )

    return (
        reward_log,
        intrinsic_rewards,
        extrinsic_rewards,
        planner_usage,
        paths_log,
        step_counts,
        success_flags,
        planner_usage_rate,
    )
