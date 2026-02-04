from __future__ import annotations

import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .policy import ActorNet, batch_flatten_obs
from .critic import CriticNet
from .buffer import RolloutBuffer
from .action_assembler import assemble_actions


def compute_gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv = 0.0
    last_value = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * last_value * mask - values[t]
        last_adv = delta + gamma * lam * mask * last_adv
        advantages[t] = last_adv
        last_value = values[t]
    returns = advantages + values
    return advantages, returns


def train(env, cfg, log_dir: str, total_updates: int = 50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine observation and state dimensions
    obs_sample, _ = env.reset()
    obs_dim = batch_flatten_obs(list(obs_sample.values()), cfg).shape[1]
    global_state = env.get_global_state()
    state_dim = global_state.shape[0]

    actor = ActorNet(obs_dim, cfg).to(device)
    critic = CriticNet(state_dim, cfg).to(device)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    from ..utils.logging import MetricLogger
    from ..utils.progress import Progress

    metric_fields = [
        "episode_reward",
        "policy_loss",
        "value_loss",
        "entropy",
        "gu_queue_mean",
        "uav_queue_mean",
        "sat_queue_mean",
        "gu_queue_max",
        "uav_queue_max",
        "sat_queue_max",
        "gu_drop_sum",
        "uav_drop_sum",
        "sat_processed_sum",
        "sat_incoming_sum",
        "energy_mean",
        "update_time_sec",
        "rollout_time_sec",
        "optim_time_sec",
        "env_steps",
        "env_steps_per_sec",
        "update_steps_per_sec",
        "total_env_steps",
        "total_time_sec",
    ]
    logger = MetricLogger(log_dir, fieldnames=metric_fields)
    progress = Progress(total_updates, desc="Train")
    training_start = time.perf_counter()
    total_env_steps = 0
    best_ma = -float("inf")
    no_improve = 0
    reward_history = []

    obs, _ = env.reset()
    use_direct_logprob = (
        (not cfg.enable_bw_action)
        and cfg.fixed_satellite_strategy
        and not (cfg.energy_enabled and cfg.energy_safety_enabled)
    )

    for update in range(total_updates):
        update_start = time.perf_counter()
        buffer = RolloutBuffer(capacity=cfg.buffer_size)
        ep_reward = 0.0
        steps_count = 0
        gu_queue_sum = 0.0
        uav_queue_sum = 0.0
        sat_queue_sum = 0.0
        gu_queue_max = 0.0
        uav_queue_max = 0.0
        sat_queue_max = 0.0
        gu_drop_sum = 0.0
        uav_drop_sum = 0.0
        sat_processed_sum = 0.0
        sat_incoming_sum = 0.0
        energy_mean_sum = 0.0

        rollout_start = time.perf_counter()
        for step in range(cfg.buffer_size):
            obs_list = list(obs.values())
            obs_batch = batch_flatten_obs(obs_list, cfg)
            obs_tensor = torch.from_numpy(obs_batch).to(device)

            state_np = env.get_global_state()
            with torch.inference_mode():
                policy_out = actor.act(obs_tensor)
                value = critic(torch.from_numpy(state_np).to(device))

            accel_actions = policy_out.accel.cpu().numpy()
            bw_logits = policy_out.bw_logits.cpu().numpy() if policy_out.bw_logits is not None else None
            sat_logits = policy_out.sat_logits.cpu().numpy() if policy_out.sat_logits is not None else None

            # Apply action masks before execution/storage
            users_mask = np.stack([o["users_mask"] for o in obs_list], axis=0)
            sats_mask = np.stack([o["sats_mask"] for o in obs_list], axis=0)
            if cfg.doppler_enabled:
                nu_norm = np.stack([o["sats"][:, 6] for o in obs_list], axis=0)
                doppler_ok = (np.abs(nu_norm) <= 1.0).astype(np.float32)
                sats_mask = sats_mask * doppler_ok

            bw_exec = None
            if bw_logits is not None:
                bw_exec = bw_logits * users_mask
            elif cfg.enable_bw_action:
                bw_exec = np.zeros((len(obs_list), cfg.users_obs_max), dtype=np.float32)

            sat_exec = None
            if sat_logits is not None:
                sat_exec = sat_logits * sats_mask
            elif not cfg.fixed_satellite_strategy:
                sat_exec = np.zeros((len(obs_list), cfg.sats_obs_max), dtype=np.float32)

            action_dict = assemble_actions(cfg, env.agents, accel_actions, bw_logits=bw_exec, sat_logits=sat_exec)
            next_obs, rewards, terms, truncs, _ = env.step(action_dict)

            accel_exec = getattr(env, "last_exec_accel", accel_actions)
            if use_direct_logprob:
                accel_exec_norm = accel_actions
            else:
                accel_exec_norm = accel_exec / max(cfg.a_max, 1e-6)
                accel_exec_norm = np.clip(accel_exec_norm, -1.0, 1.0).astype(np.float32, copy=False)
            exec_parts = [accel_exec_norm]
            if cfg.enable_bw_action and bw_exec is not None:
                exec_parts.append(bw_exec)
            if not cfg.fixed_satellite_strategy and sat_exec is not None:
                exec_parts.append(sat_exec)
            action_vec_exec = np.concatenate(exec_parts, axis=1).astype(np.float32, copy=False)
            if use_direct_logprob:
                logprobs = policy_out.logprob.detach().cpu().numpy()
            else:
                with torch.inference_mode():
                    action_vec_exec_t = torch.from_numpy(action_vec_exec).to(device)
                    logprobs = (
                        actor.evaluate_actions(obs_tensor, action_vec_exec_t, out=policy_out.dist_out)[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )

            reward_scalar = list(rewards.values())[0]
            done_scalar = list(terms.values())[0] or list(truncs.values())[0]
            ep_reward += reward_scalar
            steps_count += 1
            total_env_steps += 1
            gu_queue_sum += float(np.mean(env.gu_queue))
            uav_queue_sum += float(np.mean(env.uav_queue))
            sat_queue_sum += float(np.mean(env.sat_queue))
            gu_queue_max = max(gu_queue_max, float(np.max(env.gu_queue)))
            uav_queue_max = max(uav_queue_max, float(np.max(env.uav_queue)))
            sat_queue_max = max(sat_queue_max, float(np.max(env.sat_queue)))
            gu_drop_sum += float(np.sum(env.gu_drop))
            uav_drop_sum += float(np.sum(env.uav_drop))
            if hasattr(env, "last_sat_processed"):
                sat_processed_sum += float(np.sum(env.last_sat_processed))
            if hasattr(env, "last_sat_incoming"):
                sat_incoming_sum += float(np.sum(env.last_sat_incoming))
            if cfg.energy_enabled:
                energy_mean_sum += float(np.mean(env.uav_energy))

            buffer.add(
                obs_batch,
                action_vec_exec,
                logprobs,
                reward_scalar,
                float(value.item()),
                done_scalar,
                state_np,
            )

            obs = next_obs
            if done_scalar:
                obs, _ = env.reset()
        rollout_time = time.perf_counter() - rollout_start

        # Prepare batches
        obs_arr, act_arr, logp_arr, rewards, values, dones, state_arr = buffer.as_arrays()
        adv, rets = compute_gae(rewards, values, dones, cfg.gamma, cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        T, N, _ = obs_arr.shape
        obs_flat = obs_arr.reshape(T * N, -1)
        act_flat = act_arr.reshape(T * N, -1)
        logp_flat = logp_arr.reshape(T * N)
        adv_flat = np.repeat(adv, N)

        # Convert to torch
        obs_flat_t = torch.from_numpy(obs_flat).to(device)
        act_flat_t = torch.from_numpy(act_flat).to(device)
        logp_flat_t = torch.from_numpy(logp_flat).to(device)
        adv_flat_t = torch.from_numpy(adv_flat).to(device)
        state_t = torch.from_numpy(state_arr).to(device)
        ret_t = torch.from_numpy(rets).to(device)

        batch_size = len(obs_flat)
        minibatch_size = max(1, batch_size // cfg.num_mini_batch)
        indices = np.arange(batch_size)

        policy_losses = []
        value_losses = []
        entropies = []

        optim_start = time.perf_counter()
        for _ in range(cfg.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                new_logp, entropy = actor.evaluate_actions(obs_flat_t[mb_idx], act_flat_t[mb_idx])

                ratio = torch.exp(new_logp - logp_flat_t[mb_idx])
                surr1 = ratio * adv_flat_t[mb_idx]
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv_flat_t[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                actor_optim.zero_grad()
                (policy_loss - cfg.entropy_coef * entropy.mean()).backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                actor_optim.step()

                policy_losses.append(policy_loss.item())
                entropies.append(entropy.mean().item())

            # critic update (full batch for simplicity)
            value_pred = critic(state_t)
            value_loss = F.mse_loss(value_pred, ret_t)
            critic_optim.zero_grad()
            (cfg.value_coef * value_loss).backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
            critic_optim.step()
            value_losses.append(value_loss.item())
        optim_time = time.perf_counter() - optim_start

        update_time = time.perf_counter() - update_start
        steps_count = max(1, steps_count)
        episode_reward = ep_reward / steps_count
        metrics = {
            "episode_reward": episode_reward,
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "gu_queue_mean": gu_queue_sum / steps_count,
            "uav_queue_mean": uav_queue_sum / steps_count,
            "sat_queue_mean": sat_queue_sum / steps_count,
            "gu_queue_max": gu_queue_max,
            "uav_queue_max": uav_queue_max,
            "sat_queue_max": sat_queue_max,
            "gu_drop_sum": gu_drop_sum,
            "uav_drop_sum": uav_drop_sum,
            "sat_processed_sum": sat_processed_sum,
            "sat_incoming_sum": sat_incoming_sum,
            "energy_mean": (energy_mean_sum / steps_count) if cfg.energy_enabled else 0.0,
            "update_time_sec": update_time,
            "rollout_time_sec": rollout_time,
            "optim_time_sec": optim_time,
            "env_steps": float(steps_count),
            "env_steps_per_sec": steps_count / max(1e-9, rollout_time),
            "update_steps_per_sec": steps_count / max(1e-9, update_time),
            "total_env_steps": float(total_env_steps),
            "total_time_sec": time.perf_counter() - training_start,
        }

        logger.log(
            update,
            metrics,
        )
        progress.update(update + 1)

        if cfg.early_stop_enabled:
            reward_history.append(episode_reward)
            if update + 1 >= cfg.early_stop_min_updates and len(reward_history) >= cfg.early_stop_window:
                ma = float(np.mean(reward_history[-cfg.early_stop_window :]))
                if ma > best_ma + cfg.early_stop_min_delta:
                    best_ma = ma
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= cfg.early_stop_patience:
                    print(
                        f"Early stopping at update {update + 1}: "
                        f"moving average={ma:.6f}, best={best_ma:.6f}"
                    )
                    break

    # Save checkpoints
    os.makedirs(log_dir, exist_ok=True)
    torch.save(actor.state_dict(), os.path.join(log_dir, "actor.pt"))
    torch.save(critic.state_dict(), os.path.join(log_dir, "critic.pt"))

    progress.close()
    logger.close()
    return actor, critic
