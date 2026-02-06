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
from .baselines import queue_aware_policy
from ..utils.normalization import RunningMeanStd


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


def train(
    env,
    cfg,
    log_dir: str,
    total_updates: int = 50,
    init_actor_path: str | None = None,
    init_critic_path: str | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine observation and state dimensions
    obs_sample, _ = env.reset()
    obs_dim = batch_flatten_obs(list(obs_sample.values()), cfg).shape[1]
    global_state = env.get_global_state()
    state_dim = global_state.shape[0]

    actor = ActorNet(obs_dim, cfg).to(device)
    critic = CriticNet(state_dim, cfg).to(device)
    if init_actor_path:
        try:
            state = torch.load(init_actor_path, map_location=device)
            actor.load_state_dict(state, strict=False)
            print(f"Loaded actor init from {init_actor_path} (strict=False)")
        except Exception as exc:
            print(f"Warning: failed to load actor init from {init_actor_path}: {exc}")
    if init_critic_path:
        try:
            state = torch.load(init_critic_path, map_location=device)
            critic.load_state_dict(state, strict=False)
            print(f"Loaded critic init from {init_critic_path} (strict=False)")
        except Exception as exc:
            print(f"Warning: failed to load critic init from {init_critic_path}: {exc}")

    actor_optim = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    from ..utils.logging import MetricLogger
    from ..utils.progress import Progress

    metric_fields = [
        "episode_reward",
        "policy_loss",
        "value_loss",
        "entropy",
        "imitation_loss",
        "r_service_ratio",
        "r_drop_ratio",
        "r_queue_pen",
        "r_queue_topk",
        "r_centroid",
        "centroid_dist_mean",
        "r_bw_align",
        "r_sat_score",
        "r_assoc_ratio",
        "r_queue_delta",
        "r_dist",
        "r_dist_delta",
        "r_energy",
        "r_fail_penalty",
        "r_term_service",
        "r_term_drop",
        "r_term_queue",
        "r_term_topk",
        "r_term_assoc",
        "r_term_q_delta",
        "r_term_dist",
        "r_term_dist_delta",
        "r_term_centroid",
        "r_term_bw_align",
        "r_term_sat_score",
        "r_term_energy",
        "reward_raw",
        "arrival_sum",
        "outflow_sum",
        "service_norm",
        "drop_norm",
        "queue_total",
        "queue_total_active",
        "arrival_rate_eff",
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
    reward_rms = RunningMeanStd() if getattr(cfg, "reward_norm_enabled", False) else None
    imitation_coef = float(getattr(cfg, "imitation_coef", 0.0) or 0.0)
    imitation_enabled = bool(getattr(cfg, "imitation_enabled", False)) and imitation_coef > 0.0
    imitation_accel = bool(getattr(cfg, "imitation_accel", True))
    imitation_bw = bool(getattr(cfg, "imitation_bw", True))
    imitation_sat = bool(getattr(cfg, "imitation_sat", False))
    bw_scale = float(getattr(cfg, "bw_logit_scale", 1.0) or 1.0)
    sat_scale = float(getattr(cfg, "sat_logit_scale", 1.0) or 1.0)

    obs, _ = env.reset()
    use_direct_logprob = (
        (not cfg.enable_bw_action)
        and cfg.fixed_satellite_strategy
        and not (cfg.energy_enabled and cfg.energy_safety_enabled)
    )

    def _build_imitation_target(obs_list):
        if not imitation_enabled:
            return None
        base_accel, base_bw, base_sat = queue_aware_policy(obs_list, cfg)
        parts = []
        if imitation_accel:
            parts.append(base_accel)
        else:
            parts.append(np.zeros_like(base_accel))
        if cfg.enable_bw_action:
            if imitation_bw:
                parts.append(base_bw)
            else:
                parts.append(np.zeros_like(base_bw))
        if not cfg.fixed_satellite_strategy:
            if imitation_sat:
                parts.append(base_sat)
            else:
                parts.append(np.zeros_like(base_sat))
        return np.concatenate(parts, axis=1).astype(np.float32, copy=False)

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
        r_service_ratio_sum = 0.0
        r_drop_ratio_sum = 0.0
        r_queue_pen_sum = 0.0
        r_queue_topk_sum = 0.0
        r_centroid_sum = 0.0
        centroid_dist_mean_sum = 0.0
        r_bw_align_sum = 0.0
        r_sat_score_sum = 0.0
        r_assoc_ratio_sum = 0.0
        r_queue_delta_sum = 0.0
        r_dist_sum = 0.0
        r_dist_delta_sum = 0.0
        r_energy_sum = 0.0
        r_fail_penalty_sum = 0.0
        r_term_service_sum = 0.0
        r_term_drop_sum = 0.0
        r_term_queue_sum = 0.0
        r_term_topk_sum = 0.0
        r_term_assoc_sum = 0.0
        r_term_q_delta_sum = 0.0
        r_term_dist_sum = 0.0
        r_term_dist_delta_sum = 0.0
        r_term_centroid_sum = 0.0
        r_term_bw_align_sum = 0.0
        r_term_sat_score_sum = 0.0
        r_term_energy_sum = 0.0
        imitation_loss_sum = 0.0
        reward_raw_sum = 0.0
        arrival_sum_sum = 0.0
        outflow_sum_sum = 0.0
        service_norm_sum = 0.0
        drop_norm_sum = 0.0
        queue_total_sum = 0.0
        queue_total_active_sum = 0.0
        arrival_rate_eff_sum = 0.0

        rollout_start = time.perf_counter()
        for step in range(cfg.buffer_size):
            obs_list = list(obs.values())
            obs_batch = batch_flatten_obs(obs_list, cfg)
            if not np.isfinite(obs_batch).all():
                print(f"NaN/Inf detected in obs_batch at update={update}, step={step}")
                raise ValueError("obs_batch contains NaN/Inf")
            obs_tensor = torch.from_numpy(obs_batch).to(device)

            state_np = env.get_global_state()
            with torch.inference_mode():
                policy_out = actor.act(obs_tensor)
                value = critic(torch.from_numpy(state_np).to(device))
            if not torch.isfinite(policy_out.dist_out["mu"]).all():
                print(f"NaN/Inf detected in actor mu at update={update}, step={step}")
                raise ValueError("actor mu contains NaN/Inf")

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
            parts = getattr(env, "last_reward_parts", None)
            if parts:
                r_service_ratio_sum += float(parts.get("service_ratio", 0.0))
                r_drop_ratio_sum += float(parts.get("drop_ratio", 0.0))
                arrival_sum_sum += float(parts.get("arrival_sum", 0.0))
                outflow_sum_sum += float(parts.get("outflow_sum", 0.0))
                service_norm_sum += float(parts.get("service_norm", 0.0))
                drop_norm_sum += float(parts.get("drop_norm", 0.0))
                queue_total_sum += float(parts.get("queue_total", 0.0))
                queue_total_active_sum += float(parts.get("queue_total_active", 0.0))
                arrival_rate_eff_sum += float(parts.get("arrival_rate_eff", 0.0))
                r_queue_pen_sum += float(parts.get("queue_pen", 0.0))
                r_queue_topk_sum += float(parts.get("queue_topk", 0.0))
                r_centroid_sum += float(parts.get("centroid_reward", 0.0))
                centroid_dist_mean_sum += float(parts.get("centroid_dist_mean", 0.0))
                r_bw_align_sum += float(parts.get("bw_align", 0.0))
                r_sat_score_sum += float(parts.get("sat_score", 0.0))
                r_assoc_ratio_sum += float(parts.get("assoc_ratio", 0.0))
                r_queue_delta_sum += float(parts.get("queue_delta", 0.0))
                r_dist_sum += float(parts.get("dist_reward", 0.0))
                r_dist_delta_sum += float(parts.get("dist_delta", 0.0))
                r_energy_sum += float(parts.get("energy_reward", 0.0))
                r_fail_penalty_sum += float(parts.get("fail_penalty", 0.0))
                r_term_service_sum += float(parts.get("term_service", 0.0))
                r_term_drop_sum += float(parts.get("term_drop", 0.0))
                r_term_queue_sum += float(parts.get("term_queue", 0.0))
                r_term_topk_sum += float(parts.get("term_topk", 0.0))
                r_term_assoc_sum += float(parts.get("term_assoc", 0.0))
                r_term_q_delta_sum += float(parts.get("term_q_delta", 0.0))
                r_term_dist_sum += float(parts.get("term_dist", 0.0))
                r_term_dist_delta_sum += float(parts.get("term_dist_delta", 0.0))
                r_term_centroid_sum += float(parts.get("term_centroid", 0.0))
                r_term_bw_align_sum += float(parts.get("term_bw_align", 0.0))
                r_term_sat_score_sum += float(parts.get("term_sat_score", 0.0))
                r_term_energy_sum += float(parts.get("term_energy", 0.0))
                reward_raw_sum += float(parts.get("reward_raw", 0.0))

            buffer.add(
                obs_batch,
                action_vec_exec,
                logprobs,
                reward_scalar,
                float(value.item()),
                done_scalar,
                state_np,
                _build_imitation_target(obs_list),
            )

            obs = next_obs
            if done_scalar:
                obs, _ = env.reset()
        rollout_time = time.perf_counter() - rollout_start

        # Prepare batches
        obs_arr, act_arr, logp_arr, rewards, values, dones, state_arr, imitation_arr = buffer.as_arrays()
        if getattr(cfg, "reward_norm_enabled", False) and reward_rms is not None:
            reward_rms.update(rewards)
            rewards = (rewards - reward_rms.mean) / (np.sqrt(reward_rms.var) + 1e-8)
            clip_val = float(getattr(cfg, "reward_norm_clip", 0.0) or 0.0)
            if clip_val > 0:
                rewards = np.clip(rewards, -clip_val, clip_val)
        adv, rets = compute_gae(rewards, values, dones, cfg.gamma, cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = np.clip(adv, -5.0, 5.0)

        T, N, _ = obs_arr.shape
        obs_flat = obs_arr.reshape(T * N, -1)
        act_flat = act_arr.reshape(T * N, -1)
        logp_flat = logp_arr.reshape(T * N)
        imitation_flat = imitation_arr.reshape(T * N, -1)
        adv_flat = np.repeat(adv, N)

        # Convert to torch
        obs_flat_t = torch.from_numpy(obs_flat).to(device)
        act_flat_t = torch.from_numpy(act_flat).to(device)
        logp_flat_t = torch.from_numpy(logp_flat).to(device)
        adv_flat_t = torch.from_numpy(adv_flat).to(device)
        state_t = torch.from_numpy(state_arr).to(device)
        ret_t = torch.from_numpy(rets).to(device)
        imitation_flat_t = torch.from_numpy(imitation_flat).to(device)
        if not torch.isfinite(obs_flat_t).all():
            print(f"NaN/Inf detected in obs_flat_t at update={update}")
            raise ValueError("obs_flat_t contains NaN/Inf")
        if not torch.isfinite(act_flat_t).all():
            print(f"NaN/Inf detected in act_flat_t at update={update}")
            raise ValueError("act_flat_t contains NaN/Inf")
        if not torch.isfinite(logp_flat_t).all():
            print(f"NaN/Inf detected in logp_flat_t at update={update}")
            raise ValueError("logp_flat_t contains NaN/Inf")
        if not torch.isfinite(adv_flat_t).all():
            print(f"NaN/Inf detected in adv_flat_t at update={update}")
            raise ValueError("adv_flat_t contains NaN/Inf")
        if not torch.isfinite(ret_t).all():
            print(f"NaN/Inf detected in ret_t at update={update}")
            raise ValueError("ret_t contains NaN/Inf")
        for name, param in actor.named_parameters():
            if not torch.isfinite(param).all():
                print(f"NaN/Inf detected in actor param {name} before PPO update at update={update}")
                raise ValueError("actor parameters contain NaN/Inf before PPO update")

        batch_size = len(obs_flat)
        minibatch_size = max(1, batch_size // cfg.num_mini_batch)
        indices = np.arange(batch_size)
        imitation_mask = None
        if imitation_enabled:
            mask_parts = []
            mask_parts.append(np.ones((2,), dtype=np.float32) if imitation_accel else np.zeros((2,), dtype=np.float32))
            if cfg.enable_bw_action:
                if imitation_bw:
                    mask_parts.append(np.ones((cfg.users_obs_max,), dtype=np.float32))
                else:
                    mask_parts.append(np.zeros((cfg.users_obs_max,), dtype=np.float32))
            if not cfg.fixed_satellite_strategy:
                if imitation_sat:
                    mask_parts.append(np.ones((cfg.sats_obs_max,), dtype=np.float32))
                else:
                    mask_parts.append(np.zeros((cfg.sats_obs_max,), dtype=np.float32))
            imitation_mask = torch.from_numpy(np.concatenate(mask_parts)).to(device)
            imitation_mask_sum = float(imitation_mask.sum().item())
        else:
            imitation_mask_sum = 0.0

        policy_losses = []
        value_losses = []
        entropies = []

        optim_start = time.perf_counter()
        stop_early = False
        for _ in range(cfg.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                if not torch.isfinite(obs_flat_t[mb_idx]).all():
                    print(f"NaN/Inf detected in obs minibatch at update={update}")
                    raise ValueError("obs minibatch contains NaN/Inf")
                if not torch.isfinite(act_flat_t[mb_idx]).all():
                    print(f"NaN/Inf detected in act minibatch at update={update}")
                    raise ValueError("act minibatch contains NaN/Inf")
                out = actor.forward(obs_flat_t[mb_idx])
                new_logp, entropy = actor.evaluate_actions(obs_flat_t[mb_idx], act_flat_t[mb_idx], out=out)
                if not torch.isfinite(new_logp).all():
                    print(f"NaN/Inf detected in new_logp at update={update}")
                    raise ValueError("new_logp contains NaN/Inf")

                log_ratio = new_logp - logp_flat_t[mb_idx]
                log_ratio = torch.clamp(log_ratio, -8.0, 8.0)
                ratio = torch.exp(log_ratio)
                surr1 = ratio * adv_flat_t[mb_idx]
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv_flat_t[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                approx_kl = (logp_flat_t[mb_idx] - new_logp).mean()
                kl_coef = float(getattr(cfg, "kl_coef", 0.0) or 0.0)
                if kl_coef > 0:
                    policy_loss = policy_loss + kl_coef * approx_kl
                if getattr(cfg, "kl_stop", False):
                    target_kl = float(getattr(cfg, "target_kl", 0.0) or 0.0)
                    if target_kl > 0 and float(approx_kl.item()) > target_kl:
                        stop_early = True

                imitation_loss = torch.tensor(0.0, device=device)
                if imitation_enabled and imitation_mask is not None and imitation_mask_sum > 0:
                    pred_parts = [torch.tanh(out["mu"])]
                    if cfg.enable_bw_action:
                        pred_parts.append(torch.tanh(out["bw_mu"]) * bw_scale)
                    if not cfg.fixed_satellite_strategy:
                        pred_parts.append(torch.tanh(out["sat_mu"]) * sat_scale)
                    pred_action = torch.cat(pred_parts, dim=-1)
                    target_action = imitation_flat_t[mb_idx]
                    diff = (pred_action - target_action) * imitation_mask
                    imitation_loss = (diff.pow(2).sum(-1) / (imitation_mask_sum + 1e-9)).mean()

                actor_optim.zero_grad()
                (policy_loss + imitation_coef * imitation_loss - cfg.entropy_coef * entropy.mean()).backward()
                for name, param in actor.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        print(f"NaN/Inf detected in actor grad {name} at update={update}")
                        raise ValueError("actor gradient contains NaN/Inf")
                torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                actor_optim.step()
                for name, param in actor.named_parameters():
                    if not torch.isfinite(param).all():
                        print(f"NaN/Inf detected in actor param {name} after step at update={update}")
                        raise ValueError("actor parameters contain NaN/Inf after step")

                policy_losses.append(policy_loss.item())
                entropies.append(entropy.mean().item())
                imitation_loss_sum += float(imitation_loss.item())
                if stop_early:
                    break
            if stop_early:
                break

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
            "imitation_loss": imitation_loss_sum / max(1, len(policy_losses)),
              "r_service_ratio": r_service_ratio_sum / steps_count,
            "r_drop_ratio": r_drop_ratio_sum / steps_count,
            "r_queue_pen": r_queue_pen_sum / steps_count,
            "r_queue_topk": r_queue_topk_sum / steps_count,
            "r_centroid": r_centroid_sum / steps_count,
            "centroid_dist_mean": centroid_dist_mean_sum / steps_count,
            "r_bw_align": r_bw_align_sum / steps_count,
            "r_sat_score": r_sat_score_sum / steps_count,
              "r_assoc_ratio": r_assoc_ratio_sum / steps_count,
              "r_queue_delta": r_queue_delta_sum / steps_count,
              "r_dist": r_dist_sum / steps_count,
              "r_dist_delta": r_dist_delta_sum / steps_count,
              "r_energy": r_energy_sum / steps_count,
              "r_fail_penalty": r_fail_penalty_sum / steps_count,
              "r_term_service": r_term_service_sum / steps_count,
            "r_term_drop": r_term_drop_sum / steps_count,
            "r_term_queue": r_term_queue_sum / steps_count,
            "r_term_topk": r_term_topk_sum / steps_count,
            "r_term_assoc": r_term_assoc_sum / steps_count,
              "r_term_q_delta": r_term_q_delta_sum / steps_count,
              "r_term_dist": r_term_dist_sum / steps_count,
            "r_term_dist_delta": r_term_dist_delta_sum / steps_count,
            "r_term_centroid": r_term_centroid_sum / steps_count,
            "r_term_bw_align": r_term_bw_align_sum / steps_count,
            "r_term_sat_score": r_term_sat_score_sum / steps_count,
            "r_term_energy": r_term_energy_sum / steps_count,
            "reward_raw": reward_raw_sum / steps_count,
            "arrival_sum": arrival_sum_sum / steps_count,
            "outflow_sum": outflow_sum_sum / steps_count,
            "service_norm": service_norm_sum / steps_count,
            "drop_norm": drop_norm_sum / steps_count,
            "queue_total": queue_total_sum / steps_count,
            "queue_total_active": queue_total_active_sum / steps_count,
            "arrival_rate_eff": arrival_rate_eff_sum / steps_count,
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
