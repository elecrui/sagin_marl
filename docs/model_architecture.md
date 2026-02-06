# 模型网络结构

本文档描述当前 MAPPO 的 actor/critic 网络结构与输入输出定义，相关代码见 `sagin_marl/rl/policy.py`、`sagin_marl/rl/critic.py`、`sagin_marl/rl/mappo.py`、`sagin_marl/env/sagin_env.py`。

**整体概览**
- 训练范式：集中式 critic + 分散式 actor。
- Actor 输入：单个 UAV 的观测向量（由观测字典展平而来）。
- Critic 输入：环境全局状态向量 `env.get_global_state()`。

**观测展平**
- 展平顺序：`own`、`users`、`users_mask`、`sats`、`sats_mask`、`nbrs`、`nbrs_mask`。
- 维度公式：`own_dim + users_obs_max*user_dim + users_obs_max + sats_obs_max*sat_dim + sats_obs_max + nbrs_obs_max*nbr_dim + nbrs_obs_max`。
- 当前默认维度：`own_dim=7`、`user_dim=5`、`sat_dim=9`、`nbr_dim=4`（定义在 `SaginParallelEnv`）。

**ActorNet（`sagin_marl/rl/policy.py`）**
- 主干：`Linear(obs_dim -> actor_hidden)` + ReLU -> `Linear(actor_hidden -> actor_hidden)` + ReLU。
- 加速度头：`mu_head` 输出 2 维均值，`log_std` 为可学习参数（形状为 2）。
- 带宽头（可选）：当 `enable_bw_action=true` 时启用，`bw_head` 输出 `users_obs_max` 维均值，`bw_log_std` 为可学习参数。
- 卫星头（可选）：当 `fixed_satellite_strategy=false` 时启用，`sat_head` 输出 `sats_obs_max` 维均值，`sat_log_std` 为可学习参数。
- 标准差处理：`log_std` 夹紧到 `[-5, 2]` 后指数化得到 `std`。
- 动作采样：`Normal(μ, σ)` 通过 `rsample()` 采样；`deterministic=True` 时直接取均值。
- 动作 squash：使用 `tanh`，其中 `bw`/`sat` 额外乘以 `bw_logit_scale` / `sat_logit_scale`。
- 对数概率：使用 `_logprob_from_squashed` 对 `tanh` 进行校正。

**动作向量拼接**
- 拼接顺序：`accel(2)` -> `bw(users_obs_max, 若启用)` -> `sat(sats_obs_max, 若启用)`。
- `evaluate_actions` 使用相同顺序拆分并计算 logprob 与 entropy。

**CriticNet（`sagin_marl/rl/critic.py`）**
- 主干：`Linear(state_dim -> critic_hidden)` + ReLU -> `Linear(critic_hidden -> critic_hidden)` + ReLU。
- 输出：`Linear(critic_hidden -> 1)`，并 `squeeze(-1)` 得到标量值函数。

**全局状态（`SaginParallelEnv.get_global_state`）**
- 组成：`uav_pos`、`uav_vel`、`uav_queue`、`uav_energy`、`gu_pos`、`gu_queue`、`sat_pos`、`sat_vel`、`sat_queue`、`t`。
- 归一化：位置/速度/队列/能量/时间均按配置参数缩放（如 `map_size`、`v_max`、`queue_max_*`、`uav_energy_init`、`r_earth + sat_height`、`T_steps`）。
- 卫星状态裁剪：当 `cfg.sat_state_max` 设置且小于 `num_sat` 时，仅保留仰角最高的一部分卫星进入全局状态。
