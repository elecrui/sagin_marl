# 模型结构说明

本文档只描述当前实际启用的模型结构，不再覆盖已经停用或当前配置未使用的分支。

对应配置文件：

- `configs/phase1_actions_curriculum_stage1_accel_setpool_throughput_only_reward_q002_drop6_checkpoint_eval50_start200_norewardnorm_klstop002_earlystop_lrsplit_unfair.yaml`
- `configs/phase1_actions_curriculum_stage2_bw_setpool_throughput_only_reward_q002_drop6_from_stage1_accel_teacher_ckpteval50_start200_norewardnorm_klstop002_frozenbackbone_resbw_a05_clip1_l2e2_no_backhaul_gupen005.yaml`
- `configs/phase1_actions_curriculum_stage3_sat.yaml`

相关实现：

- `sagin_marl/rl/policy.py`
- `sagin_marl/rl/critic.py`
- `sagin_marl/rl/mappo.py`
- `sagin_marl/rl/action_assembler.py`
- `sagin_marl/rl/baselines.py`
- `sagin_marl/env/sagin_env.py`

## 1. 结论

这三个配置使用的是同一套 MAPPO 框架：

- actor 都是 `ActorNet`
- critic 都是 `CriticNet`
- actor 编码器类型都为 `actor_encoder_type: set_pool`
- actor 隐层宽度都为 `actor_hidden: 256`
- set encoder 嵌入维度都为 `actor_set_embed_dim: 64`
- critic 隐层宽度都为 `critic_hidden: 256`
- 输入归一化都开启了 `input_norm_enabled: true`

但它们不是“完全相同的网络实例”。

更准确地说，这三个阶段共享同一套骨干设计，但按阶段打开了不同动作头，并且训练/执行方式不同：

| 阶段 | actor 内部启用的头 | 动作向量维度 | 真正执行的动作来源 | 训练状态 |
| --- | --- | ---: | --- | --- |
| stage1 | `accel` | 2 | `accel` 来自当前策略；`bw=0`；`sat=0` | 训练 backbone + accel head |
| stage2 | `accel + bw` | 22 | `accel` 来自 stage1 teacher；`bw` 为启发式队列策略 + residual；`sat=0` | 冻结 backbone，仅训练 bw head |
| stage3 | `accel + bw + sat` | 28 | `accel` 来自 teacher；`bw` 仍为启发式 + residual；`sat` 来自当前策略 | 冻结 backbone，仅训练 sat head |

因此可以说：

- 三阶段使用的是同一套 actor/critic 结构范式
- 但每个阶段实际实例化的输出头、参与训练的参数、以及执行时采用的动作来源并不完全一样

## 2. 整体训练结构

当前训练仍然是标准的集中式 critic、分散式 actor：

- actor 输入是每个 UAV 的局部观测 `obs`
- critic 输入是环境全局状态 `env.get_global_state()`
- 每个 UAV 共用同一个 actor 网络参数
- critic 输出单个标量状态价值 `V(s)`

也就是说：

- actor 解决“每个 UAV 该怎么动、怎么分带宽、怎么选卫星”
- critic 解决“当前全局状态值多少钱”

## 3. Actor 输入定义

### 3.1 输入来源

actor 的输入来自 `SaginParallelEnv._get_obs()`，然后在 `sagin_marl/rl/policy.py` 中通过 `flatten_obs()` 展平成一维向量。

这三个配置下，展平顺序固定为：

1. `own`
2. `danger_nbr`
3. `users`
4. `users_mask`
5. `sats`
6. `sats_mask`
7. `nbrs`
8. `nbrs_mask`

因为当前三个配置都启用了：

- `danger_nbr_enabled: true`
- `users_obs_max: 20`
- `sats_obs_max: 6`
- `nbrs_obs_max: 4`

所以 actor 的实际输入维度固定为：

```text
obs_dim
= own(7)
+ danger_nbr(5)
+ users(20 * 5)
+ users_mask(20)
+ sats(6 * 12)
+ sats_mask(6)
+ nbrs(4 * 4)
+ nbrs_mask(4)
= 230
```

程序实测这三个配置的 `obs_dim` 都是 `230`。

### 3.2 `own`，形状 `(7,)`

当前 UAV 自身状态：

1. `uav_pos_x / map_size`
2. `uav_pos_y / map_size`
3. `uav_vel_x / v_max`
4. `uav_vel_y / v_max`
5. `uav_energy / uav_energy_init`
6. `uav_queue / queue_max_uav`
7. `t / T_steps`

虽然当前配置里 `energy_enabled: false`，但能量特征仍然保留在观测中，数值会随环境状态更新。

### 3.3 `danger_nbr`，形状 `(5,)`

这是一个“最危险邻机摘要”，不是邻机集合本身。

特征定义：

1. 邻机距离 `dist / map_size`
2. 接近速度 `closing_speed / v_max`
3. 相对方向 `dir_x`
4. 相对方向 `dir_y`
5. 有效标记 `1.0/0.0`

它的作用是给 actor 一个单独的安全提示通道，让骨干网络不必只靠 `nbrs` 集合自己推理“当前最危险的那架邻机是谁”。

### 3.4 `users`，形状 `(20, 5)`，配合 `users_mask`

每个 UAV 最多观察 `20` 个候选地面用户。当前配置中：

- `candidate_mode: nearest`
- `candidate_k: 20`
- `users_obs_max: 20`

所以这里基本就是“按距离排序后的最近 20 个候选 GU”。

单个用户特征定义：

1. `rel_x / map_size`
2. `rel_y / map_size`
3. `gu_queue / queue_max_gu`
4. 接入链路频谱效率 `eta`
5. 上一步是否关联到当前 UAV 的标记 `prev_association == u`

`users_mask[i] = 1` 表示第 `i` 个槽位有效，否则该槽位是 padding。

注意：

- actor 看到的是候选用户列表，不是“已经关联成功的用户列表”
- 真正做带宽分配时，只会对其中已经关联到当前 UAV 的用户做 masked softmax

### 3.5 `sats`，形状 `(6, 12)`，配合 `sats_mask`

每个 UAV 最多观察 `6` 个候选卫星。

这三个配置都满足：

- `visible_sats_max: 6`
- `sats_obs_max: 6`

因此 actor 的卫星输入槽位和卫星动作输出槽位是一一对应的。

候选卫星由环境先筛选可见卫星，再按候选规则排序并截断。当前配置使用的是：

- stage1/stage2：默认 `sat_candidate_mode: elevation`
- stage3：显式设置 `sat_candidate_mode: elevation`

单个卫星特征定义：

1. 相对位置 `rel_pos_x / (r_earth + sat_height)`
2. 相对位置 `rel_pos_y / (r_earth + sat_height)`
3. 相对位置 `rel_pos_z / (r_earth + sat_height)`
4. 相对速度 `rel_vel_x / (r_earth + sat_height)`
5. 相对速度 `rel_vel_y / (r_earth + sat_height)`
6. 相对速度 `rel_vel_z / (r_earth + sat_height)`
7. 多普勒归一化 `nu / nu_max`
8. 当前链路频谱效率 `spectral_efficiency`
9. 卫星队列归一化 `sat_queue / queue_max_sat`
10. 当前负载计数归一化 `load_count / num_uav`
11. 预计接入后负载倒数 `1 / projected_count`
12. 是否是当前已连接卫星 `stay_flag`

`sats_mask[i] = 1` 表示该候选槽位有效，否则为 padding。

即使 stage1 和 stage2 不训练卫星动作头，actor 仍然会看到这组卫星观测，因为 UAV 加速度和带宽决策也会用到这些卫星侧上下文。

### 3.6 `nbrs`，形状 `(4, 4)`，配合 `nbrs_mask`

每个 UAV 最多观察 `4` 个邻近 UAV，按距离排序截断。

单个邻机特征定义：

1. `rel_pos_x / map_size`
2. `rel_pos_y / map_size`
3. `rel_vel_x / v_max`
4. `rel_vel_y / v_max`

`nbrs_mask[i] = 1` 表示该槽位有效，否则为 padding。

## 4. Actor 骨干结构

### 4.1 当前启用的是 `set_pool` 编码器

当前三阶段都没有使用旧的 `flat_mlp`，实际启用的是 `set_pool`。

其核心思路是：

- 对标量/小向量输入单独编码
- 对可变长集合输入先逐元素编码
- 再通过带 mask 的 mean pooling + max pooling 聚合
- 最后把所有分支特征拼接后送入融合 MLP

### 4.2 各子编码器

当前参数：

- `actor_set_embed_dim = 64`
- `actor_hidden = 256`
- `input_norm_enabled = true`

因此 actor 各分支为：

#### `own_encoder`

```text
LayerNorm(7)
Linear(7 -> 64)
ReLU
Linear(64 -> 64)
ReLU
```

输出维度：`64`

#### `danger_nbr_encoder`

```text
LayerNorm(5)
Linear(5 -> 64)
ReLU
Linear(64 -> 64)
ReLU
```

输出维度：`64`

#### `users_encoder`

对每个用户槽位独立编码：

```text
LayerNorm(5)
Linear(5 -> 64)
ReLU
Linear(64 -> 64)
ReLU
```

然后结合 `users_mask` 做两种池化：

- masked mean pooling，得到 `64`
- masked max pooling，得到 `64`

拼接后得到用户集合特征：`128`

#### `sats_encoder`

对每个候选卫星槽位独立编码：

```text
LayerNorm(12)
Linear(12 -> 64)
ReLU
Linear(64 -> 64)
ReLU
```

再做 masked mean + max pooling，得到卫星集合特征：`128`

#### `nbrs_encoder`

对每个邻机槽位独立编码：

```text
LayerNorm(4)
Linear(4 -> 64)
ReLU
Linear(64 -> 64)
ReLU
```

再做 masked mean + max pooling，得到邻机集合特征：`128`

### 4.3 特征融合层

当前配置启用了 `danger_nbr`，因此最终拼接维度为：

```text
fusion_in_dim
= own_feat(64)
+ danger_nbr_feat(64)
+ users_pool(128)
+ sats_pool(128)
+ nbrs_pool(128)
= 512
```

融合 MLP 为：

```text
Linear(512 -> 256)
ReLU
Linear(256 -> 256)
ReLU
```

其输出就是所有动作头共享的 actor backbone 表示。

## 5. Actor 输出头定义

### 5.1 加速度头 `accel`

所有阶段都存在 `accel` 头：

```text
mu_head: Linear(256 -> 2)
log_std: trainable parameter, shape=(2,)
```

动作分布：

- 对每一维使用高斯分布 `Normal(mu, std)`
- `log_std` 先 clamp 到 `[-5, 2]`
- 再通过 `exp(log_std)` 得到 `std`
- 训练时使用 `rsample()`
- 最终经过 `tanh` squash 到 `[-1, 1]`

输出含义：

- actor 输出的是归一化加速度 `a_norm in [-1, 1]^2`
- 环境执行前会乘上 `a_max = 5.0`
- 即实际物理加速度范围是 `[-5, 5] m/s^2`

注意执行层还有额外安全处理：

- 避碰修正
- 边界硬约束
- pairwise hard filter

因此真正执行的加速度不一定等于策略原始输出。

### 5.2 带宽头 `bw`

只有 stage2 和 stage3 启用了 `enable_bw_action: true`，因此这两个阶段 actor 中存在 `bw` 头：

```text
bw_head: Linear(256 -> 20)
bw_log_std: trainable parameter, shape=(20,)
```

当前配置的特殊设置：

- `bw_head_zero_init: true`
- `bw_log_std_init: -1.5`
- `bw_log_std_trainable: false`

也就是说：

- stage2/stage3 的 `bw_head` 初始权重和偏置都为 0
- `bw_log_std` 初始化为 `-1.5`
- 训练时 `bw_log_std` 不更新

更重要的是，当前 stage2/stage3 不是“直接输出最终带宽分配”，而是 residual 带宽策略：

- `exec_bw_source: heuristic_residual`
- `bw_residual_alpha: 0.5`
- `bw_residual_clip: 1.0`

因此 actor 的 `bw` 头建模的是一个 residual logit `delta_bw`，其采样范围为：

```text
delta_bw = tanh(z_bw) * 1.0
```

即每个槽位都在 `[-1, 1]` 内。

真正送入环境执行的带宽 logit 为：

```text
bw_exec = clip( bw_heuristic + 0.5 * clip(delta_bw, [-1, 1]), [-5, 5] )
```

其中：

- `bw_heuristic` 来自 `queue_aware_policy()`
- `[-5, 5]` 来自 `bw_logit_scale`

而 PPO 计算 `logprob` 时，对应的是 residual 动作 `delta_bw`，不是上面融合后的 `bw_exec`。

### 5.3 卫星头 `sat`

只有 stage3 启用了：

- `fixed_satellite_strategy: false`
- `train_sat: true`

所以只有 stage3 actor 中存在卫星头：

```text
sat_head: Linear(256 -> 6)
sat_log_std: trainable parameter, shape=(6,)
```

其分布形式与 `bw` 类似：

```text
sat_logits = tanh(z_sat) * sat_logit_scale
```

当前 `sat_logit_scale = 5.0`，所以每个候选卫星槽位的输出范围是 `[-5, 5]`。

这些 `sat_logits` 与 `sats` 输入中的 6 个候选卫星槽位一一对应。

## 6. 动作向量拼接与环境语义

### 6.1 actor 内部动作向量顺序

在 `ActorNet` 内部，动作向量的拼接顺序固定为：

1. `accel(2)`
2. `bw(20)`，如果启用
3. `sat(6)`，如果启用

因此三阶段动作维度分别是：

- stage1：`2`
- stage2：`2 + 20 = 22`
- stage3：`2 + 20 + 6 = 28`

### 6.2 `bw_logits` 如何转成实际带宽分配

环境并不会把 `bw_logits` 直接当作带宽比例，而是做如下处理：

1. 先根据路径损耗阈值完成 GU 到 UAV 的关联
2. 对当前 UAV 的候选用户列表取出已关联子集
3. 仅在该子集上做 masked softmax
4. 得到带宽份额 `beta`
5. 用 `beta * b_acc * spectral_efficiency` 计算接入速率

因此：

- `bw` 输出本质上是“关联后用户子集上的相对优先级 logit”
- 没有关联成功的用户即使在观测里出现，也不会得到带宽

如果 `enable_bw_action: false`，环境会对已关联用户做平均分配。

### 6.3 `sat_logits` 如何转成卫星选择

stage3 中：

1. 环境先得到当前 UAV 的可见卫星候选
2. 只保留前 `sats_obs_max = 6` 个槽位
3. 读取与这 6 个槽位一一对应的 `sat_logits`
4. 按 `sat_select_mode: sample` 做抽样
5. 最多选择 `N_RF = 2` 颗卫星，且无放回

如果启用了多普勒过滤，则多普勒超限的候选卫星会被强行置为无效。

stage1 和 stage2 中，由于 `fixed_satellite_strategy: true`，不使用 `sat` 头，环境会在当前保留的候选卫星集合里选择距离最近的一颗卫星。

## 7. Critic 输入定义

critic 使用 `SaginParallelEnv.get_global_state()` 输出的一维全局状态。

当前三个配置下，global state 的拼接顺序为：

1. `uav_pos.flatten() / map_size`
2. `uav_vel.flatten() / v_max`
3. `uav_queue / queue_max_uav`
4. `uav_energy / uav_energy_init`
5. `gu_pos.flatten() / map_size`
6. `gu_queue / queue_max_gu`
7. `sat_pos.flatten() / (r_earth + sat_height)`
8. `sat_vel.flatten() / (r_earth + sat_height)`
9. `sat_queue / queue_max_sat`
10. `t / T_steps`

### 7.1 当前实际 state 维度

当前三个配置虽然 `num_sat` 不同：

- stage1/stage2：`num_sat = 72`
- stage3：`num_sat = 144`

但它们都设置了：

- `sat_state_max = 9`

因此 critic 不会看到全部卫星，只会保留“对所有 UAV 来说最大仰角最高”的前 9 颗卫星进入 global state。

所以这三个配置的全局状态维度相同：

```text
state_dim
= uav_pos(3 * 2 = 6)
+ uav_vel(3 * 2 = 6)
+ uav_queue(3)
+ uav_energy(3)
+ gu_pos(20 * 2 = 40)
+ gu_queue(20)
+ sat_pos(9 * 3 = 27)
+ sat_vel(9 * 3 = 27)
+ sat_queue(9)
+ time(1)
= 142
```

程序实测这三个配置的 `state_dim` 都是 `142`。

## 8. Critic 结构

当前 critic 没有使用 set encoder，而是一个普通 MLP：

```text
LayerNorm(142)
Linear(142 -> 256)
ReLU
Linear(256 -> 256)
ReLU
Linear(256 -> 1)
```

最后经过 `squeeze(-1)` 输出标量状态价值。

即：

- 输入：全局状态 `s`
- 输出：`V(s)`

## 9. 三阶段各自到底训练什么

这里的“只训练 accel / bw / sat”是指 actor 侧的可训练部分。

critic 在三个阶段都会继续训练，只是它的输入始终是全局状态，输出始终是状态价值 `V(s)`。

### 9.1 stage1：只训练加速度策略

配置特征：

- `enable_bw_action: false`
- `fixed_satellite_strategy: true`
- `train_accel: true`
- `train_bw: false`
- `train_sat: false`
- `train_shared_backbone: true`

含义：

- actor 只有 accel 头
- backbone 与 accel 头一起训练
- 不学习带宽分配
- 不学习卫星选择
- 执行时卫星采用固定策略，带宽平均分配

### 9.2 stage2：冻结骨干，只学 bw residual

配置特征：

- `enable_bw_action: true`
- `fixed_satellite_strategy: true`
- `train_accel: false`
- `train_bw: true`
- `train_sat: false`
- `train_shared_backbone: false`
- `exec_accel_source: teacher`
- `exec_bw_source: heuristic_residual`

含义：

- actor 内部保留 accel 头和 bw 头
- 但 backbone 冻结
- accel 头也不训练，执行时直接用 stage1 teacher 的 accel
- 只训练 bw 头输出 residual logit
- 卫星仍采用固定候选集内最近卫星策略

### 9.3 stage3：冻结骨干，只学 sat head

配置特征：

- `enable_bw_action: true`
- `fixed_satellite_strategy: false`
- `train_accel: false`
- `train_bw: false`
- `train_sat: true`
- `train_shared_backbone: false`
- `exec_accel_source: teacher`
- `exec_bw_source: heuristic_residual`
- `exec_sat_source: policy`

含义：

- actor 内部包含 accel、bw、sat 三个头
- backbone 冻结
- accel 不训练，执行时仍来自 teacher
- bw 不训练，执行时仍是启发式 + residual
- 只训练 sat 头来学习候选卫星排序/选择

在这个阶段，sat 头才第一次真正成为策略学习目标。

## 10. 文档范围说明

本文档只覆盖当前三个配置实际启用的模型结构，因此以下内容不展开：

- `flat_mlp` actor 编码器
- 当前配置未启用的其他 imitation 分支
- 与当前配置无关的旧实验开关

如果后续配置切回 `flat_mlp`、改动 `users_obs_max/sats_obs_max/nbrs_obs_max`、或取消 `danger_nbr_enabled`，那么本文中的维度数字需要同步更新。
