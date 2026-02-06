# SAGIN-MARL

面向空天地一体化网络（SAGIN）的多智能体强化学习仿真与训练代码。环境采用 PettingZoo `ParallelEnv` 接口，建模 UAV（无人机）、GU（地面用户）与卫星的接入、回传和队列演化，并使用 MAPPO 进行训练。

**主要特性**
- 可配置的 SAGIN 环境（用户分布、卫星轨道、队列、信道与多种物理约束开关）
- MAPPO 训练管线，自动保存模型与日志
- 评估脚本与渲染脚本（GIF）
- 训练与评估日志包含队列状态、丢包与卫星处理量、用时与吞吐统计
- 训练支持早停（基于奖励滑动均值的收敛判定）

**目录结构**
- `configs/`：实验配置（YAML）
- `sagin_marl/env/`：环境与物理模型
- `sagin_marl/rl/`：MAPPO、策略与训练组件
- `scripts/`：训练、评估与渲染入口
- `tests/`：基础单元测试

**安装**
1. 创建虚拟环境（可选）
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. 安装依赖
```powershell
python -m pip install -r requirements.txt
```

**快速开始**
完整流程（训练 → 评估 → 查看结果）：
1. 激活虚拟环境（如果已创建）
```powershell
.\.venv\Scripts\Activate.ps1
```
说明：阶段一当前默认使用 `configs/phase1_actions.yaml`，关键条件/开关为：
- `enable_bw_action=true`（带宽分配动作）
- `fixed_satellite_strategy=false` + `sat_select_mode=sample`（卫星选择动作）
- `avoidance_enabled=true`（避障安全层）
对照用简化版为 `configs/phase1.yaml`（固定卫星策略、关闭带宽分配与避障安全层）。

可选：吞吐估算（判断到达率是否合理）
```powershell
python scripts/estimate_throughput.py --config configs/phase1_actions.yaml
```
2. 训练（自动生成独立目录，避免多次流程数据堆在一起）
```powershell
python scripts/train.py --config configs/phase1_actions.yaml --log_dir runs/phase1_actions --run_id auto --updates 200
```
说明：终端会输出 `Run dir: runs/phase1_actions/20260204_121530`，下文用 `<RUN_DIR>` 指代该目录。
如需手动指定目录，可用 `--run_dir runs/phase1_actions/exp1`。
3. 评估（训练模型）
```powershell
python scripts/evaluate.py --config configs/phase1_actions.yaml --run_dir <RUN_DIR> --episodes 20
```
可选：混合策略评估（accel 用训练模型，bw/sat 用 queue_aware）
```powershell
python scripts/evaluate.py --config configs/phase1_actions.yaml --run_dir <RUN_DIR> --episodes 20 --hybrid_bw_sat queue_aware
```
4. 评估（启发式基线，推荐）
```powershell
python scripts/evaluate.py --config configs/phase1_actions.yaml --run_dir <RUN_DIR> --episodes 20 --baseline queue_aware
```
可选：零加速度基线（更弱，但便于对照）
```powershell
python scripts/evaluate.py --config configs/phase1_actions.yaml --run_dir <RUN_DIR> --episodes 20 --baseline zero_accel
```
5. 查看训练结果
- 训练指标：`<RUN_DIR>/metrics.csv`
- TensorBoard（训练 + 评估）：`tensorboard --logdir <RUN_DIR>`
- 训练指标分析脚本（滚动均值 + 斜率）：`python scripts/analyze_metrics.py --run_dir <RUN_DIR> --window 20`
6. 查看评估结果
- 评估指标：`<RUN_DIR>/eval_trained.csv`, `<RUN_DIR>/eval_baseline.csv`
- 评估 TensorBoard：`<RUN_DIR>/eval_tb`（tags: `eval/trained`, `eval/baseline`）

**当前推荐配置**
- Accel-only 训练（stage1）：`configs/phase1_actions_queuefix_v3_3_stage1_accelonly.yaml`
- Full-action 训练（stage2，继承 stage1 权重）：`configs/phase1_actions_queuefix_v3_3_stage2_full.yaml`
- 固定负载主线（nearest K=20）：`configs/phase1_actions_queuefix_v3_3_imitation_fast_notanh_fixedload_nearest_k20.yaml`

**TensorBoard 查看建议（当前采用方案）**
1. 启动：
```powershell
tensorboard --logdir runs/phase1_actions
```
2. 训练曲线（Train）：
在 Runs 列表勾选 `stage1_accel` / `stage2_full` / `fixedload_nearest_k20`，查看 Scalars：
`episode_reward`, `reward_raw`, `gu_queue_mean`, `centroid_dist_mean`, `service_norm`。
3. 评估曲线（Eval）：
评估日志在 `<RUN_DIR>/eval_tb`，TensorBoard 左侧会出现 `eval_tb` 子目录。  
重点查看 `eval/trained/*` 与 `eval/baseline/*`，或在 `Custom Scalars` 中查看分组：
`Eval/Queues`, `Eval/Association`, `Eval/Reward`。
4. 只看某个方案：
在 Runs 中仅勾选对应 run（如 `stage1_accel`），并展开 `eval_tb` 查看评估曲线。

以下命令默认使用 `<RUN_DIR>`（上一步输出的 Run dir），如未设置请替换为具体目录。

训练：
```powershell
python scripts/train.py --config configs/phase1_actions.yaml --log_dir runs/phase1_actions --run_id auto --updates 200
```

评估（输出 CSV）：
```powershell
python scripts/evaluate.py --config configs/phase1_actions.yaml --run_dir <RUN_DIR> --episodes 20
```
评估（混合策略：accel=训练模型，bw/sat=queue_aware）：
```powershell
python scripts/evaluate.py --config configs/phase1_actions.yaml --run_dir <RUN_DIR> --episodes 20 --hybrid_bw_sat queue_aware
```

评估（启发式基线，推荐）：
```powershell
python scripts/evaluate.py --config configs/phase1_actions.yaml --run_dir <RUN_DIR> --episodes 20 --baseline queue_aware
```

评估（零加速度基线）：
```powershell
python scripts/evaluate.py --config configs/phase1_actions.yaml --run_dir <RUN_DIR> --episodes 20 --baseline zero_accel
```

渲染一条轨迹（输出 GIF）：
```powershell
python scripts/render_episode.py --config configs/phase1_actions.yaml --run_dir <RUN_DIR>
```

TensorBoard：
```powershell
tensorboard --logdir <RUN_DIR>
```
- 分组图在 `Custom Scalars` 标签页（训练 + 评估）。

吞吐估算（判断到达率是否合理）：
```powershell
python scripts/estimate_throughput.py --config configs/phase1_actions.yaml
```

**训练与评估输出**
训练输出：
- `<RUN_DIR>/actor.pt`、`<RUN_DIR>/critic.pt`
- `<RUN_DIR>/metrics.csv`（同时写入 TensorBoard）

评估输出：
- `<RUN_DIR>/eval_trained.csv`、`<RUN_DIR>/eval_baseline.csv`
- 指标说明文档：`docs/metrics_guide.md`
- 模型结构说明：`docs/model_architecture.md`

日志包含关键指标（部分示例）：
- `episode_reward`、`policy_loss`、`value_loss`、`entropy`
- `reward_raw`、`service_norm`、`drop_norm`、`centroid_dist_mean`
- `gu_queue_mean`、`uav_queue_mean`、`sat_queue_mean`
- `gu_queue_max`、`uav_queue_max`、`sat_queue_max`
- `gu_drop_sum`、`uav_drop_sum`
- `sat_processed_sum`、`sat_incoming_sum`
- `update_time_sec`、`rollout_time_sec`、`optim_time_sec`
- `env_steps_per_sec`、`update_steps_per_sec`
- 训练指标分析脚本：`python scripts/analyze_metrics.py --run_dir <RUN_DIR> --window 20 --metrics episode_reward,policy_loss,value_loss,entropy`

**进度条与早停**
训练与评估会显示进度条。训练支持早停（基于奖励滑动均值），相关参数在配置文件中：
- `early_stop_enabled`
- `early_stop_min_updates`
- `early_stop_window`
- `early_stop_patience`
- `early_stop_min_delta`

**环境接口**
环境类：`SaginParallelEnv`（PettingZoo Parallel API）

观测（每个 UAV 一个字典）：
- `own`: `(7,)`，自身状态（位置、速度、能量、队列、时间）
- `users`: `(users_obs_max, 5)`，候选 GU 特征
- `users_mask`: `(users_obs_max,)`
- `sats`: `(sats_obs_max, 9)`，可见卫星特征
- `sats_mask`: `(sats_obs_max,)`
- `nbrs`: `(nbrs_obs_max, 4)`，邻居 UAV 特征
- `nbrs_mask`: `(nbrs_obs_max,)`

动作（每个 UAV 一个字典）：
- `accel`: `(2,)`，二维加速度（归一化后再乘以 `a_max`）
- `bw_logits`: `(users_obs_max,)`，带宽分配权重（`enable_bw_action=true` 生效）
- `sat_logits`: `(sats_obs_max,)`，卫星选择权重（`fixed_satellite_strategy=false` 生效）

**候选 GU 机制（重要）**
- `candidate_mode=assoc`：候选集合由“当前关联到该 UAV 的 GU”组成。即使全局接入率接近 1，每个 UAV 看到的仍是自己服务的子集。
- `candidate_mode=nearest|radius`：候选集合按距离筛选，视野更广，但**带宽分配仍只对 `assoc==u` 的 GU 生效**，不会出现多个 UAV 同时给同一 GU 分配带宽。
- `candidate_k`：控制候选数量上限（不改变 `users_obs_max`，网络输入维度保持不变，超出部分由 mask 置零）。

**配置说明**
默认配置见 `sagin_marl/env/config.py`，阶段一默认使用 `configs/phase1_actions.yaml` 覆盖其中字段；对照可用 `configs/phase1.yaml`。常用参数：
- 规模：`num_uav`、`num_gu`、`num_sat`
- 时域：`tau0`、`T_steps`
- 观测截断：`users_obs_max`、`sats_obs_max`、`nbrs_obs_max`
- 物理约束：`v_max`、`a_max`、`d_safe`、`boundary_mode`
- 通信与噪声：`b_acc`、`b_sat_total`、`gu_tx_power`、`uav_tx_power`、`noise_density`、`pathloss_mode`
- 天线增益与卫星算力：`uav_tx_gain`、`sat_rx_gain`、`sat_cpu_freq`
- 机制开关：`enable_bw_action`、`fixed_satellite_strategy`、`doppler_enabled`、`energy_enabled`
- 奖励与候选：`reward_tanh_enabled`、`candidate_mode`、`candidate_k`、`candidate_radius`、`queue_topk_local`
- 基线启发式（queue_aware）：`baseline_accel_gain`、`baseline_assoc_bonus`、`baseline_sat_queue_penalty`、`baseline_repulse_gain`、`baseline_repulse_radius_factor`、`baseline_energy_low`、`baseline_energy_weight`
- 训练超参：`buffer_size`、`num_mini_batch`、`ppo_epochs`、`actor_lr`、`critic_lr`

**测试**
- 测试文件说明：`docs/tests_overview.md`
```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```

**最小使用示例**
```python
from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv

cfg = load_config("configs/phase1_actions.yaml")
env = SaginParallelEnv(cfg)
obs, infos = env.reset()
```
