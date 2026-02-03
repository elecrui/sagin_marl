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
2. 训练（生成模型与训练日志）
```powershell
python scripts/train.py --config configs/phase1.yaml --log_dir runs/phase1 --updates 50
```
3. 评估（训练模型）
```powershell
python scripts/evaluate.py --config configs/phase1.yaml --checkpoint runs/phase1/actor.pt --episodes 5 --out runs/phase1/eval.csv
```
4. 评估（零加速度基线，用于对照）
```powershell
python scripts/evaluate.py --config configs/phase1.yaml --episodes 5 --out runs/phase1/eval.csv --baseline zero_accel
```
5. 查看训练结果
- 训练指标：`runs/phase1/metrics.csv`
- TensorBoard：`python -m tensorboard --logdir runs/phase1`
6. 查看评估结果
- 评估指标：`runs/phase1/eval.csv`

训练：
```powershell
python scripts/train.py --config configs/phase1.yaml --log_dir runs/phase1 --updates 50
```

评估（输出 CSV）：
```powershell
python scripts/evaluate.py --config configs/phase1.yaml --checkpoint runs/phase1/actor.pt --episodes 5 --out runs/phase1/eval.csv
```

评估（零加速度基线）：
```powershell
python scripts/evaluate.py --config configs/phase1.yaml --episodes 5 --out runs/phase1/eval.csv --baseline zero_accel
```

渲染一条轨迹（输出 GIF）：
```powershell
python scripts/render_episode.py --config configs/phase1.yaml --checkpoint runs/phase1/actor.pt --out runs/phase1/episode.gif
```

TensorBoard：
```powershell
python -m tensorboard --logdir runs/phase1
```

**训练与评估输出**
训练输出：
- `runs/phase1/actor.pt`、`runs/phase1/critic.pt`
- `runs/phase1/metrics.csv`（同时写入 TensorBoard）

评估输出：
- `runs/phase1/eval.csv`

日志包含关键指标（部分示例）：
- `episode_reward`、`policy_loss`、`value_loss`、`entropy`
- `gu_queue_mean`、`uav_queue_mean`、`sat_queue_mean`
- `gu_queue_max`、`uav_queue_max`、`sat_queue_max`
- `gu_drop_sum`、`uav_drop_sum`
- `sat_processed_sum`、`sat_incoming_sum`
- `update_time_sec`、`rollout_time_sec`、`optim_time_sec`
- `env_steps_per_sec`、`update_steps_per_sec`

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

**配置说明**
默认配置见 `sagin_marl/env/config.py`，`configs/phase1.yaml` 会覆盖其中字段。常用参数：
- 规模：`num_uav`、`num_gu`、`num_sat`
- 时域：`tau0`、`T_steps`
- 观测截断：`users_obs_max`、`sats_obs_max`、`nbrs_obs_max`
- 物理约束：`v_max`、`a_max`、`d_safe`
- 通信与噪声：`b_acc`、`b_sat_total`、`gu_tx_power`、`uav_tx_power`、`noise_density`
- 天线增益与卫星算力：`uav_tx_gain`、`sat_rx_gain`、`sat_cpu_freq`
- 机制开关：`enable_bw_action`、`fixed_satellite_strategy`、`doppler_enabled`、`energy_enabled`
- 训练超参：`buffer_size`、`num_mini_batch`、`ppo_epochs`、`actor_lr`、`critic_lr`

**测试**
```powershell
python -m pytest -q
```

**最小使用示例**
```python
from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv

cfg = load_config("configs/phase1.yaml")
env = SaginParallelEnv(cfg)
obs, infos = env.reset()
```
