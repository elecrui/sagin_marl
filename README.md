# SAGIN-MARL

面向空-天-地一体化网络（SAGIN）的多智能体强化学习仿真与训练代码。环境采用 PettingZoo `ParallelEnv` 接口，建模 UAV（无人机）、GU（地面用户）与卫星的接入与回传队列演化，并使用 MAPPO 进行训练。

**主要特性**
- 可配置的 SAGIN 环境（用户聚类、卫星轨道、队列、信道与多种物理约束开关）
- MAPPO 训练管线，自动保存模型与日志
- 评估脚本与渲染脚本（GIF）

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
其他启发式 baseline 说明见 `docs/heuristic_baselines.md`。

渲染一条轨迹（输出 GIF）：
```powershell
python scripts/render_episode.py --config configs/phase1.yaml --checkpoint runs/phase1/actor.pt --out runs/phase1/episode.gif
```

TensorBoard：
```powershell
python -m tensorboard --logdir runs/phase1
```

**环境接口**
环境类：`SaginParallelEnv`（PettingZoo Parallel API）

观测（每个 UAV 一个字典）：
- `own`: `(7,)`，自身状态（位置/速度/能量/队列/时间）
- `users`: `(users_obs_max, 5)`，候选 GU 特征
- `users_mask`: `(users_obs_max,)`
- `sats`: `(sats_obs_max, 9)`，可见卫星特征
- `sats_mask`: `(sats_obs_max,)`
- `nbrs`: `(nbrs_obs_max, 4)`，邻居 UAV 特征
- `nbrs_mask`: `(nbrs_obs_max,)`

动作（每个 UAV 一个字典）：
- `accel`: `(2,)`，二维加速度（归一化后再乘以 `a_max`）
- `bw_logits`: `(users_obs_max,)`，带宽分配权重（`enable_bw_action=true` 时生效）
- `sat_logits`: `(sats_obs_max,)`，卫星选择权重（`fixed_satellite_strategy=false` 时生效）

**配置说明**
默认配置见 `sagin_marl/env/config.py`，`configs/phase1.yaml` 会覆盖其中字段。常用参数：
- 规模：`num_uav`, `num_gu`, `num_sat`
- 时域：`tau0`, `T_steps`
- 观测截断：`users_obs_max`, `sats_obs_max`, `nbrs_obs_max`
- 物理与约束：`v_max`, `a_max`, `d_safe`
- 通信：`b_acc`, `b_sat_total`, `gu_tx_power`, `uav_tx_power`
- 机制开关：`enable_bw_action`, `fixed_satellite_strategy`, `doppler_enabled`, `energy_enabled`
- 训练超参：`buffer_size`, `num_mini_batch`, `ppo_epochs`, `actor_lr`, `critic_lr`

**输出文件**
训练完成后自动生成：
- `runs/phase1/actor.pt`、`runs/phase1/critic.pt`
- `runs/phase1/metrics.csv`
- TensorBoard 日志（同目录）

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
