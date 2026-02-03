# 训练与评估指标说明

本项目的训练指标写入 `runs/<log_dir>/metrics.csv` 与 TensorBoard，评估指标写入 `runs/<log_dir>/eval_trained.csv` / `runs/<log_dir>/eval_baseline.csv`，并同步写入 TensorBoard（默认在 `runs/<log_dir>/eval_tb`，标签为 `eval/trained` 与 `eval/baseline`）。

**训练指标（TensorBoard + metrics.csv）**
- `episode_reward`：每次 update 的平均回报（总回报 / 步数）。趋势上升或稳定提升表示在学。
- `policy_loss`：策略损失（PPO）。不要求单调下降，稳定且不过分震荡更健康。
- `value_loss`：价值网络误差。持续过大或波动剧烈常表示估值不稳。
- `entropy`：策略随机性。逐步下降是收敛迹象，过快接近 0 可能探索不足。
- `gu_queue_mean`：GU 平均队列长度。越小越好。
- `uav_queue_mean`：UAV 平均队列长度。越小越好。
- `sat_queue_mean`：卫星平均队列长度。越小越好。
- `gu_queue_max`：GU 队列峰值。尖刺多表示拥塞或波动。
- `uav_queue_max`：UAV 队列峰值。
- `sat_queue_max`：卫星队列峰值。
- `gu_drop_sum`：GU 丢包/丢任务总量。越小越好。
- `uav_drop_sum`：UAV 丢包/丢任务总量。
- `sat_processed_sum`：卫星处理总量。与 `sat_incoming_sum` 对比看处理能力是否匹配负载。
- `sat_incoming_sum`：卫星到达总量。
- `energy_mean`：平均能量消耗（`energy_enabled=true` 时有效）。需与性能权衡。
- `update_time_sec`：一次 update 总耗时。
- `rollout_time_sec`：采样耗时。
- `optim_time_sec`：优化耗时。
- `env_steps`：本次 update 的环境步数。
- `env_steps_per_sec`：采样吞吐。
- `update_steps_per_sec`：整体吞吐（含优化）。
- `total_env_steps`：累计环境步数。
- `total_time_sec`：累计训练时间。

**评估指标（TensorBoard + eval.csv）**
- `reward_sum`：单个 episode 的总回报。用于与基线直接对比。
- `steps`：该 episode 步数。
- `episode_time_sec`：该 episode 耗时。
- `steps_per_sec`：评估时的吞吐速度。
- `gu_queue_mean`：GU 平均队列长度。
- `uav_queue_mean`：UAV 平均队列长度。
- `sat_queue_mean`：卫星平均队列长度。
- `gu_queue_max`：GU 队列峰值。
- `uav_queue_max`：UAV 队列峰值。
- `sat_queue_max`：卫星队列峰值。
- `gu_drop_sum`：GU 丢包/丢任务总量。
- `uav_drop_sum`：UAV 丢包/丢任务总量。
- `sat_processed_sum`：卫星处理总量。
- `sat_incoming_sum`：卫星到达总量。
- `energy_mean`：平均能量消耗（`energy_enabled=true` 时有效）。

**分析方法（建议顺序）**
1. 先看 `episode_reward`（训练）和 `reward_sum`（评估）。训练看趋势，评估看均值与方差，并与 `eval/baseline` 对比。
2. 看 `queue_mean` 与 `drop_sum` 是否下降，说明拥塞缓解与服务质量改善。
3. 看 `queue_max` 是否明显收敛，峰值下降代表系统更稳定。
4. 结合 `entropy` 与回报变化判断探索是否充足，过早低熵且回报停滞可考虑提高探索或训练步数。
5. 看 `sat_processed_sum` 与 `sat_incoming_sum` 是否接近，偏离大可能存在处理瓶颈或调度不合理。
6. 若关注能耗，综合 `energy_mean` 与回报/队列指标，寻找性能与能耗的折中点。
7. 在 TensorBoard 中适度提高 Smoothing（如 0.6~0.9）查看趋势，必要时回看原始曲线排查波动。

**TensorBoard 分组视图**
- 训练分组在 TensorBoard 的 `Custom Scalars` 标签页中，按 Reward / Losses / Queues / Drops / Satellite / Performance / Totals / Energy 分类。
- 评估分组同样在 `Custom Scalars` 中，默认标签为 `eval/trained` 与 `eval/baseline`，并提供对比曲线分组（Reward、Queues、Drops、SatFlow）。
