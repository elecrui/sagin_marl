import torch

from sagin_marl.env.config import SaginConfig
from sagin_marl.rl.mappo import _checkpoint_eval_fieldnames, _checkpoint_eval_summary, _update_checkpoint_eval_state


def test_checkpoint_eval_stop_triggers_after_consecutive_sat_drop_worsening():
    cfg = SaginConfig(
        checkpoint_eval_sat_drop_worsen_delta=1e-3,
        checkpoint_eval_front_queue_rel_improve_tol=0.05,
        checkpoint_eval_worsen_patience=2,
    )
    state = {}

    first = _update_checkpoint_eval_state(
        state,
        {
            'gu_queue_arrival_steps_mean': 10.0,
            'uav_queue_arrival_steps_mean': 20.0,
            'sat_drop_ratio': 0.020,
        },
        cfg,
    )
    second = _update_checkpoint_eval_state(
        state,
        {
            'gu_queue_arrival_steps_mean': 10.0,
            'uav_queue_arrival_steps_mean': 20.0,
            'sat_drop_ratio': 0.022,
        },
        cfg,
    )
    third = _update_checkpoint_eval_state(
        state,
        {
            'gu_queue_arrival_steps_mean': 10.2,
            'uav_queue_arrival_steps_mean': 20.1,
            'sat_drop_ratio': 0.024,
        },
        cfg,
    )

    assert first['early_stop_triggered'] == 0.0
    assert second['sat_drop_worse_streak'] == 1.0
    assert second['early_stop_triggered'] == 0.0
    assert third['sat_drop_worse_streak'] == 2.0
    assert third['early_stop_triggered'] == 1.0


def test_checkpoint_eval_front_queue_improvement_resets_worsening_streak():
    cfg = SaginConfig(
        checkpoint_eval_sat_drop_worsen_delta=1e-3,
        checkpoint_eval_front_queue_rel_improve_tol=0.05,
        checkpoint_eval_worsen_patience=2,
    )
    state = {}

    _update_checkpoint_eval_state(
        state,
        {
            'gu_queue_arrival_steps_mean': 10.0,
            'uav_queue_arrival_steps_mean': 20.0,
            'sat_drop_ratio': 0.020,
        },
        cfg,
    )
    _update_checkpoint_eval_state(
        state,
        {
            'gu_queue_arrival_steps_mean': 10.0,
            'uav_queue_arrival_steps_mean': 20.0,
            'sat_drop_ratio': 0.022,
        },
        cfg,
    )
    improved = _update_checkpoint_eval_state(
        state,
        {
            'gu_queue_arrival_steps_mean': 9.0,
            'uav_queue_arrival_steps_mean': 20.0,
            'sat_drop_ratio': 0.024,
        },
        cfg,
    )

    assert improved['gu_improved'] == 1.0
    assert improved['front_improved'] == 1.0
    assert improved['sat_drop_worse_streak'] == 0.0
    assert improved['early_stop_triggered'] == 0.0


def test_checkpoint_eval_summary_reports_reward_and_throughput_fields():
    cfg = SaginConfig(
        seed=7,
        T_steps=3,
        num_uav=1,
        num_gu=1,
        num_sat=1,
        users_obs_max=1,
        sats_obs_max=1,
        nbrs_obs_max=1,
        fixed_satellite_strategy=True,
        enable_bw_action=False,
        reward_mode='throughput_only',
        eta_crash=0.0,
    )

    fieldnames = _checkpoint_eval_fieldnames()
    assert 'reward_sum' in fieldnames
    assert 'throughput_access_norm_mean' in fieldnames
    assert 'throughput_backhaul_norm_mean' in fieldnames

    summary = _checkpoint_eval_summary(
        cfg,
        actor=None,
        device=torch.device('cpu'),
        episodes=1,
        episode_seed_base=123,
        exec_accel_source='policy',
        exec_bw_source='zero',
        exec_sat_source='zero',
        teacher_actor=None,
        teacher_deterministic=True,
        need_heuristic_exec=False,
        fixed_baseline=True,
    )

    assert 'reward_sum' in summary
    assert 'throughput_access_norm_mean' in summary
    assert 'throughput_backhaul_norm_mean' in summary