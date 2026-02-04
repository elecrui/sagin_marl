from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import re

import math

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


@dataclass
class SaginConfig:
    seed: int = 42

    # Map and timing
    map_size: float = 1000.0
    tau0: float = 1.0
    T_steps: int = 200

    # Entity counts
    num_uav: int = 3
    num_gu: int = 20
    num_sat: int = 6

    # Observation limits
    users_obs_max: int = 20
    sats_obs_max: int = 6
    nbrs_obs_max: int = 4
    visible_sats_max: int | None = None
    visible_sats_min: int | None = None

    # Geometry
    uav_height: float = 100.0
    sat_height: float = 500_000.0
    r_earth: float = 6_371_000.0
    theta_min_deg: float = 10.0
    ref_lat_deg: float = 0.0
    ref_lon_deg: float = 0.0

    # Walker-Delta constellation
    walker_num_planes: int = 3
    walker_inclination_deg: float = 53.0
    walker_phase_factor: int = 1
    earth_rotation_rate: float = 7.2921159e-5  # rad/s

    # UAV dynamics
    v_max: float = 30.0
    a_max: float = 5.0
    d_safe: float = 20.0
    boundary_mode: str = "clip"  # "clip" or "reflect"

    # Queues (bits)
    queue_max_gu: float = 5e6
    queue_max_uav: float = 1e7
    queue_max_sat: float = 5e7

    # Task arrivals
    task_arrival_rate: float = 2e5  # bits per slot (mean)
    task_arrival_poisson: bool = True

    # Communication
    b_acc: float = 5e6
    b_sat_total: float = 20e6
    gu_tx_power: float = 0.2  # Watts
    uav_tx_power: float = 1.0  # Watts
    uav_tx_gain: float = 300.0
    sat_rx_gain: float = 300.0
    noise_density: float = 4e-21  # W/Hz (thermal noise at ~290K)
    carrier_freq: float = 2e9
    speed_of_light: float = 3e8
    pathloss_const_db: float = 32.4
    los_a: float = 9.61
    los_b: float = 0.16
    xi_los: float = 1.0
    xi_nlos: float = 20.0
    pl_threshold_db: float = 140.0
    pathloss_mode: str = "prob_los"  # "prob_los" or "free_space"
    rician_K: float = 10.0
    atm_loss_enabled: bool = False
    atm_loss_db: float = 2.0
    subcarrier_spacing: float = 15e3

    # Satellite compute
    sat_cpu_freq: float = 1e10  # cycles/s
    task_cycles_per_bit: float = 1000.0  # cycles/bit

    # Doppler
    nu_max: float = 2000.0
    doppler_observed: bool = True
    doppler_atten_enabled: bool = False

    # Phase toggles
    doppler_enabled: bool = False
    energy_enabled: bool = False
    fading_enabled: bool = False
    interference_enabled: bool = False
    enable_bw_action: bool = False
    fixed_satellite_strategy: bool = True
    N_RF: int = 1
    sat_select_mode: str = "topk"
    sat_state_max: int | None = None

    # Collision avoidance (optional safety layer)
    avoidance_enabled: bool = False
    avoidance_eta: float = 100.0
    avoidance_alert_factor: float = 1.5

    # Energy placeholders
    uav_energy_init: float = 1.0
    p_fly_base: float = 0.01
    p_fly_coeff: float = 0.001
    p_comm_link: float = 0.01
    energy_model: str = "simple"  # "simple" or "rotor"
    energy_safety_enabled: bool = False
    energy_safe_threshold: float = 0.2  # fraction of init energy
    uav_opt_speed: float = 10.0

    # Rotorcraft power model params (for energy_model="rotor")
    rotor_p0: float = 79.86
    rotor_pi: float = 88.63
    rotor_u_tip: float = 120.0
    rotor_v0: float = 4.03
    rotor_d0: float = 0.6
    rotor_rho: float = 1.225
    rotor_s: float = 0.05
    rotor_area: float = 0.503

    # Action logit scales
    bw_logit_scale: float = 5.0
    sat_logit_scale: float = 5.0

    # Reward shaping
    omega_q: float = 1.0
    omega_e: float = 0.0
    eta_crash: float = 1.0
    eta_batt: float = 1.0
    eta_drop: float = 1.0
    eta_cong: float = 0.1
    queue_th_gu: float | None = None
    queue_th_uav: float | None = None
    queue_th_gu_frac: float = 0.8
    queue_th_uav_frac: float = 0.8

    # PPO defaults (hardware aware)
    buffer_size: int = 4000
    num_mini_batch: int = 32
    ppo_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    actor_hidden: int = 128
    critic_hidden: int = 128

    # Early stopping (convergence)
    early_stop_enabled: bool = True
    early_stop_min_updates: int = 20
    early_stop_window: int = 5
    early_stop_patience: int = 10
    early_stop_min_delta: float = 1e-3

    @property
    def theta_min_rad(self) -> float:
        return math.radians(self.theta_min_deg)


def load_config(path: str) -> SaginConfig:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load config files.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return update_config(SaginConfig(), data)


def update_config(cfg: SaginConfig, updates: Dict[str, Any]) -> SaginConfig:
    for key, value in updates.items():
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown config key: {key}")
        current = getattr(cfg, key)
        value = _coerce_scalar(value, current)
        if isinstance(current, bool) and isinstance(value, int):
            value = bool(value)
        if isinstance(current, int) and isinstance(value, float) and value.is_integer():
            value = int(value)
        setattr(cfg, key, value)
    return cfg


def _coerce_scalar(value: Any, current: Any) -> Any:
    if not isinstance(value, str):
        return value
    s = value.strip()
    low = s.lower()

    if isinstance(current, bool) or low in ("true", "false", "1", "0", "yes", "no", "y", "n"):
        if low in ("true", "1", "yes", "y"):
            return True
        if low in ("false", "0", "no", "n"):
            return False

    if isinstance(current, (int, float)) or current is None:
        if re.fullmatch(r"[+-]?\d+", s):
            try:
                return int(s)
            except Exception:
                pass
        try:
            return float(s)
        except Exception:
            return value

    return value
