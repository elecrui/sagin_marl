from __future__ import annotations

import numpy as np


def los_probability(phi_deg: np.ndarray, a: float, b: float) -> np.ndarray:
    # Probabilistic LoS model (phi in degrees)
    return 1.0 / (1.0 + a * np.exp(-b * (phi_deg - a)))


def pathloss_db(d: np.ndarray, phi_rad: np.ndarray, cfg) -> np.ndarray:
    d = np.maximum(d, 1e-9)
    pl_base = cfg.pathloss_const_db + 20.0 * np.log10(cfg.carrier_freq / 1e9)
    pl_los = pl_base + cfg.xi_los + 20.0 * np.log10(d)
    pl_nlos = pl_base + cfg.xi_nlos + 20.0 * np.log10(d)
    if getattr(cfg, "pathloss_mode", "prob_los") == "free_space":
        return pl_los
    phi_deg = np.degrees(phi_rad)
    p_los = los_probability(phi_deg, cfg.los_a, cfg.los_b)
    return p_los * pl_los + (1.0 - p_los) * pl_nlos


def rician_power_gain(K: float, size, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    K = max(float(K), 0.0)
    s = np.sqrt(K / (K + 1.0))
    sigma = np.sqrt(1.0 / (2.0 * (K + 1.0)))
    h_real = rng.normal(loc=s, scale=sigma, size=size)
    h_imag = rng.normal(loc=0.0, scale=sigma, size=size)
    return h_real**2 + h_imag**2


def atmospheric_loss_db(theta_rad: np.ndarray, base_loss_db: float) -> np.ndarray:
    sin_el = np.maximum(np.sin(theta_rad), 1e-3)
    return base_loss_db / sin_el


def doppler_attenuation(nu: np.ndarray, subcarrier_spacing: float) -> np.ndarray:
    if subcarrier_spacing <= 0:
        return np.ones_like(nu, dtype=np.float32)
    return np.sinc(nu / subcarrier_spacing) ** 2


def snr_linear(
    power: float,
    gain: np.ndarray,
    noise_density: float,
    bandwidth: float,
    interference: float | np.ndarray = 0.0,
) -> np.ndarray:
    return power * gain / (noise_density * bandwidth + interference + 1e-12)


def spectral_efficiency(snr: np.ndarray) -> np.ndarray:
    return np.log2(1.0 + snr)
