from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class WalkerDeltaOrbitModel:
    num_sat: int
    r_earth: float
    sat_height: float
    num_planes: int = 3
    inclination_deg: float = 53.0
    phase_factor: int = 1
    earth_rotation_rate: float = 7.2921159e-5  # rad/s
    mu: float = 3.986004418e14  # m^3/s^2

    def __post_init__(self) -> None:
        self.r_sat = self.r_earth + self.sat_height
        self.omega = math.sqrt(self.mu / (self.r_sat ** 3))
        self.num_planes = max(1, int(self.num_planes))
        self.sats_per_plane = int(math.ceil(self.num_sat / self.num_planes))
        self.inclination = math.radians(self.inclination_deg)

        plane_idx = []
        anomaly0 = []
        count = 0
        for p in range(self.num_planes):
            for s in range(self.sats_per_plane):
                if count >= self.num_sat:
                    break
                m0 = 2.0 * math.pi * s / self.sats_per_plane
                m0 += 2.0 * math.pi * self.phase_factor * p / max(1, self.num_sat)
                plane_idx.append(p)
                anomaly0.append(m0)
                count += 1

        self.plane_idx = np.array(plane_idx, dtype=np.int32)
        self.anomaly0 = np.array(anomaly0, dtype=np.float64)
        self.raan = 2.0 * math.pi * self.plane_idx / self.num_planes

    def get_states(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        m = self.anomaly0 + self.omega * t
        cos_m = np.cos(m)
        sin_m = np.sin(m)

        # Orbital plane coordinates
        x_orb = self.r_sat * cos_m
        y_orb = self.r_sat * sin_m
        z_orb = np.zeros_like(x_orb)

        v_mag = self.r_sat * self.omega
        vx_orb = -v_mag * sin_m
        vy_orb = v_mag * cos_m
        vz_orb = np.zeros_like(vx_orb)

        # Rotate by inclination around x-axis
        cos_i = math.cos(self.inclination)
        sin_i = math.sin(self.inclination)
        x_inc = x_orb
        y_inc = y_orb * cos_i
        z_inc = y_orb * sin_i
        vx_inc = vx_orb
        vy_inc = vy_orb * cos_i
        vz_inc = vy_orb * sin_i

        # Rotate by RAAN around z-axis
        cos_o = np.cos(self.raan)
        sin_o = np.sin(self.raan)
        x_eci = x_inc * cos_o - y_inc * sin_o
        y_eci = x_inc * sin_o + y_inc * cos_o
        z_eci = z_inc
        vx_eci = vx_inc * cos_o - vy_inc * sin_o
        vy_eci = vx_inc * sin_o + vy_inc * cos_o
        vz_eci = vz_inc

        if self.earth_rotation_rate != 0.0:
            theta = self.earth_rotation_rate * t
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            x = x_eci * cos_t + y_eci * sin_t
            y = -x_eci * sin_t + y_eci * cos_t
            z = z_eci
            vx = vx_eci * cos_t + vy_eci * sin_t
            vy = -vx_eci * sin_t + vy_eci * cos_t
            vz = vz_eci
        else:
            x, y, z = x_eci, y_eci, z_eci
            vx, vy, vz = vx_eci, vy_eci, vz_eci

        pos = np.stack([x, y, z], axis=1).astype(np.float32)
        vel = np.stack([vx, vy, vz], axis=1).astype(np.float32)
        return pos, vel
