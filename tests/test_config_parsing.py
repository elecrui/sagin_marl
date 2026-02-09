from __future__ import annotations

from sagin_marl.env.config import load_config


def test_load_config_coerces_numeric_strings(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "noise_density: 4e-21\n"
        "sat_cpu_freq: 1e10\n"
        "early_stop_enabled: true\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_path))
    assert isinstance(cfg.noise_density, float)
    assert isinstance(cfg.sat_cpu_freq, float)
    assert cfg.early_stop_enabled is True


def test_load_config_nested_ablation_flags(tmp_path):
    cfg_path = tmp_path / "cfg_ablation.yaml"
    cfg_path.write_text(
        "ablation:\n"
        "  use_imitation_loss: true\n"
        "  use_curriculum_spawn: false\n"
        "ablation_flags:\n"
        "  use_heuristic_mask: true\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_path))
    assert cfg.ablation.use_imitation_loss is True
    assert cfg.ablation.use_curriculum_spawn is False
    assert cfg.ablation.use_heuristic_mask is True
