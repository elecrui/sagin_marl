from __future__ import annotations

from sagin_marl.env.config import SaginConfig, ablation_flag, load_config, update_config


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


def test_ablation_flag_legacy_fallback_compatibility():
    cfg = SaginConfig()
    cfg.imitation_enabled = True
    cfg.ablation.use_imitation_loss = False
    assert ablation_flag(cfg, "use_imitation_loss", fallback_attr="imitation_enabled", default=False) is True


def test_ablation_flag_false_when_both_disabled():
    cfg = SaginConfig()
    cfg.imitation_enabled = False
    cfg.ablation.use_imitation_loss = False
    assert ablation_flag(cfg, "use_imitation_loss", fallback_attr="imitation_enabled", default=False) is False


def test_load_config_queue_init_abs_and_steps(tmp_path):
    cfg_path = tmp_path / "cfg_queue_init.yaml"
    cfg_path.write_text(
        "queue_init_gu_abs: 1234\n"
        "queue_init_uav_steps: 0.5\n"
        "queue_init_sat_steps: 2\n"
        "queue_ref_uav_per_step: 456.5\n"
        "queue_ref_sat_per_step: 789\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_path))
    assert cfg.queue_init_gu_abs == 1234
    assert cfg.queue_init_uav_steps == 0.5
    assert cfg.queue_init_sat_steps == 2
    assert cfg.queue_ref_uav_per_step == 456.5
    assert cfg.queue_ref_sat_per_step == 789


def test_load_config_queue_max_steps_uses_layer_refs(tmp_path):
    cfg_path = tmp_path / "cfg_queue_max_steps.yaml"
    cfg_path.write_text(
        "num_gu: 20\n"
        "num_uav: 3\n"
        "num_sat: 144\n"
        "queue_max_gu_steps: 40\n"
        "queue_max_uav_steps: 80\n"
        "queue_max_sat_steps: 120\n"
        "queue_ref_gu_per_step: 2.4e7\n"
        "queue_ref_uav_per_step: 2.4e7\n"
        "queue_ref_sat_per_step: 2.3e7\n"
        "queue_ref_sat_active_count: 4\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_path))
    assert cfg.queue_max_gu == 4.8e7
    assert cfg.queue_max_uav == 6.4e8
    assert cfg.queue_max_sat == 6.9e8


def test_update_config_queue_max_steps_falls_back_to_total_sat_count():
    cfg = update_config(
        SaginConfig(),
        {
            "num_sat": 144,
            "queue_max_sat_steps": 120,
            "queue_ref_sat_per_step": 2.16e7,
        },
    )
    assert cfg.queue_max_sat == 1.8e7
