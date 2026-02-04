from __future__ import annotations

import os

import scripts.evaluate as evaluate_script
import scripts.render_episode as render_script
import scripts.train as train_script


def test_resolve_log_dir_with_run_dir():
    assert train_script._resolve_log_dir("runs/base", "runs/custom", None) == "runs/custom"


def test_resolve_log_dir_with_run_id(monkeypatch):
    monkeypatch.setattr(train_script.time, "strftime", lambda fmt: "20240101_000000")
    assert train_script._resolve_log_dir("runs/base", None, "auto") == os.path.join("runs/base", "20240101_000000")
    assert train_script._resolve_log_dir("runs/base", None, "exp1") == os.path.join("runs/base", "exp1")


def test_resolve_eval_paths_with_run_dir():
    checkpoint, out, tb_dir = evaluate_script._resolve_eval_paths("runs/x", None, None, None, "none")
    assert checkpoint == os.path.join("runs/x", "actor.pt")
    assert out == os.path.join("runs/x", "eval_trained.csv")
    assert tb_dir == os.path.join("runs/x", "eval_tb")


def test_resolve_eval_paths_baseline_filename():
    checkpoint, out, tb_dir = evaluate_script._resolve_eval_paths("runs/x", None, None, None, "zero_accel")
    assert checkpoint == os.path.join("runs/x", "actor.pt")
    assert out == os.path.join("runs/x", "eval_baseline.csv")
    assert tb_dir == os.path.join("runs/x", "eval_tb")


def test_resolve_eval_paths_custom_out():
    checkpoint, out, tb_dir = evaluate_script._resolve_eval_paths(
        None, None, "runs/y/eval.csv", None, "none"
    )
    assert checkpoint == "runs/phase1/actor.pt"
    assert out == "runs/y/eval.csv"
    assert tb_dir == os.path.join("runs/y", "eval_tb")


def test_resolve_render_paths_with_run_dir():
    checkpoint, out = render_script._resolve_render_paths("runs/x", None, None)
    assert checkpoint == os.path.join("runs/x", "actor.pt")
    assert out == os.path.join("runs/x", "episode.gif")


def test_resolve_render_paths_custom():
    checkpoint, out = render_script._resolve_render_paths(None, "c.pt", "o.gif")
    assert checkpoint == "c.pt"
    assert out == "o.gif"
