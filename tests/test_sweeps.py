from dataclasses import replace
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from flow_curvature.dit import DiTConfig, TinyDiT, train_dit
from flow_curvature.image_data import fit_scalar_transform
from flow_curvature.image_sweeps import (
    SweepConfig, budget_guard, comparison_changes, evaluate_checkpoint, grid_status, run_sweep, work_plan,
)
from flow_curvature.representations import Pixels
from flow_curvature.sweep_results import load_results, paired_differences, seed_summaries, summarize_sweep


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_checkpoint_prefix_resume_and_validation_rng(tmp_path):
    rng = np.random.default_rng(7)
    targets = rng.random((4, 3, 16, 16), dtype=np.float32)
    validation = rng.random((3, 3, 16, 16), dtype=np.float32)
    transform = fit_scalar_transform(targets)
    config = DiTConfig((3, 16, 16), 1, width=8, depth=1, heads=2, steps=4, batch_size=2)
    _, losses, full = train_dit(
        targets, transform, "ot", config, checkpoint_dir=tmp_path / "full",
        checkpoint_steps=(2, 4), validation=validation,
    )
    prefix = torch.load(tmp_path / "full" / "step-2.pt", weights_only=True)
    _, _, short = train_dit(targets, transform, "ot", replace(config, steps=2), validation=validation)
    for key in short["model"]:
        torch.testing.assert_close(prefix["model"][key], short["model"][key], rtol=0, atol=0)
    _, resumed_losses, resumed = train_dit(
        targets, transform, "ot", config, resume=prefix, validation=validation,
        checkpoint_steps=(4,), checkpoint_dir=tmp_path / "resumed",
    )
    np.testing.assert_array_equal(losses, resumed_losses)
    for key in full["model"]:
        torch.testing.assert_close(resumed["model"][key], full["model"][key], rtol=0, atol=0)
    for key in ("raw_batches_sha256", "validation_history", "sample_rng", "pairing_rng"):
        assert resumed[key] == full[key]
    for index, values in full["optimizer"]["state"].items():
        for key, value in values.items():
            torch.testing.assert_close(value, resumed["optimizer"]["state"][index][key], rtol=0, atol=0)
    _, _, without_validation = train_dit(targets, transform, "ot", config)
    assert without_validation["raw_batches_sha256"] == full["raw_batches_sha256"]
    for key in full["model"]:
        torch.testing.assert_close(full["model"][key], without_validation["model"][key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="Incompatible"):
        train_dit(targets + 1, transform, "ot", config, resume=prefix, validation=validation)
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            train_dit(targets, transform, "ot", config, device="cuda")


def test_grid_evaluation_never_trains_and_decodes_finest_once(tmp_path):
    rng = np.random.default_rng(5)
    targets = rng.random((3, 3, 16, 16), dtype=np.float32)
    validation = rng.random((3, 3, 16, 16), dtype=np.float32)
    config = DiTConfig((3, 16, 16), 1, width=8, depth=1, heads=2, steps=2, batch_size=2)
    transform = fit_scalar_transform(targets)
    _, _, state = train_dit(targets, transform, "independent", config, validation=validation)
    original = {key: value.clone() for key, value in state["model"].items()}

    class CountingPixels(Pixels):
        calls = 0

        def decode(self, latents):
            assert latents.dtype == torch.float32
            self.calls += 1
            return latents

    representation = CountingPixels(16)
    smoke = SweepConfig(
        representations=("pixels",), grids=(2, 4, 8), checkpoints=(1, 2),
        n_train=3, n_validation=3,
    )
    initial = rng.standard_normal((2, 3, 16, 16), dtype=np.float32)
    result = evaluate_checkpoint(
        state, transform, smoke, representation, initial, tmp_path,
        validation, validation, 0.1,
    )
    assert representation.calls == 2 * 9 + smoke.n_eval
    assert set(result["native"]) == {"2", "4", "8"}
    assert result["decoded_particles"] == [0]
    assert result["pixel_solver_status"] == "not_checked_independently"
    assert result["solver_shared_time_rms_errors"]["8"] == 0
    for key in original:
        torch.testing.assert_close(original[key], state["model"][key], rtol=0, atol=0)
    for grid in smoke.grids:
        with np.load(tmp_path / f"native-{grid}.npz") as native, np.load(tmp_path / f"pixel-{grid}.npz") as pixel:
            np.testing.assert_allclose(native["bending_energy"][:1], pixel["bending_energy"])


def test_plan_convergence_status_and_model_seed_statistics(tmp_path):
    plan = work_plan(SweepConfig())
    assert plan["training_streams"] == 40
    assert plan["evaluated_checkpoints"] == 80
    config = SweepConfig()
    values = {name: np.zeros(2) for name in (
        "bending_energy", "normalized_bending", "relative_length_excess", "stationary_fraction",
    )}
    changes = comparison_changes(values, values, config)
    assert grid_status([changes, changes]) == "two_successive_checks_passed"
    assert grid_status([changes]) == "unresolved_at_max_grid"
    changed = {**values, "normalized_bending": np.ones(2)}
    assert grid_status([changes, comparison_changes(values, changed, config)]) == "unresolved_at_max_grid"
    with pytest.raises(ValueError, match="divide"):
        replace(config, grids=(8, 12, 32))
    with pytest.raises(ValueError, match="nested"):
        replace(config, grids=(6, 8, 24))
    assert load_results(tmp_path) == ({}, [])
    rows = []
    for seed in (42, 43):
        for coupling in ("independent", "ot"):
            rows.append(dict(
                representation="pixels", coupling=coupling, width=32, depth=1, seed=seed,
                update=25, grid=32, space="native",
                normalized_bending=2.0 if coupling == "ot" else 3.0,
            ))
    summaries = seed_summaries(rows)
    assert all(item["n_training_seeds"] == 2 for item in summaries)
    assert all("bootstrap_mean_interval_95" not in item for item in summaries)
    assert all(row["ot_minus_independent"] == -1 for row in paired_differences(rows))


def test_staged_runner_resume_selection_and_aggregation_offline(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    images = {
        "train": rng.random((3, 3, 16, 16), dtype=np.float32),
        "validation": rng.random((2, 3, 16, 16), dtype=np.float32),
        "test": rng.random((2, 3, 16, 16), dtype=np.float32),
    }
    monkeypatch.setattr("flow_curvature.image_sweeps.prepare_splits", lambda *args: (images, {"fixture": True}))
    monkeypatch.setattr("flow_curvature.image_sweeps.load_representation", lambda *args, **kwargs: Pixels(16))
    config = SweepConfig(
        representations=("pixels",), capacities=((8, 1),), grids=(2, 4, 8),
        checkpoints=(1, 2), n_train=3, n_validation=2, batch_size=2,
    )
    root = tmp_path / "run"
    cache = tmp_path / "cache"
    trained = run_sweep(config, root, cache, "train")
    assert trained["status"] == "trained_not_evaluated"
    result = run_sweep(config, root, cache, "evaluate")
    assert result["status"] == "complete"
    assert result["execution_provenance"][-1]["source_sha256"]["image_sweeps.py"]
    assert all(record["peak_process_rss_bytes"] > 0 for record in result["execution_provenance"])
    assert "peak_process_rss_bytes" not in result
    aggregation = summarize_sweep(root)
    assert aggregation["rows"] == 4 * 2 * 3 * 2
    _, rows = load_results(root)
    assert len(rows) == aggregation["rows"]
    runtime = json.loads((root / "runtime.json").read_text())
    assert runtime["coverage"]["evaluated_checkpoints"] == 8
    assert runtime["gpu_measured"] is False
    assert len(runtime["gpu_sensitivity_scenarios"]) == 3
    snapshot = root / "pixels-ot-w8-d1-s42/checkpoints/step-2.pt"
    unchanged = snapshot.stat().st_mtime_ns
    assert run_sweep(config, root, cache)["status"] == "complete"
    assert snapshot.stat().st_mtime_ns == unchanged
    result["status"] = "failed"
    result["error"] = {"type": "Fixture", "message": "Previous failed attempt"}
    (root / "sweep.json").write_text(json.dumps(result))
    assert "error" not in run_sweep(config, root, cache)
    with pytest.raises(ValueError, match="Unknown"):
        run_sweep(config, root, cache, selection={"schema_version": 1, "stream_ids": ["unknown"]})
    with pytest.raises(OSError, match="budget"):
        budget_guard(replace(config, disk_budget_gb=1e-8), root, cache)
    (root / "pixels-independent-w8-d1-s42/checkpoints/step-1.pt").unlink()
    with pytest.raises(FileNotFoundError, match="earlier training prefix"):
        run_sweep(config, root, cache)
