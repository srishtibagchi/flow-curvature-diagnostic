import json

import numpy as np
import pytest

from flow_curvature import (
    GaussianPair, gaussian_velocity, integrate, sample_endpoints, trajectory_diagnostics,
)
from flow_curvature.coupling import couple
from flow_curvature.experiment import ExperimentConfig, run_experiment


def test_population_ot_is_straight_but_independent_flow_is_curved():
    pair = GaussianPair()
    source, _ = sample_endpoints(pair, 128, seed=7)
    times = np.linspace(0, 1, 201)
    ot = integrate(gaussian_velocity(pair, "ot"), source, times)
    independent = integrate(gaussian_velocity(pair, "independent"), source, times)
    expected = (1 - times[:, None, None]) * source + times[:, None, None] * pair.transport(source)
    np.testing.assert_allclose(ot, expected, atol=1e-12)
    ot_metrics = trajectory_diagnostics(ot, times)
    independent_metrics = trajectory_diagnostics(independent, times)
    assert ot_metrics["bending_energy"].max() < 1e-16
    assert independent_metrics["bending_energy"].mean() > 0.01
    assert independent_metrics["length_excess"].mean() > 0.01
    # Independent Gaussian marginal trajectories also have known endpoints.
    np.testing.assert_allclose(independent[-1], pair.transport(source), atol=1e-7)


def test_curvature_distinguishes_turning_from_tangential_acceleration():
    times = np.linspace(0, 1, 1001)
    parabola_in_time = np.stack((times + times**2, np.zeros_like(times)), axis=-1)[:, None]
    straight = trajectory_diagnostics(parabola_in_time, times)
    assert straight["bending_energy"][0] < 1e-18
    assert straight["acceleration_energy"][0] > 3.9
    circle = np.stack((np.cos(times), np.sin(times)), axis=-1)[:, None]
    curved = trajectory_diagnostics(circle, times)
    np.testing.assert_allclose(curved["mean_curvature"], 1, atol=1e-6)
    np.testing.assert_allclose(curved["bending_energy"], 1, atol=0.003)


def test_stationary_and_invalid_inputs_are_explicit():
    times = np.linspace(0, 1, 11)
    metrics = trajectory_diagnostics(np.ones((11, 2, 2)), times)
    np.testing.assert_array_equal(metrics["stationary_fraction"], 1)
    np.testing.assert_array_equal(metrics["bending_energy"], 0)
    with pytest.raises(ValueError, match="uniform"):
        trajectory_diagnostics(np.ones((3, 2, 2)), np.array([0, 0.1, 1]))
    with pytest.raises(ValueError, match="nonfinite"):
        integrate(lambda t, x: x * np.nan, np.ones((2, 2)), times)
    with pytest.raises(ValueError, match="Unknown"):
        gaussian_velocity(GaussianPair(), "invalid")


def test_ot_assignment_reduces_cost_and_preserves_marginals():
    source = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    target = source[::-1].copy()
    x0, x1 = couple(source, target, "ot", np.random.default_rng(42))
    np.testing.assert_array_equal(x0, x1)
    np.testing.assert_array_equal(np.sort(x1[:, 0]), np.sort(target[:, 0]))
    a = couple(source, target, "independent", np.random.default_rng(42))[1]
    b = couple(source, target, "independent", np.random.default_rng(42))[1]
    np.testing.assert_array_equal(a, b)


def test_grid_refinement_preserves_independent_curvature():
    initial, _ = sample_endpoints(GaussianPair(), 64, 42)
    energies = []
    for steps in (200, 400):
        times = np.linspace(0, 1, steps + 1)
        paths = integrate(gaussian_velocity(GaussianPair(), "independent"), initial, times)
        energies.append(trajectory_diagnostics(paths, times)["bending_energy"].mean())
    assert abs(energies[1] - energies[0]) / energies[1] < 0.02


def test_runner_persists_reproducible_results(tmp_path):
    config = ExperimentConfig(n_train=32, n_eval=16, integration_steps=30)
    first = run_experiment(config, tmp_path / "first", tmp_path / "data")
    second = run_experiment(config, tmp_path / "second", tmp_path / "data")
    assert first == second
    saved = json.loads((tmp_path / "first" / "summary.json").read_text())
    assert saved == first
    with np.load(tmp_path / "first" / "trajectories.npz") as paths:
        assert paths["analytic_ot"].shape == (31, 16, 2)
        assert paths["times"].shape == (31,)
    with np.load(tmp_path / "first" / "analytic_ot_metrics.npz") as metrics:
        assert metrics["bending_energy"].shape == (16,)
    with pytest.raises(FileExistsError):
        run_experiment(config, tmp_path / "first", tmp_path / "data")
