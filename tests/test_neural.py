import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flow_curvature.data import GaussianPair, sample_endpoints
from flow_curvature.experiment import ExperimentConfig, run_experiment
from flow_curvature.neural import TrainingConfig, as_velocity, train_velocity


@pytest.mark.parametrize("coupling", ["independent", "ot"])
def test_training_is_repeatable_and_keeps_rng_state(coupling):
    source, target = sample_endpoints(GaussianPair(), 32, seed=1)
    config = TrainingConfig(steps=5, batch_size=8, hidden_width=8)
    state = torch.random.get_rng_state().clone()
    first, first_losses = train_velocity(source, target, coupling, config)
    second, second_losses = train_velocity(source, target, coupling, config)
    assert torch.equal(state, torch.random.get_rng_state())
    np.testing.assert_array_equal(first_losses, second_losses)
    np.testing.assert_array_equal(
        as_velocity(first)(0.5, source), as_velocity(second)(0.5, source)
    )
    assert np.all(np.isfinite(first_losses))


def test_learned_runner_saves_loadable_models(tmp_path):
    config = ExperimentConfig(
        n_train=32, n_eval=8, integration_steps=10,
        train_steps=3, batch_size=8, hidden_width=8,
    )
    report = run_experiment(config, tmp_path / "run", tmp_path / "data")
    for coupling in ("independent", "ot"):
        name = f"learned_{coupling}"
        assert name in report["summary"]
        checkpoint = torch.load(tmp_path / "run" / f"{name}.pt", weights_only=True)
        assert "net.0.weight" in checkpoint
        assert np.load(tmp_path / "run" / f"{name}_loss.npy").shape == (3,)
