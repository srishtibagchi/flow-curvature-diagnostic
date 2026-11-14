"""Run an analytic benchmark and optional learned flows, persisting all inputs."""

import argparse
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform

import numpy as np
from scipy.linalg import sqrtm

from .data import GaussianPair, sample_endpoints
from .diagnostics import summarize, trajectory_diagnostics
from .fields import Coupling, gaussian_velocity
from .integration import integrate


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    n_train: int = 4096
    n_eval: int = 256
    integration_steps: int = 200
    train_steps: int = 0
    batch_size: int = 128
    hidden_width: int = 64
    learning_rate: float = 1e-3

    def __post_init__(self) -> None:
        if self.seed < 0 or min(self.n_train, self.n_eval) < 2:
            raise ValueError("Use a nonnegative seed and at least two train/evaluation points.")
        if self.integration_steps < 2 or self.train_steps < 0:
            raise ValueError("Need at least two integration steps and nonnegative train_steps.")
        if min(self.batch_size, self.hidden_width) < 1:
            raise ValueError("batch_size and hidden_width must be positive.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")


def run_experiment(
    config: ExperimentConfig,
    output_dir: str | Path = "results/baseline",
    data_dir: str | Path = "data/processed",
) -> dict:
    """Save data, paths, per-particle metrics, summary, and optional checkpoints.

    Refuse to overwrite an existing run. Distinct directories are required for
    different seeds/configurations; both methods share held-out initial points.
    """
    torch_module = None
    if config.train_steps:
        try:
            import torch
        except ModuleNotFoundError as error:
            if error.name != "torch":
                raise
            raise ModuleNotFoundError(
                "Neural runs require: uv sync --locked --extra neural"
            ) from error
        torch_module = torch
    output_dir, data_dir = Path(output_dir), Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    data_dir.mkdir(parents=True, exist_ok=True)
    pair = GaussianPair()
    seeds = np.random.SeedSequence(config.seed).spawn(3)
    train_seed, eval_seed, model_seed = [int(s.generate_state(1)[0]) for s in seeds]
    source, target = sample_endpoints(pair, config.n_train, train_seed)
    initial, _ = sample_endpoints(pair, config.n_eval, eval_seed)
    data_path = data_dir / f"gaussian-seed{config.seed}-train{config.n_train}-eval{config.n_eval}.npz"
    np.savez_compressed(
        data_path, source=source, target=target, initial=initial, allow_pickle=False
    )
    times = np.linspace(0, 1, config.integration_steps + 1)
    trajectories = {}
    summary = {}
    couplings: tuple[Coupling, ...] = ("independent", "ot")
    for coupling in couplings:
        name = f"analytic_{coupling}"
        trajectories[name] = integrate(gaussian_velocity(pair, coupling), initial, times)
    if torch_module is not None:
        from .neural import TrainingConfig, as_velocity, train_velocity

        training = TrainingConfig(
            steps=config.train_steps, batch_size=config.batch_size,
            hidden_width=config.hidden_width, learning_rate=config.learning_rate,
            seed=model_seed,
        )
        for coupling in couplings:
            model, losses = train_velocity(source, target, coupling, training)
            name = f"learned_{coupling}"
            trajectories[name] = integrate(as_velocity(model), initial, times)
            np.save(output_dir / f"{name}_loss.npy", losses, allow_pickle=False)
            torch_module.save(model.state_dict(), output_dir / f"{name}.pt")
    for name, paths in trajectories.items():
        metrics = trajectory_diagnostics(paths, times)
        np.savez_compressed(
            output_dir / f"{name}_metrics.npz", allow_pickle=False, **metrics
        )
        summary[name] = summarize(metrics)
        mean_error = np.linalg.norm(paths[-1].mean(axis=0) - np.asarray(pair.target_mean))
        covariance_error = np.linalg.norm(np.cov(paths[-1].T) - pair.target_covariance)
        # Gaussian moment W2 is a quality check, not a full distribution metric.
        empirical_cov = np.cov(paths[-1].T)
        target_root = pair.transport_matrix
        cross_root = sqrtm(target_root @ empirical_cov @ target_root)
        moment_w2_squared = mean_error**2 + np.trace(
            empirical_cov + pair.target_covariance - 2 * cross_root
        )
        summary[name]["endpoint"] = {
            "mean_error": float(mean_error),
            "covariance_frobenius_error": float(covariance_error),
            "gaussian_moment_w2": float(np.sqrt(max(float(np.real(moment_w2_squared)), 0))),
        }
    np.savez_compressed(
        output_dir / "trajectories.npz", times=times, allow_pickle=False, **trajectories
    )
    packages = {}
    for package in ("flow-curvature-diagnostic", "numpy", "scipy", "matplotlib", "torch"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    report = {
        "config": asdict(config),
        "dataset": {
            "target_mean": list(pair.target_mean),
            "target_scales": list(pair.target_stds),
            "angle": pair.rotation_angle,
        },
        "seeds": {"train": train_seed, "evaluation": eval_seed, "model": model_seed},
        "environment": {"python": platform.python_version(), "packages": packages},
        "data_path": str(data_path.resolve()), "summary": summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/baseline"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-eval", type=int, default=256)
    parser.add_argument("--integration-steps", type=int, default=200)
    parser.add_argument("--train-steps", type=int, default=0, help="0 runs analytic fields only")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = vars(parser.parse_args())
    output_dir = args.pop("output_dir")
    data_dir = args.pop("data_dir")
    report = run_experiment(ExperimentConfig(**args), output_dir, data_dir)
    print(json.dumps(report["summary"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
