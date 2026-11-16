"""Explicit, resumable CIFAR smoke sweeps; never downloads datasets or weights."""

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np
from scipy.spatial.distance import pdist
import torch

from .diagnostics import trajectory_diagnostics
from .dit import DiTConfig, TinyDiT, as_image_velocity, train_dit
from .image_data import ScalarTransform, encode_targets, fit_scalar_transform, image_tensor, write_json
from .image_experiment import endpoint_quality, image_metrics, metric_summary, resource_preflight, save_image_grid
from .integration import integrate
from .representations import NAMES, SPECS, load_representation


@dataclass(frozen=True)
class SweepConfig:
    schema_version: int = 1
    dataset_seed: int = 42
    noise_seed: int = 123
    validation_seed: int = 456
    training_seeds: tuple[int, ...] = (42, 43)
    checkpoints: tuple[int, ...] = (10, 25)
    capacities: tuple[tuple[int, int], ...] = ((32, 1), (64, 2))
    grids: tuple[int, ...] = (8, 16, 32)
    representations: tuple[str, ...] = NAMES
    n_train: int = 32
    n_validation: int = 8
    n_eval: int = 2
    n_decode: int = 1
    batch_size: int = 8
    device: str = "cpu"
    threads: int = 2
    disk_budget_gb: float = 20
    relative_tolerance: float = 0.05
    absolute_tolerance: float = 1e-6

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported sweep schema.")
        if not self.training_seeds or len(set(self.training_seeds)) != len(self.training_seeds):
            raise ValueError("Training seeds must be nonempty and unique.")
        if min((*self.training_seeds, self.dataset_seed, self.noise_seed, self.validation_seed)) < 0:
            raise ValueError("Seeds must be nonnegative.")
        if any(tuple(sorted(set(values))) != tuple(values) for values in (self.checkpoints, self.grids)):
            raise ValueError("Checkpoints and grids must be strictly increasing.")
        if not self.checkpoints or min(self.checkpoints) < 1 or len(self.grids) < 3 or min(self.grids) < 2:
            raise ValueError("Need positive checkpoints and at least three grids.")
        if any(self.grids[-1] % grid for grid in self.grids):
            raise ValueError("Measurement grids must divide the finest grid.")
        if any(finer % coarser for coarser, finer in zip(self.grids, self.grids[1:])):
            raise ValueError("Successive measurement grids must be nested integer refinements.")
        if (
            min(self.n_train, self.n_validation, self.n_eval) < 2
            or self.n_train + self.n_validation > 50000 or self.n_validation > 10000
            or not 1 <= self.n_decode <= self.n_eval
            or min(self.batch_size, self.threads) < 1 or self.disk_budget_gb <= 0
            or not np.isfinite(self.disk_budget_gb)
            or not 0 < self.relative_tolerance < 1 or not 0 < self.absolute_tolerance < 1
        ):
            raise ValueError("Invalid subset, evaluation or resource limits.")
        if not self.capacities or len(set(self.capacities)) != len(self.capacities):
            raise ValueError("Capacities must be unique.")
        for width, depth in self.capacities:
            DiTConfig((3, 256, 256), 16, width=width, depth=depth)
        if not self.representations or len(set(self.representations)) != len(self.representations) or any(
            name not in NAMES for name in self.representations
        ):
            raise ValueError("Unknown or duplicate representation.")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("Device must be cpu or cuda.")

    @classmethod
    def from_dict(cls, values: dict):
        values = dict(values)
        for key in ("training_seeds", "checkpoints", "grids", "representations"):
            if key in values:
                values[key] = tuple(values[key])
        if "capacities" in values:
            values["capacities"] = tuple(tuple(item) for item in values["capacities"])
        return cls(**values)


def directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.exists() else 0


def work_plan(config: SweepConfig) -> dict:
    streams = len(config.representations) * 2 * len(config.capacities) * len(config.training_seeds)
    checkpoints = streams * len(config.checkpoints)
    steps = config.grids[-1]
    return {
        "schema_version": 1, "config": asdict(config),
        "training_streams": streams, "evaluated_checkpoints": checkpoints,
        "total_training_updates": streams * config.checkpoints[-1],
        "native_grid_evaluations": checkpoints * len(config.grids),
        "decoded_path_frames": checkpoints * config.n_decode * (steps + 1),
        "decoded_control_frames": checkpoints * config.n_decode * (steps + 1),
        "neural_decode_frames": sum(
            name != "pixels" for name in config.representations
        ) * 2 * len(config.capacities) * len(config.training_seeds) * len(config.checkpoints) * (
            2 * config.n_decode * (steps + 1) + config.n_eval
        ),
        "note": "Pixel identity requires no neural decoder. Coarser measurement grids reuse finest frames.",
    }


def budget_guard(config: SweepConfig, root: Path, cache: Path) -> dict:
    # Account for all retained results, not just this sweep's output.
    used = directory_bytes(cache) + directory_bytes(root.parent)
    dimensions = {**{s.name: int(np.prod(s.shape)) for s in SPECS}, "pixels": 3 * 256**2}
    largest = max(dimensions[name] for name in config.representations)
    trajectory_workspace = (config.grids[-1] + 1) * (
        config.n_eval * largest + config.n_decode * 3 * 256**2
    ) * 4
    parameter_bytes = 0
    largest_training_model = 0
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        for name in config.representations:
            spec = next((s for s in SPECS if s.name == name), None)
            shape, patch = (spec.shape, spec.patch_size) if spec else ((3, 256, 256), 16)
            for width, depth in config.capacities:
                model = TinyDiT(DiTConfig(shape, patch, width=width, depth=depth))
                model_bytes = sum(p.numel() for p in model.parameters()) * 4
                parameter_bytes += model_bytes * 3
                largest_training_model = max(largest_training_model, model_bytes * 4)
    checkpoints_bytes = parameter_bytes * 2 * len(config.training_seeds) * len(config.checkpoints)
    cache_reserve = (config.n_train + 2 * config.n_validation) * (
        3 * 256**2 + sum(dimensions[n] for n in config.representations)
    ) * 4
    reserve = trajectory_workspace * 3 + checkpoints_bytes + cache_reserve + 100_000_000
    if used + reserve > config.disk_budget_gb * 1e9:
        raise OSError(f"Retained {used} + reserved {reserve} bytes exceed cumulative budget; no files deleted.")
    minimum_ram = (
        (4 * 1024**3 if any(name != "pixels" for name in config.representations) else 0)
        + trajectory_workspace * 3 + largest_training_model
    )
    observed = resource_preflight(root, reserve, minimum_ram)
    return {"retained_bytes": used, "reserved_bytes": reserve, **observed}


def prepare_splits(config: SweepConfig, cache: Path) -> tuple[dict, dict]:
    from torchvision.datasets import CIFAR10

    identity = {
        "dataset": "CIFAR-10", "seed": config.dataset_seed, "train": config.n_train,
        "validation": config.n_validation, "render": "RGB PIL bicubic 256; float32 [0,1]",
        "split_policy": "disjoint train/validation from training split; test reserved",
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    directory = cache / f"sweep-splits-{digest}"
    if (directory / "metadata.json").exists():
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata["identity"] != identity:
            raise ValueError("Split identity mismatch.")
        return {
            key: np.load(directory / f"{key}.npy", mmap_mode="r")
            for key in ("train", "validation", "test")
        }, metadata
    directory.mkdir(parents=True, exist_ok=True)
    train = CIFAR10(cache / "raw", train=True, download=False)
    test = CIFAR10(cache / "raw", train=False, download=False)
    rng = np.random.default_rng(config.dataset_seed)
    indices = rng.choice(50000, config.n_train + config.n_validation, replace=False)
    selected = {
        "train": indices[:config.n_train], "validation": indices[config.n_train:],
        "test": rng.choice(10000, config.n_validation, replace=False),
    }
    for split, ids in selected.items():
        path = directory / f"{split}.npy"
        images = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(ids), 3, 256, 256))
        dataset = test if split == "test" else train
        for i, index in enumerate(ids):
            images[i] = image_tensor(dataset[int(index)][0], 256)[0].numpy()
        images.flush()
        del images
    metadata = {"identity": identity, "indices": {key: values.tolist() for key, values in selected.items()}}
    write_json(directory / "metadata.json", metadata)
    return {key: np.load(directory / f"{key}.npy", mmap_mode="r") for key in selected}, metadata


def comparison_changes(before: dict, after: dict, config: SweepConfig) -> dict:
    result = {}
    for name in ("bending_energy", "normalized_bending", "relative_length_excess", "stationary_fraction"):
        a, b = before[name], after[name]
        valid = np.isfinite(a) & np.isfinite(b)
        absolute = np.abs(a[valid] - b[valid])
        allowed = config.absolute_tolerance + config.relative_tolerance * np.abs(b[valid])
        result[name] = {
            "passed": bool(valid.all() and np.all(absolute <= allowed)),
            "valid_particles": int(valid.sum()), "particles": len(a),
            "max_absolute_change": float(absolute.max()) if len(absolute) else None,
            "max_relative_change": float((absolute / np.maximum(np.abs(b[valid]), config.absolute_tolerance)).max()) if len(absolute) else None,
        }
    return result


def grid_status(changes: list[dict]) -> str:
    passed = [all(item["passed"] for item in change.values()) for change in changes]
    return "two_successive_checks_passed" if len(passed) >= 2 and all(passed[-2:]) else "unresolved_at_max_grid"


def evaluate_checkpoint(
    state: dict, transform: ScalarTransform, config: SweepConfig, representation,
    initial: np.ndarray, directory: Path, validation: np.ndarray, test_images: np.ndarray,
    bandwidth: float,
) -> dict:
    model = TinyDiT(DiTConfig(**state["config"])).to(config.device).eval()
    model.load_state_dict(state["model"])
    velocity = as_image_velocity(model)
    times = np.linspace(0, 1, config.grids[-1] + 1)
    before = time.perf_counter()
    finest = np.stack([
        integrate(velocity, particle.reshape(1, -1), times)[:, 0].reshape(len(times), *model.config.shape)
        for particle in initial
    ], axis=1)
    integration_seconds = time.perf_counter() - before
    before = time.perf_counter()
    native = transform.invert(finest)
    extra_integration_seconds = 0.0
    common_times = times[::config.grids[-1] // config.grids[0]]
    native_metrics, solver_metrics, position_errors = {}, {}, {}
    for grid in config.grids:
        stride = config.grids[-1] // grid
        t = times[::stride]
        paths = native[::stride]
        native_metrics[str(grid)] = image_metrics(paths, t)
        if grid == config.grids[-1]:
            solver_metrics[str(grid)] = image_metrics(native[::config.grids[-1] // config.grids[0]], common_times)
            position_errors[str(grid)] = 0.0
        else:
            start = time.perf_counter()
            coarse = np.stack([
                integrate(velocity, point.reshape(1, -1), t)[:, 0].reshape(len(t), *model.config.shape)
                for point in initial
            ], axis=1)
            elapsed = time.perf_counter() - start
            integration_seconds += elapsed
            extra_integration_seconds += elapsed
            coarse = transform.invert(coarse)
            position_errors[str(grid)] = float(np.sqrt(np.square(coarse.astype(np.float64) - paths).mean()))
            solver_metrics[str(grid)] = image_metrics(coarse[::grid // config.grids[0]], common_times)
    diagnostics_seconds = time.perf_counter() - before - extra_integration_seconds
    before = time.perf_counter()
    decoded, control = [], []
    for i in range(config.n_decode):
        images, straight = [], []
        for j, t in enumerate(times):
            point = torch.from_numpy(native[j, i:i+1].copy())
            images.append(representation.decode(point).numpy()[0])
            linear = np.asarray((1-t) * native[0, i:i+1] + t * native[-1, i:i+1], dtype=np.float32)
            straight.append(representation.decode(torch.from_numpy(linear.copy())).numpy()[0])
        decoded.append(np.stack(images))
        control.append(np.stack(straight))
    decoded = np.stack(decoded, axis=1)
    control = np.stack(control, axis=1)
    endpoints = np.concatenate([
        representation.decode(torch.from_numpy(native[-1, i:i+1].copy())).numpy()
        for i in range(config.n_eval)
    ])
    decode_seconds = time.perf_counter() - before
    before = time.perf_counter()
    pixel_metrics = {
        str(grid): image_metrics(decoded[::config.grids[-1] // grid], times[::config.grids[-1] // grid])
        for grid in config.grids
    }
    controls = image_metrics(control, times)
    directory.mkdir(parents=True, exist_ok=True)
    for space, metrics in (("native", native_metrics), ("solver", solver_metrics), ("pixel", pixel_metrics)):
        for grid, values in metrics.items():
            np.savez_compressed(directory / f"{space}-{grid}.npz", **values)
    np.savez_compressed(directory / "control.npz", **controls)
    save_image_grid(decoded[:, 0][::max(1, config.grids[-1] // 8)], directory / "path.png")
    save_image_grid(endpoints, directory / "endpoints.png")
    changes = {}
    for space, metrics in (("native_measurement", native_metrics), ("native_solver", solver_metrics), ("pixel_measurement", pixel_metrics)):
        sequence = [metrics[str(grid)] for grid in config.grids]
        pairs = [comparison_changes(a, b, config) for a, b in zip(sequence, sequence[1:])]
        changes[space] = {"status": grid_status(pairs), "successive_changes": pairs}
    return {
        "completed_updates": state["completed_updates"],
        "validation_history": state["validation_history"],
        "training_loss": state["losses"],
        "native": {grid: metric_summary(values) for grid, values in native_metrics.items()},
        "pixel": {grid: metric_summary(values) for grid, values in pixel_metrics.items()},
        "control": metric_summary(controls), "grid_checks": changes,
        "solver_shared_time_rms_errors": position_errors,
        "solver_metric_measurement_steps": config.grids[0],
        "native_particles": config.n_eval, "decoded_particles": list(range(config.n_decode)),
        "pixel_solver_status": "not_checked_independently",
        "endpoint_quality_validation": endpoint_quality(endpoints, validation, bandwidth),
        "test_quality": "reserved_for_explicit_finalist_evaluation",
        "timings": {
            "integration_seconds": integration_seconds, "decode_seconds": decode_seconds,
            "diagnostics_and_io_seconds": max(0, diagnostics_seconds + time.perf_counter() - before),
        },
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "patch_features": model.config.shape[0] * model.config.patch_size**2,
        "backbone_width": model.config.width,
    }


def run_sweep(config: SweepConfig, root: Path, cache: Path, phase: str = "all", selection: dict | None = None) -> dict:
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback.")
    torch.set_num_threads(config.threads)
    if config.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    root.mkdir(parents=True, exist_ok=True)
    identity = json.loads(json.dumps(asdict(config)))
    manifest_path = root / "sweep.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["config"] != identity:
            raise ValueError("Sweep config mismatch; choose a different output directory.")
    else:
        manifest = {
            "schema_version": 1, "config": identity, "status": "in_progress", "streams": {},
            "hardware": {"platform": platform.platform(), "torch": torch.__version__, "device": config.device},
            "work_plan": work_plan(config), "setup_seconds": 0.0,
        }
    if selection is not None and (selection.get("schema_version") != 1 or not isinstance(selection.get("stream_ids"), list)):
        raise ValueError("Selection requires schema_version=1 and explicit stream_ids.")
    if selection is not None:
        allowed = {
            f"{name}-{coupling}-w{width}-d{depth}-s{seed}"
            for name in config.representations for coupling in ("independent", "ot")
            for width, depth in config.capacities for seed in config.training_seeds
        }
        unknown = set(selection["stream_ids"]) - allowed
        if unknown or not selection["stream_ids"]:
            raise ValueError(f"Unknown or empty selected stream IDs: {sorted(unknown)}")
    manifest.pop("error", None)
    manifest["status"] = "in_progress"
    manifest.setdefault("execution_provenance", []).append({
        "phase": phase, "python": platform.python_version(),
        "libraries": {
            name: version(name) for name in
            ("numpy", "scipy", "torch", "torchvision", "transformers", "diffusers")
        },
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.glob("*.py")
        },
        "precision": "float32 model inference; float64 integration and geometry",
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name() if config.device == "cuda" else None,
    })
    manifest["budget"] = budget_guard(config, root, cache)
    write_json(manifest_path, manifest)
    try:
        start = time.perf_counter()
        images, metadata = prepare_splits(config, cache)
        manifest["dataset"] = metadata
        bandwidth_count = min(config.n_train, 32)
        bandwidth_data = np.asarray(images["train"][:bandwidth_count], dtype=np.float64).reshape(bandwidth_count, -1)
        bandwidth = float(np.median(pdist(bandwidth_data, "sqeuclidean")) / bandwidth_data.shape[1])
        if bandwidth <= 0 or not np.isfinite(bandwidth):
            raise ValueError("Training images have no usable kernel bandwidth.")
        manifest["pixel_kernel_bandwidth_squared"] = bandwidth
        manifest["pixel_kernel_bandwidth_images"] = bandwidth_count
        del bandwidth_data
        manifest["setup_seconds"] += time.perf_counter() - start
        for name in config.representations:
            start = time.perf_counter()
            representation = load_representation(name, cache / "models", device=config.device)
            spec = next((asdict(s) for s in SPECS if s.name == name), {"name": name})
            all_targets = {}
            for split in ("train", "validation"):
                identity_cache = {"split": split, "dataset": metadata, "representation": spec}
                targets, _ = encode_targets(representation, images[split], cache / "sweep-latents", identity_cache)
                all_targets[split] = targets
            transform = fit_scalar_transform(all_targets["train"])
            manifest.setdefault("representation_setup_seconds", {}).setdefault(name, 0)
            manifest["representation_setup_seconds"][name] += time.perf_counter() - start
            initial = np.random.default_rng(config.noise_seed).standard_normal(
                (config.n_eval, *representation.shape), dtype=np.float32,
            )
            for width, depth in config.capacities:
                for seed in config.training_seeds:
                    fingerprints = []
                    for coupling in ("independent", "ot"):
                        stream_id = f"{name}-{coupling}-w{width}-d{depth}-s{seed}"
                        if selection is not None and stream_id not in selection["stream_ids"]:
                            manifest["streams"].setdefault(stream_id, {"status": "not_selected"})
                            continue
                        stream_dir = root / stream_id
                        stream_dir.mkdir(exist_ok=True)
                        entry = manifest["streams"].setdefault(stream_id, {})
                        entry.update(representation=name, coupling=coupling, width=width, depth=depth, seed=seed)
                        manifest["current_stream"] = stream_id
                        entry["status"] = "running"
                        write_json(manifest_path, manifest)
                        training = DiTConfig(
                            representation.shape, representation.patch_size, width=width, depth=depth,
                            steps=config.checkpoints[-1], batch_size=config.batch_size, seed=seed,
                        )
                        checkpoints_dir = stream_dir / "checkpoints"
                        existing = sorted(
                            checkpoints_dir.glob("step-*.pt"),
                            key=lambda p: int(p.stem.split("-")[-1]),
                        ) if checkpoints_dir.exists() else []
                        completed = int(existing[-1].stem.split("-")[-1]) if existing else 0
                        if phase in ("all", "train") and completed < training.steps:
                            resumed = torch.load(existing[-1], weights_only=True, map_location=config.device) if existing else None
                            start = time.perf_counter()
                            model, losses, state = train_dit(
                                all_targets["train"], transform, coupling, training,
                                device=config.device, checkpoint_dir=checkpoints_dir,
                                checkpoint_steps=config.checkpoints, resume=resumed,
                                validation=all_targets["validation"], validation_seed=config.validation_seed,
                            )
                            entry["training_wall_seconds"] = entry.get("training_wall_seconds", 0) + time.perf_counter() - start
                            entry["training_timings"] = state["timings"]
                            del model, losses, state, resumed
                        entry.setdefault("evaluations", {})
                        for update in config.checkpoints:
                            checkpoint = checkpoints_dir / f"step-{update}.pt"
                            if not checkpoint.exists():
                                raise FileNotFoundError(
                                    f"Missing checkpoint: {checkpoint}. An earlier training prefix "
                                    "cannot be reconstructed from a later snapshot; use a fresh output directory."
                                )
                            state = torch.load(checkpoint, weights_only=True, map_location="cpu")
                            if update == config.checkpoints[-1]:
                                fingerprints.append((state["initialization_sha256"], state["raw_batches_sha256"]))
                            evaluation_dir = stream_dir / f"evaluation-{update}"
                            saved = evaluation_dir / "summary.json"
                            if phase in ("all", "evaluate"):
                                if saved.exists():
                                    result = json.loads(saved.read_text())
                                else:
                                    result = evaluate_checkpoint(
                                        state, transform, config, representation, initial,
                                        evaluation_dir, images["validation"], images["test"], bandwidth,
                                    )
                                    result.update(
                                        representation=name, coupling=coupling, width=width, depth=depth,
                                        seed=seed, checkpoint=f"{stream_id}/checkpoints/step-{update}.pt",
                                    )
                                    write_json(saved, result)
                                entry["evaluations"][str(update)] = f"{stream_id}/evaluation-{update}/summary.json"
                                write_json(manifest_path, manifest)
                            del state
                        expected_updates = {str(update) for update in config.checkpoints}
                        entry["status"] = (
                            "complete" if set(entry["evaluations"]) == expected_updates
                            else "trained_not_evaluated"
                        )
                        write_json(manifest_path, manifest)
                        print(f"{entry['status']}: {stream_id}", flush=True)
                        gc.collect()
                    if len(fingerprints) == 2 and fingerprints[0] != fingerprints[1]:
                        raise AssertionError("Independent/OT training fingerprints differ.")
            del representation, all_targets
            gc.collect()
        manifest["status"] = "complete" if all(
            entry["status"] in ("complete", "not_selected") for entry in manifest["streams"].values()
        ) else "trained_not_evaluated"
        manifest.pop("current_stream", None)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        if "current_stream" in manifest:
            manifest["streams"][manifest["current_stream"]]["status"] = "failed"
        raise
    finally:
        manifest["execution_provenance"][-1]["peak_process_rss_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        manifest.pop("peak_process_rss_bytes", None)
        write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cifar-sweeps-smoke"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--phase", choices=("plan", "train", "evaluate", "all", "aggregate"), default="plan")
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    config = SweepConfig.from_dict(json.loads(args.config.read_text())) if args.config else SweepConfig()
    if args.phase == "plan":
        print(json.dumps({**work_plan(config), "resources": budget_guard(config, args.output_dir, args.cache_dir)}, indent=2))
    elif args.phase == "aggregate":
        from .sweep_results import summarize_sweep
        summarize_sweep(args.output_dir)
    else:
        selection = json.loads(args.selection.read_text()) if args.selection else None
        run_sweep(config, args.output_dir, args.cache_dir, args.phase, selection)
        from .sweep_results import summarize_sweep
        summarize_sweep(args.output_dir)


if __name__ == "__main__":
    main()
