"""CPU image flow matching with explicitly approved, pinned frozen representations."""

import argparse
import csv
from dataclasses import asdict, dataclass
import gc
import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
import platform
import resource
import shutil
import time
from typing import Callable

import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist, pdist
import torch

from .diagnostics import scale_invariant_diagnostics, trajectory_diagnostics
from .dit import DiTConfig, TinyDiT, as_image_velocity, train_dit
from .image_data import (
    CIFAR_BYTES, CIFAR_MD5, CIFAR_URL, ScalarTransform, encode_targets,
    prepare_images, write_json,
)
from .integration import integrate
from .representations import NAMES, SPECS, Pixels, Representation, load_representation


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageConfig:
    seed: int = 42
    n_train: int = 256
    n_eval: int = 2
    image_size: int = 256
    train_steps: int = 200
    integration_steps: int = 8
    batch_size: int = 8
    hidden_width: int = 64
    depth: int = 2
    heads: int = 4
    learning_rate: float = 1e-3
    threads: int = 2
    disk_budget_gb: float = 20
    grid_refinement: bool = True
    representations: tuple[str, ...] = NAMES

    def __post_init__(self) -> None:
        if (
            self.seed < 0 or min(self.n_train, self.n_eval, self.integration_steps) < 2
            or min(self.train_steps, self.batch_size, self.threads, self.image_size) < 1
            or self.image_size < 16 or self.image_size % 16
            or not self.representations or len(set(self.representations)) != len(self.representations)
            or any(name not in NAMES for name in self.representations)
            or not np.isfinite(self.disk_budget_gb) or self.disk_budget_gb <= 0
        ):
            raise ValueError("Invalid image experiment configuration.")
        DiTConfig(
            (3, self.image_size, self.image_size), self.image_size // 16,
            self.hidden_width, self.depth, self.heads, self.train_steps,
            self.batch_size, self.learning_rate, self.seed,
        )
        if any(name != "pixels" for name in self.representations) and self.image_size != 256:
            raise ValueError("Released image representations require image_size=256.")


def resource_preflight(root: Path, required_disk: int = 0, minimum_ram: int = 0) -> dict:
    location = root.resolve()
    while not location.exists():
        location = location.parent
    available = None
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                break
    free_disk = shutil.disk_usage(location).free
    if free_disk < required_disk:
        raise OSError(f"Need {required_disk} free disk bytes; only {free_disk} available.")
    if minimum_ram and (available is None or available < minimum_ram):
        raise MemoryError(f"Need {minimum_ram} available RAM bytes; observed {available}.")
    return {"available_ram_bytes": available, "free_disk_bytes": free_disk}


def download_manifest(config: ImageConfig, cache_dir: Path | None = None) -> dict:
    from huggingface_hub import try_to_load_from_cache

    files = []
    for spec in SPECS:
        if spec.name not in config.representations:
            continue
        for filename, size in spec.files:
            cached = try_to_load_from_cache(
                spec.repository, filename, revision=spec.revision, cache_dir=cache_dir,
            )
            present = isinstance(cached, str) and Path(cached).stat().st_size == size
            files.append({
                "repository": spec.repository, "revision": spec.revision,
                "filename": filename, "bytes": size, "license": spec.license,
                "cached": present,
            })
    dimensions = {
        **{spec.name: int(np.prod(spec.shape)) for spec in SPECS},
        "pixels": 3 * config.image_size**2,
    }
    pixel_dimension = dimensions["pixels"]
    cache_bytes = (config.n_train + config.n_eval) * pixel_dimension * 4
    cache_bytes += config.n_train * sum(dimensions[name] for name in config.representations) * 4
    path_bytes = 2 * (config.integration_steps + 1) * config.n_eval * 4 * sum(
        dimensions[name] + (pixel_dimension if name != "pixels" else 0)
        for name in config.representations
    )
    return {
        "models": files,
        "dataset": {"url": CIFAR_URL, "bytes": CIFAR_BYTES, "md5": CIFAR_MD5},
        "maximum_payload_bytes": sum(item["bytes"] for item in files) + CIFAR_BYTES,
        "estimated_cache_and_trajectory_bytes": cache_bytes + path_bytes,
        "disk_budget_bytes": int(config.disk_budget_gb * 1e9),
        "note": "No Stage-2 pretrained DiT; local cache reuse reduces actual new downloads.",
    }


def download_assets(root: Path, config: ImageConfig, approved: bool = False) -> dict:
    if not approved:
        raise PermissionError("Dataset/model payloads require explicit manifest approval and --download.")
    from huggingface_hub import snapshot_download
    from torchvision.datasets import CIFAR10

    manifest = download_manifest(config, root / "models")
    expected = manifest["maximum_payload_bytes"] + manifest["estimated_cache_and_trajectory_bytes"]
    if expected > manifest["disk_budget_bytes"]:
        raise ValueError(f"Estimated payload/cache/paths {expected} exceed the approved disk budget.")
    manifest["resources"] = resource_preflight(root, expected, 4 * 1024**3)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "approved-download-manifest.json", manifest)
    for spec in SPECS:
        if spec.name not in config.representations:
            continue
        location = Path(snapshot_download(
            spec.repository, revision=spec.revision, cache_dir=root / "models",
            allow_patterns=[name for name, _ in spec.files], max_workers=1,
        ))
        for filename, size in spec.files:
            if (location / filename).stat().st_size != size:
                raise ValueError(f"Checkpoint file size mismatch: {spec.repository}/{filename}")
    CIFAR10.url = CIFAR_URL
    CIFAR10(root / "raw", train=True, download=True)
    return manifest


def image_metrics(paths: np.ndarray, times: np.ndarray) -> dict[str, np.ndarray]:
    if paths.ndim < 3 or paths.shape[0] != len(times) or paths.shape[1] < 1:
        raise ValueError("Expected image paths (time, particle, ...).")
    per_particle = []
    dimension = int(np.prod(paths.shape[2:]))
    for particle in range(paths.shape[1]):
        flattened = np.asarray(paths[:, particle], dtype=np.float64).reshape(len(times), 1, -1)
        metrics = trajectory_diagnostics(flattened, times)
        if any(not np.isfinite(values).all() for values in metrics.values()):
            raise FloatingPointError("Nonfinite full-dimensional trajectory diagnostic.")
        metrics.update(scale_invariant_diagnostics(metrics))
        if not np.isfinite(metrics["normalized_bending"]).all():
            raise FloatingPointError("Nonfinite scale-invariant bending energy.")
        metrics["rms_path_length"] = metrics["path_length"] / np.sqrt(dimension)
        per_particle.append(metrics)
    return {name: np.concatenate([item[name] for item in per_particle]) for name in per_particle[0]}


def metric_summary(metrics: dict[str, np.ndarray]) -> dict:
    result = {}
    for name, values in metrics.items():
        finite = values[np.isfinite(values)]
        result[name] = {
            "mean": float(finite.mean()) if finite.size else None,
            "median": float(np.median(finite)) if finite.size else None,
            "p95": float(np.quantile(finite, 0.95)) if finite.size else None,
            "undefined_count": int(values.size - finite.size),
        }
    return result


def save_image_grid(images: np.ndarray, path: Path, columns: int = 4) -> None:
    if not np.isfinite(images).all():
        raise ValueError("Cannot render nonfinite images.")
    count, _, height, width = images.shape
    columns = min(columns, count)
    canvas = Image.new("RGB", (columns * width, ((count + columns - 1) // columns) * height))
    for i, image in enumerate(images):
        pixels = (np.clip(image.transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8)
        canvas.paste(Image.fromarray(pixels), ((i % columns) * width, (i // columns) * height))
    canvas.save(path)


def decode_path(representation: Representation, native: np.ndarray) -> np.ndarray:
    decoded = []
    for point in native:
        tensor = torch.from_numpy(np.array(point[None], dtype=np.float32, copy=True))
        image = representation.decode(tensor).detach().numpy()[0]
        if not np.isfinite(image).all():
            raise FloatingPointError("Decoder returned nonfinite output.")
        decoded.append(image)
    return np.stack(decoded)


def sample_paths(
    model: TinyDiT, representation: Representation, transform: ScalarTransform,
    initial: np.ndarray, times: np.ndarray, directory: Path, image_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, dict]:
    directory.mkdir()
    native = np.lib.format.open_memmap(
        directory / "native.npy", mode="w+", dtype=np.float32,
        shape=(len(times), len(initial), *representation.shape),
    )
    pixels = native if isinstance(representation, Pixels) else (
        np.lib.format.open_memmap(
            directory / "decoded.npy", mode="w+", dtype=np.float32,
            shape=(len(times), len(initial), 3, image_size, image_size),
        )
    )
    controls = []
    velocity = as_image_velocity(model)
    for i, point in enumerate(initial):
        path = integrate(velocity, point.reshape(1, -1), times)[:, 0]
        native[:, i] = transform.invert(path).reshape(len(times), *representation.shape)
        if pixels is not native:
            pixels[:, i] = decode_path(representation, native[:, i])
        endpoints = native[[0, -1], i]
        interpolation = (
            (1 - times[:, None, None, None]) * endpoints[0]
            + times[:, None, None, None] * endpoints[1]
        )
        control = decode_path(representation, interpolation)
        controls.append(image_metrics(control[:, None], times))
    native.flush()
    pixels.flush()
    control_metrics = {key: np.concatenate([item[key] for item in controls]) for key in controls[0]}
    return native, pixels, control_metrics


def endpoint_quality(endpoints: np.ndarray, heldout: np.ndarray, bandwidth: float) -> dict:
    x = np.asarray(endpoints, dtype=np.float64).reshape(len(endpoints), -1)
    y = np.asarray(heldout, dtype=np.float64).reshape(len(heldout), -1)
    dimension = x.shape[1]
    if len(x) < 2 or len(y) < 2 or not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Two-sample quality requires positive bandwidth and at least two samples.")
    xx = np.exp(-cdist(x, x, "sqeuclidean") / dimension / (2 * bandwidth))
    yy = np.exp(-cdist(y, y, "sqeuclidean") / dimension / (2 * bandwidth))
    xy = np.exp(-cdist(x, y, "sqeuclidean") / dimension / (2 * bandwidth))
    mmd = (
        (xx.sum() - np.trace(xx)) / (len(x) * (len(x) - 1))
        + (yy.sum() - np.trace(yy)) / (len(y) * (len(y) - 1))
        - 2 * xy.mean()
    )
    return {
        "pixel_mmd2_unbiased": float(mmd),
        "mean_pairwise_rms_distance": float(np.sqrt(pdist(x, "sqeuclidean") / dimension).mean()),
        "out_of_range_fraction": float(((x < 0) | (x > 1)).mean()),
        "note": "Small-sample pixel discrepancy; negative unbiased MMD2 estimates are valid. Not FID.",
    }


def save_comparison(report: dict, directory: Path) -> None:
    from matplotlib.figure import Figure

    changes = [
        method["grid_refinement"][space][metric]["relative_change"]
        for method in report["methods"].values() if "grid_refinement" in method
        for space in ("native", "pixel")
        for metric in ("bending_energy", "normalized_bending", "stationary_fraction")
    ]
    report["measurement_status"] = (
        "grid_sensitive" if any(change > 0.05 for change in changes)
        else "representative_grid_check_passed" if changes else "grid_not_checked"
    )
    names = list(report["representations"])
    fields = [
        "representation", "coupling", "native_bending_median", "native_normalized_bending_median",
        "pixel_bending_median", "pixel_normalized_bending_median",
        "decoded_straight_control_normalized_bending_median", "pixel_mmd2_unbiased",
    ]
    with (directory / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for name in names:
            for coupling in ("independent", "ot"):
                method = report["methods"][f"{name}_{coupling}"]
                writer.writerow({
                    "representation": name, "coupling": coupling,
                    "native_bending_median": method["native"]["bending_energy"]["median"],
                    "native_normalized_bending_median": method["native"]["normalized_bending"]["median"],
                    "pixel_bending_median": method["pixel"]["bending_energy"]["median"],
                    "pixel_normalized_bending_median": method["pixel"]["normalized_bending"]["median"],
                    "decoded_straight_control_normalized_bending_median": method["decoded_straight_latent_control"]["normalized_bending"]["median"],
                    "pixel_mmd2_unbiased": method["endpoint_quality"]["pixel_mmd2_unbiased"],
                })
    figure = Figure(figsize=(12, 4.5))
    axes = figure.subplots(1, 2)
    positions = np.arange(len(names))
    for axis, space in zip(axes, ("native", "pixel")):
        for coupling, offset in (("independent", -0.18), ("ot", 0.18)):
            values = [
                report["methods"][f"{name}_{coupling}"][space]["normalized_bending"]["median"]
                for name in names
            ]
            axis.bar(positions + offset, values, width=0.36, label=coupling)
        axis.set_xticks(positions, names, rotation=20)
        axis.set_yscale("symlog", linthresh=1e-6)
        axis.set(title=f"{space}: median bending x path length", ylabel="Scale-invariant bending (symlog)")
        axis.legend()
    figure.suptitle(
        f"CPU smoke test ({report['measurement_status'].replace('_', ' ')}): "
        f"{report['config']['n_eval']} particles, "
        f"{report['config']['train_steps']} updates; not a representation ranking"
    )
    figure.tight_layout()
    figure.savefig(directory / "curvature_comparison.png", dpi=160)
    figure.clear()


def refine_grid(
    model: TinyDiT, representation: Representation, transform: ScalarTransform,
    initial: np.ndarray, coarse_native: np.ndarray, coarse_pixels: np.ndarray, times: np.ndarray,
) -> dict:
    coarse = image_metrics(coarse_native, times)
    index = int(np.argsort(coarse["bending_energy"])[len(initial) // 2])
    refined_times = np.linspace(0, 1, 2 * (len(times) - 1) + 1)
    path = integrate(as_image_velocity(model), initial[index].reshape(1, -1), refined_times)[:, 0]
    native = transform.invert(path).reshape(len(refined_times), *representation.shape)
    pixels = decode_path(representation, native)
    result = {"particle": index, "coarse_steps": len(times) - 1, "refined_steps": len(refined_times) - 1}
    for name, old, new in (
        ("native", coarse_native[:, index:index + 1], native[:, None]),
        ("pixel", coarse_pixels[:, index:index + 1], pixels[:, None]),
    ):
        before, after = image_metrics(old, times), image_metrics(new, refined_times)
        result[name] = {}
        for metric in ("bending_energy", "normalized_bending", "stationary_fraction"):
            a, b = float(before[metric][0]), float(after[metric][0])
            change = abs(a - b) / max(abs(b), 1e-8)
            result[name][metric] = {"coarse": a, "refined": b, "absolute_change": abs(a - b), "relative_change": change}
            if change > 0.05:
                LOGGER.warning("%s %s grid change exceeds 5%%: %.3g", name, metric, change)
    return result


def run_image_experiment(
    config: ImageConfig, output_dir: Path, cache_dir: Path,
    representation_loader: Callable[[str, Path, int], Representation] = load_representation,
    image_data: tuple[np.ndarray, np.ndarray, dict] | None = None,
) -> dict:
    torch.set_num_threads(config.threads)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "in_progress", "config": asdict(config), "methods": {},
        "environment": {
            "python": platform.python_version(),
            "packages": {name: version(name) for name in ("torch", "numpy", "diffusers", "transformers")},
            "source_sha256": {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in (
                    "image_experiment.py", "image_data.py", "representations.py",
                    "dit.py", "diagnostics.py", "integration.py", "coupling.py",
                )
            },
        },
        "interpretation": [
            "CPU smoke test, not converged image generation or a representation ranking.",
            "Native metrics depend on representation dimension and checkpoint normalization.",
            "Pixel paths include decoder effects and off-manifold decoded noise.",
            "Training is unconditional; all fields use linear CFM without guidance.",
            "The tiny backbone compresses RAE/pixel patch features but not VAE patch features; capacity is not equivalent.",
        ],
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, report)
    try:
        estimate = download_manifest(config, cache_dir / "models")["estimated_cache_and_trajectory_bytes"]
        if estimate > config.disk_budget_gb * 1e9:
            raise ValueError("Estimated caches/trajectories exceed the configured disk budget.")
        report["resources"] = resource_preflight(
            output_dir, estimate,
            4 * 1024**3 if any(n != "pixels" for n in config.representations) else 0,
        )
        train, heldout, metadata = image_data if image_data is not None else prepare_images(
            cache_dir, config.seed, config.n_train, config.n_eval, config.image_size,
        )
        if train.shape != (config.n_train, 3, config.image_size, config.image_size) or heldout.shape != (
            config.n_eval, 3, config.image_size, config.image_size,
        ):
            raise ValueError("Dataset shapes do not match the requested experiment.")
        report["dataset"] = metadata
        save_image_grid(heldout, output_dir / "heldout.png")
        bandwidth_images = np.asarray(train[:min(len(train), 32)], dtype=np.float64).reshape(min(len(train), 32), -1)
        bandwidth = float(np.median(pdist(bandwidth_images, "sqeuclidean")) / bandwidth_images.shape[1])
        if bandwidth <= 0 or not np.isfinite(bandwidth):
            raise ValueError("Training images have no usable kernel bandwidth.")
        report["pixel_kernel_bandwidth_squared"] = bandwidth
        del bandwidth_images
        seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(config.seed).spawn(2)]
        report["seeds"] = {"training": seeds[0], "initial_noise": seeds[1], "dataset": config.seed}
        times = np.linspace(0, 1, config.integration_steps + 1)
        np.save(output_dir / "times.npy", times, allow_pickle=False)
        for name in config.representations:
            report["current_representation"] = name
            report.pop("current_method", None)
            write_json(summary_path, report)
            resource_preflight(output_dir, 0, 4 * 1024**3 if name != "pixels" else 0)
            representation = representation_loader(name, cache_dir / "models", config.image_size)
            patch_dimension = representation.shape[0] * representation.patch_size**2
            if patch_dimension > config.hidden_width:
                LOGGER.warning(
                    "%s patch projection compresses %d features to %d; do not interpret this smoke test as a capacity-matched ranking.",
                    name, patch_dimension, config.hidden_width,
                )
            spec = next((asdict(s) for s in SPECS if s.name == name), {"name": "pixels"})
            target_identity = {"dataset": metadata, "representation": spec}
            targets, transform = encode_targets(representation, train, cache_dir / "latents", target_identity)
            reconstructions = []
            errors = []
            for image in heldout:
                tensor = torch.from_numpy(np.array(image[None], copy=True))
                reconstruction = representation.decode(representation.encode(tensor)).detach().numpy()[0]
                reconstructions.append(reconstruction)
                errors.append(float(np.square(reconstruction.astype(np.float64) - image).mean()))
            save_image_grid(np.stack(reconstructions), output_dir / f"{name}_reconstructions.png")
            report.setdefault("representations", {})[name] = {
                "checkpoint": spec, "training_transform": asdict(transform),
                "shape": list(representation.shape), "patch_size": representation.patch_size,
                "patch_feature_dimension": patch_dimension,
                "backbone_width": config.hidden_width,
                "input_projection_compresses_patch": patch_dimension > config.hidden_width,
                "reconstruction_mse": errors,
                "reconstruction_psnr_db": [float(-10 * np.log10(e)) if e > 0 else None for e in errors],
                "exact_reconstruction_count": sum(e == 0 for e in errors),
            }
            initial = np.random.default_rng(seeds[1]).standard_normal(
                (config.n_eval, *representation.shape), dtype=np.float32,
            )
            np.save(output_dir / f"{name}_initial_noise.npy", initial, allow_pickle=False)
            previous_state = None
            for coupling in ("independent", "ot"):
                method_name = f"{name}_{coupling}"
                report["current_method"] = method_name
                write_json(summary_path, report)
                training = DiTConfig(
                    representation.shape, representation.patch_size, config.hidden_width,
                    config.depth, config.heads, config.train_steps, config.batch_size,
                    config.learning_rate, seeds[0],
                )
                start = time.perf_counter()
                model, losses, state = train_dit(targets, transform, coupling, training)
                training_seconds = time.perf_counter() - start
                if previous_state is not None and (
                    state["initialization_sha256"] != previous_state["initialization_sha256"]
                    or state["raw_batches_sha256"] != previous_state["raw_batches_sha256"]
                ):
                    raise AssertionError("Coupling pair did not share initialization and raw batches.")
                previous_state = {key: state[key] for key in ("initialization_sha256", "raw_batches_sha256")}
                checkpoint = output_dir / f"{method_name}.pt"
                torch.save(state, checkpoint)
                np.save(output_dir / f"{method_name}_loss.npy", losses, allow_pickle=False)
                loaded = torch.load(checkpoint, weights_only=True)
                reloaded = TinyDiT(training).eval()
                reloaded.load_state_dict(loaded["model"])
                probe = initial[:1].reshape(1, -1)
                np.testing.assert_array_equal(
                    as_image_velocity(model)(0.5, probe), as_image_velocity(reloaded)(0.5, probe),
                )
                del loaded, reloaded, state
                native, pixels, control = sample_paths(
                    model, representation, transform, initial, times, output_dir / method_name,
                    config.image_size,
                )
                method = {
                    "training_seconds": training_seconds,
                    "initialization_and_batches": previous_state,
                    "total_parameters": sum(p.numel() for p in model.parameters()),
                    "backbone_parameters": sum(
                        p.numel() for parameter_name, p in model.named_parameters()
                        if not parameter_name.startswith(("input_projection", "output_projection"))
                    ),
                    "native_path": str((output_dir / method_name / "native.npy").resolve()),
                    "pixel_path": str((output_dir / method_name / ("native.npy" if name == "pixels" else "decoded.npy")).resolve()),
                    "endpoint_quality": endpoint_quality(pixels[-1], heldout, bandwidth),
                }
                for space, metrics in (
                    ("native", image_metrics(native, times)), ("pixel", image_metrics(pixels, times)),
                    ("decoded_straight_latent_control", control),
                ):
                    np.savez_compressed(output_dir / method_name / f"{space}_metrics.npz", **metrics)
                    method[space] = metric_summary(metrics)
                if config.grid_refinement:
                    method["grid_refinement"] = refine_grid(
                        model, representation, transform, initial, native, pixels, times,
                    )
                save_image_grid(pixels[-1], output_dir / f"{method_name}_endpoints.png")
                save_image_grid(pixels[:, 0][::max(1, config.integration_steps // 8)], output_dir / f"{method_name}_path.png")
                report["methods"][method_name] = method
                method["total_seconds"] = time.perf_counter() - start
                write_json(summary_path, report)
                print(f"Completed {method_name}: native and decoded pixel diagnostics", flush=True)
                del native, pixels, model
                gc.collect()
            del targets, representation, initial
            gc.collect()
        save_comparison(report, output_dir)
        report["status"] = "complete"
        report["final_resources"] = resource_preflight(output_dir)
        report["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report.pop("current_method", None)
        report.pop("current_representation", None)
        return report
    finally:
        if report["status"] != "complete":
            report["status"] = "failed"
        write_json(summary_path, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/images-seed42"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--manifest", action="store_true", help="Print exact payload/resource manifest without downloading.")
    parser.add_argument("--download", action="store_true", help="Explicit approval of the selected payload manifest.")
    parser.add_argument("--download-only", action="store_true")
    for name in ("seed", "n_train", "n_eval", "image_size", "train_steps", "integration_steps", "batch_size", "hidden_width", "depth", "heads", "threads"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(ImageConfig(), name))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--disk-budget-gb", type=float, default=20)
    parser.add_argument("--no-grid-refinement", action="store_true")
    parser.add_argument("--representations", nargs="+", choices=NAMES, default=list(NAMES))
    args = vars(parser.parse_args())
    output, cache = args.pop("output_dir"), args.pop("cache_dir")
    manifest, download, download_only = args.pop("manifest"), args.pop("download"), args.pop("download_only")
    args["grid_refinement"] = not args.pop("no_grid_refinement")
    args["representations"] = tuple(args["representations"])
    config = ImageConfig(**args)
    logging.basicConfig(level=logging.INFO)
    if manifest:
        print(json.dumps(download_manifest(config, cache / "models"), indent=2))
        return
    if download:
        download_assets(cache, config, approved=True)
    if download_only:
        if not download:
            parser.error("--download-only requires --download.")
        return
    report = run_image_experiment(config, output, cache)
    print(json.dumps({"status": report["status"], "methods": list(report["methods"])}, indent=2))


if __name__ == "__main__":
    main()
