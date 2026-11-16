"""Read-only aggregation for image sweeps and the results notebook."""

import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .image_data import write_json


def load_results(root: Path) -> tuple[dict, list[dict]]:
    manifest_path = root / "sweep.json"
    if not manifest_path.exists():
        return {}, []
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported result schema: {manifest_path}")
    rows = []
    for stream_id, entry in manifest["streams"].items():
        for update, relative_path in entry.get("evaluations", {}).items():
            path = root / relative_path
            if not path.exists():
                raise FileNotFoundError(f"Manifest references missing evaluation: {path}")
            result = json.loads(path.read_text())
            for grid in manifest["config"]["grids"]:
                for space in ("native", "pixel"):
                    metrics = result[space][str(grid)]
                    rows.append({
                        "stream_id": stream_id, "representation": entry["representation"],
                        "coupling": entry["coupling"], "width": entry["width"],
                        "depth": entry["depth"], "seed": entry["seed"], "update": int(update),
                        "grid": grid, "space": space,
                        "normalized_bending": metrics["normalized_bending"]["median"],
                        "bending_energy": metrics["bending_energy"]["median"],
                        "relative_length_excess": metrics["relative_length_excess"]["median"],
                        "validation_loss": result["validation_history"][-1]["loss"],
                        "endpoint_mmd2": result["endpoint_quality_validation"]["pixel_mmd2_unbiased"],
                        "particles": result["native_particles"] if space == "native" else len(result["decoded_particles"]),
                        "grid_status": result["grid_checks"][f"{space}_measurement"]["status"],
                        "path": str(path),
                    })
    return manifest, rows


def seed_summaries(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in ("representation", "coupling", "width", "depth", "update", "grid", "space"))
        groups.setdefault(key, []).append(row)
    result = []
    for key, items in groups.items():
        values = np.asarray([row["normalized_bending"] for row in items], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite seed-level bending.")
        record = dict(zip(("representation", "coupling", "width", "depth", "update", "grid", "space"), key))
        record.update(
            n_training_seeds=len(items), seed_values=values.tolist(),
            mean=float(values.mean()), minimum=float(values.min()), maximum=float(values.max()),
            uncertainty_unit="trained model seed; particles are not training replications",
        )
        if len(items) >= 5:
            rng = np.random.default_rng(42)
            means = rng.choice(values, (2000, len(values)), replace=True).mean(axis=1)
            record["bootstrap_mean_interval_95"] = np.quantile(means, [0.025, 0.975]).tolist()
            record["interval_caveat"] = "Seed bootstrap; exploratory with few models."
        result.append(record)
    return result


def paired_differences(rows: list[dict]) -> list[dict]:
    pairs = {}
    for row in rows:
        key = tuple(row[name] for name in ("representation", "width", "depth", "seed", "update", "grid", "space"))
        pairs.setdefault(key, {})[row["coupling"]] = row["normalized_bending"]
    return [
        {
            **dict(zip(("representation", "width", "depth", "seed", "update", "grid", "space"), key)),
            "ot_minus_independent": values["ot"] - values["independent"],
        }
        for key, values in pairs.items() if set(values) == {"independent", "ot"}
    ]


def runtime_report(manifest: dict, rows: list[dict]) -> dict:
    config = manifest["config"]
    stages = {"setup": manifest.get("setup_seconds", 0) + sum(manifest.get("representation_setup_seconds", {}).values())}
    stages["training_validation_serialization"] = sum(
        entry.get("training_wall_seconds", 0) for entry in manifest["streams"].values()
    )
    stages["integration"] = stages["decoding"] = stages["diagnostics_io"] = 0.0
    assignments = 0.0
    evaluated = set()
    by_shape = []
    for stream, entry in manifest["streams"].items():
        assignments += entry.get("training_timings", {}).get("assignment_seconds", 0)
        by_shape.append({
            "stream": stream, "training_wall_seconds": entry.get("training_wall_seconds"),
            "timings": entry.get("training_timings", {}),
        })
    for row in rows:
        if row["path"] in evaluated:
            continue
        evaluated.add(row["path"])
        timing = json.loads(Path(row["path"]).read_text())["timings"]
        stages["integration"] += timing["integration_seconds"]
        stages["decoding"] += timing["decode_seconds"]
        stages["diagnostics_io"] += timing["diagnostics_and_io_seconds"]
    total = sum(stages.values())
    fixed = min(total, assignments + stages["diagnostics_io"])
    scenarios = []
    # These are sensitivity assumptions, not measured A100 performance.
    for assumed_speedup in (5, 15, 30):
        scenarios.append({
            "hardware_scenario": "one NVIDIA A100 40GB; unmeasured",
            "assumed_accelerable_stage_speedup": assumed_speedup,
            "assumed_fixed_cpu_seconds": fixed,
            "estimated_seconds_for_same_work": fixed + (total - fixed) / assumed_speedup,
            "warning": "Not an A100 benchmark. Transfers, batch efficiency and storage may invalidate this assumption.",
        })
    return {
        "schema_version": 1, "measured_device": config["device"], "hardware": manifest["hardware"],
        "stage_seconds": stages, "accounted_seconds": total, "per_stream_training": by_shape,
        "coverage": {"evaluated_checkpoints": len(evaluated), "training_seeds": config["training_seeds"]},
        "cpu_warm_cache_projection": {
            "same_measured_work_seconds": total,
            "illustrative_variability_range_seconds": [total * 0.75, total * 1.5],
            "caveat": "Scenario range, not a statistical confidence interval. New shapes require fresh benchmarks.",
        } if config["device"] == "cpu" else None,
        "gpu_sensitivity_scenarios": scenarios if config["device"] == "cpu" else [],
        "gpu_measured": config["device"] == "cuda",
        "cache_policy": "Existing frozen assets; setup may include new subset encoding. Download time excluded.",
        "cuda_benchmark_instruction": "Run the same sweep config with device=cuda in a separate CUDA environment; compare stage_seconds.",
    }


def summarize_sweep(root: Path) -> dict:
    manifest, rows = load_results(root)
    if not manifest:
        raise FileNotFoundError(f"No sweep manifest in {root}")
    aggregation = {
        "schema_version": 1, "status": manifest["status"], "rows": len(rows),
        "seed_summaries": seed_summaries(rows), "paired_differences": paired_differences(rows),
    }
    write_json(root / "aggregate.json", aggregation)
    write_json(root / "runtime.json", runtime_report(manifest, rows))
    if rows:
        with (root / "measurements.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return aggregation


def plot_results(rows: list[dict], view: str = "grid") -> Figure:
    if view not in ("grid", "training", "capacity", "seed"):
        raise ValueError(f"Unknown plot view: {view}")
    representations = sorted({row["representation"] for row in rows})
    figure = Figure(figsize=(14, 4 * max(1, len(representations))))
    axes = figure.subplots(max(1, len(representations)), 2, squeeze=False)
    x_key = {"grid": "grid", "training": "update", "capacity": "width", "seed": "seed"}[view]
    for pair, representation in zip(axes, representations):
        for axis, space in zip(pair, ("native", "pixel")):
            groups = {}
            dimensions = ["coupling", "width", "depth", "seed", "update", "grid"]
            dimensions.remove(x_key)
            if view == "capacity":
                dimensions.remove("depth")
            for row in rows:
                if row["space"] == space and row["representation"] == representation:
                    key = tuple(row[name] for name in dimensions)
                    groups.setdefault(key, []).append(row)
            for key, items in groups.items():
                items.sort(key=lambda row: (row[x_key], row["depth"]))
                xs = [f'w{row["width"]}/d{row["depth"]}' if view == "capacity" else row[x_key] for row in items]
                axis.plot(xs, [row["normalized_bending"] for row in items], ".-", label=str(key))
            axis.set(xlabel={"grid": "RK4 / measurement steps", "training": "Training updates",
                             "capacity": "Backbone capacity", "seed": "Training seed"}[view],
                     ylabel="Median normalized bending", title=f"{representation}: {space}")
            axis.set_yscale("symlog", linthresh=1e-6)
            if axis.lines:
                axis.legend(title=", ".join(dimensions), fontsize=5, title_fontsize=6, ncol=2)
    figure.tight_layout()
    return figure


def plot_losses(rows: list[dict]) -> Figure:
    latest = {}
    for row in rows:
        if row["update"] >= latest.get(row["stream_id"], {}).get("update", -1):
            latest[row["stream_id"]] = row
    representations = sorted({row["representation"] for row in rows})
    figure = Figure(figsize=(14, 3 * max(1, len(representations))))
    axes = figure.subplots(max(1, len(representations)), 2, squeeze=False)
    for pair, representation in zip(axes, representations):
        for axis, coupling in zip(pair, ("independent", "ot")):
            for row in latest.values():
                if row["representation"] != representation or row["coupling"] != coupling:
                    continue
                result = json.loads(Path(row["path"]).read_text())
                label = f'w{row["width"]}/d{row["depth"]}/s{row["seed"]}'
                line, = axis.plot(np.arange(1, len(result["training_loss"]) + 1), result["training_loss"], label=label)
                history = result["validation_history"]
                axis.plot([value["update"] for value in history], [value["loss"] for value in history],
                          "x--", color=line.get_color(), label=f"{label} validation")
            axis.set(title=f"{representation}: {coupling}", xlabel="Training updates", ylabel="Velocity MSE")
            if axis.lines:
                axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    return figure
