"""Small spatial DiT backbone with paired linear conditional flow matching."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .coupling import couple
from .fields import Coupling, Velocity
from .image_data import ScalarTransform


@dataclass(frozen=True)
class DiTConfig:
    shape: tuple[int, int, int]
    patch_size: int
    width: int = 64
    depth: int = 2
    heads: int = 4
    steps: int = 200
    batch_size: int = 8
    learning_rate: float = 1e-3
    seed: int = 42

    def __post_init__(self) -> None:
        if (
            len(self.shape) != 3 or min(self.shape) < 1
            or min(self.patch_size, self.width, self.depth, self.heads, self.steps, self.batch_size) < 1
            or self.width % 4 or self.width % self.heads
            or self.shape[1] % self.patch_size or self.shape[2] % self.patch_size
            or self.seed < 0 or not np.isfinite(self.learning_rate) or self.learning_rate <= 0
        ):
            raise ValueError("Invalid tiny DiT configuration.")


class DiTBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 6))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(condition).chunk(6, dim=-1)
        normalized = self.norm1(tokens) * (1 + scale1[:, None]) + shift1[:, None]
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        tokens = tokens + gate1[:, None] * attended
        normalized = self.norm2(tokens) * (1 + scale2[:, None]) + shift2[:, None]
        return tokens + gate2[:, None] * self.mlp(normalized)


class TinyDiT(nn.Module):
    def __init__(self, config: DiTConfig) -> None:
        super().__init__()
        self.config = config
        width = config.width
        self.time_embedding = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width),
        )
        self.blocks = nn.ModuleList([DiTBlock(width, config.heads) for _ in range(config.depth)])
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 2))
        patch_dimension = config.shape[0] * config.patch_size**2
        self.input_projection = nn.Linear(patch_dimension, width)
        self.output_projection = nn.Linear(width, patch_dimension)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        h, w = config.shape[1] // config.patch_size, config.shape[2] // config.patch_size
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        frequencies = 1 / (10000 ** (torch.arange(width // 4) / (width // 4)))
        xpos, ypos = x.flatten()[:, None] * frequencies, y.flatten()[:, None] * frequencies
        self.register_buffer("position", torch.cat((xpos.sin(), xpos.cos(), ypos.sin(), ypos.cos()), dim=1))
        self.register_buffer("time_frequencies", 1 / (10000 ** (torch.arange(width // 2) / (width // 2))))

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != self.config.shape:
            raise ValueError("Images do not match the configured DiT shape.")
        batch, channels, height, width = images.shape
        patch = self.config.patch_size
        return images.reshape(batch, channels, height // patch, patch, width // patch, patch).permute(
            0, 2, 4, 1, 3, 5,
        ).reshape(batch, -1, channels * patch**2)

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        channels, height, width = self.config.shape
        patch = self.config.patch_size
        return tokens.reshape(-1, height // patch, width // patch, channels, patch, patch).permute(
            0, 3, 1, 4, 2, 5,
        ).reshape(-1, channels, height, width)

    def forward(self, times: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        angles = times.reshape(-1, 1) * (2 * torch.pi) * self.time_frequencies
        condition = self.time_embedding(torch.cat((angles.cos(), angles.sin()), dim=-1))
        tokens = self.input_projection(self.patchify(images)) + self.position
        for block in self.blocks:
            tokens = block(tokens, condition)
        shift, scale = self.final_modulation(condition).chunk(2, dim=-1)
        tokens = self.final_norm(tokens) * (1 + scale[:, None]) + shift[:, None]
        return self.unpatchify(self.output_projection(tokens))


def train_dit(
    targets: np.ndarray, transform: ScalarTransform, method: Coupling, config: DiTConfig,
    *, device: str = "cpu", checkpoint_dir: Path | None = None,
    checkpoint_steps: tuple[int, ...] = (), resume: dict | None = None,
    validation: np.ndarray | None = None, validation_seed: int = 0,
) -> tuple[TinyDiT, np.ndarray, dict]:
    if method not in ("independent", "ot"):
        raise ValueError(f"Unknown coupling: {method}")
    if tuple(targets.shape[1:]) != config.shape or len(targets) < 2:
        raise ValueError("Targets do not match the DiT training configuration.")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback.")
    if any(step < 1 or step > config.steps for step in checkpoint_steps):
        raise ValueError("Checkpoint updates must be within the training budget.")
    if validation is not None and (len(validation) < 2 or tuple(validation.shape[1:]) != config.shape):
        raise ValueError("Invalid validation targets.")
    data_digest = hashlib.sha256()
    for array in (targets, validation if validation is not None else targets[:0]):
        data_digest.update(str(array.shape).encode())
        for item in array:
            data_digest.update(np.asarray(item, dtype=np.float32).tobytes())
    identity = {
        "data_sha256": data_digest.hexdigest(), "transform": asdict(transform),
        "validation_seed": validation_seed, "device_type": target_device.type,
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = TinyDiT(config)
    initialization = hashlib.sha256()
    for parameter in model.parameters():
        initialization.update(parameter.detach().numpy().tobytes())
    model.to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    sample_rng = np.random.default_rng(config.seed)
    pairing_rng = np.random.default_rng(np.random.SeedSequence(config.seed).spawn(1)[0])
    losses = np.empty(config.steps, dtype=np.float64)
    costs = np.empty(config.steps, dtype=np.float64)
    raw_digest = ""
    history = []
    completed = 0
    timings = {"updates_seconds": 0.0, "assignment_seconds": 0.0, "validation_seconds": 0.0}
    if resume is not None:
        expected = asdict(config)
        actual = dict(resume["config"])
        expected.pop("steps")
        actual.pop("steps")
        actual["shape"] = tuple(actual["shape"])
        if (
            resume.get("schema_version") != 2 or actual != expected
            or resume.get("coupling") != method or resume.get("identity") != identity
            or resume["completed_updates"] > config.steps
        ):
            raise ValueError("Incompatible checkpoint resume (data, architecture, objective, device or schema).")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        completed = resume["completed_updates"]
        losses[:completed] = resume["losses"]
        costs[:completed] = resume["normalized_assignment_cost"]
        sample_rng.bit_generator.state = json.loads(resume["sample_rng"])
        pairing_rng.bit_generator.state = json.loads(resume["pairing_rng"])
        raw_digest = resume["raw_batches_sha256"]
        history = list(resume["validation_history"])
        timings = dict(resume["timings"])
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def synchronize() -> None:
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)

    def snapshot(update: int) -> dict:
        return {
            "schema_version": 2, "completed_updates": update,
            "config": asdict(config), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "coupling": method, "identity": identity,
            "initialization_sha256": initialization.hexdigest(),
            "raw_batches_sha256": raw_digest, "normalized_assignment_cost": costs[:update].tolist(),
            "losses": losses[:update].tolist(), "validation_history": list(history),
            "sample_rng": json.dumps(sample_rng.bit_generator.state),
            "pairing_rng": json.dumps(pairing_rng.bit_generator.state), "timings": dict(timings),
        }

    def validate() -> float:
        rng = np.random.default_rng(validation_seed)
        source = rng.standard_normal((config.batch_size, *config.shape), dtype=np.float32)
        target = transform.normalize(validation[rng.integers(len(validation), size=config.batch_size)])
        times = rng.random((config.batch_size, 1, 1, 1), dtype=np.float32)
        x0, x1 = couple(source.reshape(config.batch_size, -1), target.reshape(config.batch_size, -1), method, rng)
        start = torch.from_numpy(x0.reshape(source.shape)).to(target_device)
        end = torch.from_numpy(x1.reshape(source.shape)).to(target_device)
        t = torch.from_numpy(times).to(target_device)
        with torch.inference_mode():
            value = (model(t, (1-t)*start+t*end) - (end-start)).square().mean().item()
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite validation loss.")
        return value

    model.train()
    for step in range(completed, config.steps):
        synchronize()
        update_start = time.perf_counter()
        selected = sample_rng.integers(len(targets), size=config.batch_size)
        source = sample_rng.standard_normal((config.batch_size, *config.shape), dtype=np.float32)
        target = transform.normalize(targets[selected])
        times = sample_rng.random((config.batch_size, 1), dtype=np.float32)
        batch_digest = hashlib.sha256()
        for values in (selected, source, target, times):
            batch_digest.update(values.tobytes())
        raw_digest = hashlib.sha256((raw_digest + batch_digest.hexdigest()).encode()).hexdigest()
        assignment_start = time.perf_counter()
        x0, x1 = couple(
            source.reshape(config.batch_size, -1), target.reshape(config.batch_size, -1),
            method, pairing_rng,
        )
        timings["assignment_seconds"] += time.perf_counter() - assignment_start
        start = torch.from_numpy(x0.reshape(source.shape)).to(target_device)
        end = torch.from_numpy(x1.reshape(source.shape)).to(target_device)
        t = torch.from_numpy(times).reshape(-1, 1, 1, 1).to(target_device)
        prediction = model(t, (1 - t) * start + t * end)
        loss = (prediction - (end - start)).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{method}: nonfinite training loss at step {step}.")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses[step] = loss.item()
        costs[step] = float(np.square(x1 - x0).mean())
        synchronize()
        timings["updates_seconds"] += time.perf_counter() - update_start
        if step + 1 in checkpoint_steps or step + 1 == config.steps:
            if validation is not None:
                before = time.perf_counter()
                model.eval()
                value = validate()
                model.train()
                timings["validation_seconds"] += time.perf_counter() - before
                history.append({"update": step + 1, "loss": value})
            if checkpoint_dir is not None:
                destination = checkpoint_dir / f"step-{step + 1}.pt"
                if destination.exists():
                    raise FileExistsError(destination)
                temporary = destination.with_suffix(".pt.tmp")
                torch.save(snapshot(step + 1), temporary)
                temporary.replace(destination)
    model.eval()
    state = snapshot(config.steps)
    return model, losses, state


def as_image_velocity(model: TinyDiT) -> Velocity:
    def velocity(time: float, points: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            device = next(model.parameters()).device
            images = torch.from_numpy(np.asarray(points, dtype=np.float32)).reshape(-1, *model.config.shape).to(device)
            times = torch.full((len(images), 1), time, device=device)
            result = model(times, images).flatten(1).cpu().numpy().astype(np.float64)
        return result
    return velocity
