"""Deterministic CIFAR subsets and bounded checkpoint-native latent caches."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .representations import Representation


CIFAR_URL = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"
CIFAR_BYTES = 170498071
CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def image_tensor(image: Image.Image, size: int) -> torch.Tensor:
    rendered = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(rendered, dtype=np.float32) / 255
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)


def prepare_images(
    root: Path, seed: int, n_train: int, n_eval: int, image_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    from torchvision.datasets import CIFAR10

    if not 2 <= n_train <= 50000 or not 2 <= n_eval <= 10000 or seed < 0:
        raise ValueError("Invalid CIFAR subset size or seed.")
    identity = {
        "dataset": "CIFAR-10", "seed": seed, "n_train": n_train, "n_eval": n_eval,
        "image_size": image_size, "rendering": "PIL RGB bicubic; continuous RGB [0,1]",
        "archive_md5": CIFAR_MD5,
    }
    cache = root / f"subset-{seed}-{n_train}-{n_eval}-{image_size}"
    metadata = cache / "metadata.json"
    if metadata.exists():
        saved = json.loads(metadata.read_text())
        if saved["identity"] != identity:
            raise ValueError("CIFAR cache metadata mismatch.")
        return (
            np.load(cache / "train.npy", mmap_mode="r", allow_pickle=False),
            np.load(cache / "test.npy", mmap_mode="r", allow_pickle=False), saved,
        )
    if cache.exists():
        raise FileExistsError(f"Incomplete image cache: {cache}")
    datasets = [
        CIFAR10(root / "raw", train=True, download=False),
        CIFAR10(root / "raw", train=False, download=False),
    ]
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    indices = [
        rng.choice(len(dataset), count, replace=False)
        for rng, dataset, count in zip(rngs, datasets, (n_train, n_eval))
    ]
    cache.mkdir(parents=True)
    for split, dataset, selected in zip(("train", "test"), datasets, indices):
        images = np.lib.format.open_memmap(
            cache / f"{split}.npy", mode="w+", dtype=np.float32,
            shape=(len(selected), 3, image_size, image_size),
        )
        for i, index in enumerate(selected):
            images[i] = image_tensor(dataset[int(index)][0], image_size)[0].numpy()
        images.flush()
        del images
    saved = {
        "identity": identity,
        "train_indices": indices[0].tolist(), "test_indices": indices[1].tolist(),
        "train_labels": [datasets[0].targets[int(i)] for i in indices[0]],
        "test_labels": [datasets[1].targets[int(i)] for i in indices[1]],
    }
    write_json(metadata, saved)
    return (
        np.load(cache / "train.npy", mmap_mode="r", allow_pickle=False),
        np.load(cache / "test.npy", mmap_mode="r", allow_pickle=False), saved,
    )


@dataclass(frozen=True)
class ScalarTransform:
    mean: float
    std: float

    def normalize(self, points: np.ndarray) -> np.ndarray:
        return np.asarray((points - self.mean) / self.std, dtype=np.float32)

    def invert(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points * self.std + self.mean, dtype=np.float32)


def fit_scalar_transform(targets: np.ndarray) -> ScalarTransform:
    if len(targets) < 2:
        raise ValueError("At least two training targets are required.")
    total, squared, count = 0.0, 0.0, 0
    for target in targets:
        values = np.asarray(target, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite training targets.")
        total += float(values.sum())
        squared += float(np.square(values).sum())
        count += values.size
    mean = total / count
    std = float(np.sqrt(max(squared / count - mean**2, 0)))
    if not np.isfinite(std) or std < 1e-8:
        raise ValueError("Training representation has zero or invalid global variance.")
    return ScalarTransform(mean, std)


def encode_targets(
    representation: Representation, images: np.ndarray, root: Path, identity: dict,
) -> tuple[np.ndarray, ScalarTransform]:
    identity = json.loads(json.dumps(identity, sort_keys=True))
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    directory = root / digest
    metadata = directory / "metadata.json"
    if metadata.exists():
        saved = json.loads(metadata.read_text())
        if saved["identity"] != identity:
            raise ValueError("Representation cache metadata mismatch.")
        return (
            np.load(directory / "targets.npy", mmap_mode="r", allow_pickle=False),
            ScalarTransform(**saved["training_transform"]),
        )
    if directory.exists():
        raise FileExistsError(f"Incomplete representation cache: {directory}")
    directory.mkdir(parents=True)
    targets = np.lib.format.open_memmap(
        directory / "targets.npy", mode="w+", dtype=np.float32,
        shape=(len(images), *representation.shape),
    )
    for i, image in enumerate(images):
        tensor = torch.from_numpy(np.array(image[None], copy=True))
        encoded = representation.encode(tensor)
        if encoded.shape != (1, *representation.shape) or not torch.isfinite(encoded).all():
            raise ValueError("Representation returned wrong-shaped or nonfinite targets.")
        if i == 0 and not torch.equal(encoded, representation.encode(tensor)):
            raise ValueError("Frozen encoder is not deterministic.")
        targets[i] = encoded[0].detach().numpy()
    targets.flush()
    transform = fit_scalar_transform(targets)
    write_json(metadata, {
        "identity": identity, "shape": list(targets.shape),
        "training_transform": asdict(transform),
    })
    del targets
    return np.load(directory / "targets.npy", mmap_mode="r", allow_pickle=False), transform
