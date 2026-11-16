"""Frozen, checkpoint-native image representations for the optional image pilot."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch


@dataclass(frozen=True)
class RepresentationSpec:
    name: str
    repository: str
    revision: str
    shape: tuple[int, int, int]
    patch_size: int
    files: tuple[tuple[str, int], ...]
    license: str


SPECS = (
    RepresentationSpec(
        "dinov2", "nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08",
        "eb1744fb3fc4269bb37a804a1afeed1194bc105c", (768, 16, 16), 1,
        (("config.json", 5691714), ("diffusion_pytorch_model.safetensors", 2008606636)),
        "Weight license unspecified in model card; upstream software MIT.",
    ),
    RepresentationSpec(
        "siglip2", "nyu-visionx/RAE-siglip2-base-p16-i256-ViTXL-n08",
        "9ed221bb479c83f610d3b3924f8a4a2f9a763b05", (768, 16, 16), 1,
        (("config.json", 11648892), ("diffusion_pytorch_model.safetensors", 2034780576)),
        "Weight license unspecified in model card; upstream software MIT.",
    ),
    RepresentationSpec(
        "mae", "nyu-visionx/RAE-mae-base-p16-ViTXL-n08",
        "2dc45f481873a1c7159e7b94d7b85c7e33b7f2d9", (768, 16, 16), 1,
        (("config.json", 11624881), ("diffusion_pytorch_model.safetensors", 2006251392)),
        "Weight license unspecified in model card; upstream software MIT.",
    ),
    RepresentationSpec(
        "sd_vae", "stabilityai/sd-vae-ft-mse",
        "31f26fdeee1355a5c34592e401dd41e45d25a493", (4, 32, 32), 2,
        (("config.json", 547), ("diffusion_pytorch_model.safetensors", 334643276)),
        "MIT",
    ),
)
NAMES = tuple(spec.name for spec in SPECS) + ("pixels",)


class Representation(Protocol):
    shape: tuple[int, int, int]
    patch_size: int

    def encode(self, images: torch.Tensor) -> torch.Tensor: ...
    def decode(self, latents: torch.Tensor) -> torch.Tensor: ...


class Pixels:
    def __init__(self, image_size: int = 256, patch_size: int | None = None) -> None:
        self.shape = (3, image_size, image_size)
        self.patch_size = image_size // 16 if patch_size is None else patch_size

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return images

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents


class FrozenAutoencoder:
    def __init__(self, spec: RepresentationSpec, cache_dir: Path, device: str = "cpu") -> None:
        from diffusers import AutoencoderKL, AutoencoderRAE
        from huggingface_hub import snapshot_download
        from transformers.modeling_utils import no_init_weights

        location = snapshot_download(
            spec.repository, revision=spec.revision, cache_dir=cache_dir,
            allow_patterns=[name for name, _ in spec.files], local_files_only=True,
        )
        model_class = AutoencoderKL if spec.name == "sd_vae" else AutoencoderRAE
        # RAE nests Transformers models inside Diffusers' no-initialization loader.
        with no_init_weights():
            self.model, loading = model_class.from_pretrained(
                location, local_files_only=True, use_safetensors=True,
                torch_dtype=torch.float32, output_loading_info=True,
            )
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Incompatible frozen checkpoint {spec.repository}: {loading}")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback.")
        self.model.to(self.device).eval().requires_grad_(False)
        self.shape, self.patch_size = spec.shape, spec.patch_size
        self.is_vae = spec.name == "sd_vae"
        self.scaling_factor = float(self.model.config.scaling_factor)
        if not self.is_vae:
            self.model.noise_tau = 0.0
        self.model.enable_slicing()

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        if self.is_vae:
            result = self.model.encode(images * 2 - 1).latent_dist.mean
            result = result * self.scaling_factor
        else:
            result = self.model.encode(images).latent
        if result.shape[1:] != self.shape or not torch.isfinite(result).all():
            raise ValueError(f"Invalid encoded latents: {tuple(result.shape)}.")
        return result.cpu()

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(self.device)
        if self.is_vae:
            result = (self.model.decode(latents / self.scaling_factor).sample + 1) / 2
        else:
            result = self.model.decode(latents).sample
        if result.shape[1:] != (3, 256, 256) or not torch.isfinite(result).all():
            raise ValueError(f"Invalid decoded images: {tuple(result.shape)}.")
        return result.cpu()


def load_representation(name: str, cache_dir: Path, image_size: int = 256, device: str = "cpu") -> Representation:
    if name == "pixels":
        return Pixels(image_size)
    if image_size != 256:
        raise ValueError("Released RAEs in this pilot require 256x256 rendered images.")
    spec = next((spec for spec in SPECS if spec.name == name), None)
    if spec is None:
        raise ValueError(f"Unknown representation: {name}")
    return FrozenAutoencoder(spec, cache_dir, device)
