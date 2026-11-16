import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from flow_curvature.diagnostics import scale_invariant_diagnostics, trajectory_diagnostics
from flow_curvature.dit import DiTConfig, TinyDiT, as_image_velocity, train_dit
from flow_curvature.image_data import ScalarTransform, encode_targets, fit_scalar_transform
from flow_curvature.image_experiment import (
    ImageConfig, download_assets, image_metrics, metric_summary,
    run_image_experiment, sample_paths,
)
from flow_curvature.representations import FrozenAutoencoder, Pixels, RepresentationSpec
from flow_curvature.integration import integrate


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("shape,patch", [((768, 16, 16), 1), ((4, 32, 32), 2), ((3, 256, 256), 16)])
def test_dit_pairs_share_inputs_update_and_reload(shape, patch, tmp_path):
    targets = np.random.default_rng(3).random((3, *shape), dtype=np.float32)
    config = DiTConfig(shape, patch, width=8, depth=1, heads=2, steps=2, batch_size=2)
    transform = fit_scalar_transform(targets)
    global_state = torch.random.get_rng_state().clone()
    independent, _, first = train_dit(targets, transform, "independent", config)
    ot, losses, second = train_dit(targets, transform, "ot", config)
    assert torch.equal(global_state, torch.random.get_rng_state())
    assert first["initialization_sha256"] == second["initialization_sha256"]
    assert first["raw_batches_sha256"] == second["raw_batches_sha256"]
    assert np.isfinite(losses).all()
    assert torch.count_nonzero(ot.output_projection.weight) > 0
    images = torch.from_numpy(targets[:1])
    torch.testing.assert_close(ot.unpatchify(ot.patchify(images)), images, rtol=0, atol=0)
    assert ot.patchify(images).shape[1] == 256
    checkpoint = tmp_path / "model.pt"
    torch.save(second, checkpoint)
    loaded = torch.load(checkpoint, weights_only=True)
    restored = TinyDiT(config).eval()
    restored.load_state_dict(loaded["model"])
    probe = targets[:1].reshape(1, -1)
    np.testing.assert_array_equal(as_image_velocity(restored)(0.4, probe), as_image_velocity(ot)(0.4, probe))
    _, _, repeated = train_dit(targets, transform, "ot", config)
    for name, weights in second["model"].items():
        torch.testing.assert_close(weights, repeated["model"][name], rtol=0, atol=0)
    assert not torch.equal(independent.output_projection.weight, ot.output_projection.weight)


def test_chunked_high_dimensional_geometry_and_scalar_invariance():
    times = np.linspace(0, 1, 201)
    paths = np.zeros((201, 3, 32))
    paths[:, :, 0] = np.cos(times[:, None])
    paths[:, :, 1] = np.sin(times[:, None])
    expected = trajectory_diagnostics(paths, times)
    measured = image_metrics(paths, times)
    for name in expected:
        np.testing.assert_allclose(measured[name], expected[name], atol=1e-12)
    scaled = image_metrics(paths * 7 + 3, times)
    for name in ("normalized_bending", "relative_length_excess"):
        np.testing.assert_allclose(scaled[name], measured[name], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(measured["rms_path_length"], expected["path_length"] / np.sqrt(32))
    stationary = trajectory_diagnostics(np.zeros((201, 1, 32)), times)
    invariant = scale_invariant_diagnostics(stationary)
    assert np.isnan(invariant["relative_length_excess"]).all()
    assert metric_summary(invariant)["relative_length_excess"]["undefined_count"] == 1
    assert metric_summary(invariant)["relative_length_excess"]["median"] is None
    closed = np.stack((np.cos(2 * np.pi * times), np.sin(2 * np.pi * times)), axis=-1)[:, None]
    assert np.isnan(image_metrics(closed, times)["relative_length_excess"]).all()


def test_continuous_time_dit_curvature_is_resolved_on_a_refined_grid():
    rng = np.random.default_rng(7)
    targets = rng.random((4, 3, 16, 16), dtype=np.float32)
    config = DiTConfig((3, 16, 16), 1, width=8, depth=1, heads=2, steps=8, batch_size=2)
    model, _, _ = train_dit(targets, fit_scalar_transform(targets), "ot", config)
    initial = rng.standard_normal((1, 3 * 16 * 16))
    energies = []
    for steps in (64, 128):
        times = np.linspace(0, 1, steps + 1)
        paths = integrate(as_image_velocity(model), initial, times)
        energies.append(float(image_metrics(paths, times)["normalized_bending"][0]))
    assert energies[1] > 1e-8
    assert abs(energies[0] - energies[1]) / energies[1] < 0.05


class NonlinearDecoder:
    shape = (3, 16, 16)
    patch_size = 1

    def encode(self, images):
        return images

    def decode(self, latents):
        images = latents.clone()
        images[:, 1] = latents[:, 1].square()
        return images


def test_sampling_measures_decoder_bending_not_only_native_interpolation(tmp_path):
    config = DiTConfig((3, 16, 16), 1, width=8, depth=1, heads=2, steps=1)
    model = TinyDiT(config).eval()
    with torch.no_grad():
        model.output_projection.bias.fill_(1)
    times = np.linspace(0, 1, 21)
    initial = np.zeros((2, 3, 16, 16), dtype=np.float32)
    native, pixels, control = sample_paths(
        model, NonlinearDecoder(), ScalarTransform(0, 1), initial, times, tmp_path / "nonlinear", 16,
    )
    assert image_metrics(native, times)["bending_energy"].max() < 1e-10
    assert image_metrics(pixels, times)["bending_energy"].min() > 0.001
    assert control["bending_energy"].min() > 0.001
    identity_native, identity_pixels, _ = sample_paths(
        model, Pixels(16), ScalarTransform(0, 1), initial, times, tmp_path / "identity", 16,
    )
    assert identity_native is identity_pixels
    assert not (tmp_path / "identity" / "decoded.npy").exists()


def test_sd_vae_uses_posterior_mean_inverts_scaling_without_clipping():
    recorded = {}

    def encode(images):
        recorded["images"] = images
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=torch.full((1, 4, 32, 32), 2.0)))

    def decode(latents):
        recorded["latents"] = latents
        return SimpleNamespace(sample=torch.full((1, 3, 256, 256), 2.0))

    representation = FrozenAutoencoder.__new__(FrozenAutoencoder)
    representation.device = torch.device("cpu")
    representation.model = SimpleNamespace(encode=encode, decode=decode)
    representation.is_vae = True
    representation.shape = (4, 32, 32)
    representation.scaling_factor = 0.18215
    latents = representation.encode(torch.full((1, 3, 256, 256), 0.25))
    torch.testing.assert_close(recorded["images"], torch.full((1, 3, 256, 256), -0.5))
    torch.testing.assert_close(latents, torch.full((1, 4, 32, 32), 0.3643))
    images = representation.decode(latents)
    torch.testing.assert_close(recorded["latents"], torch.full((1, 4, 32, 32), 2.0))
    assert images.min() == 1.5


def test_nested_rae_checkpoint_loading_disables_transformers_initialization(tmp_path, monkeypatch):
    from diffusers import AutoencoderRAE

    model = AutoencoderRAE(
        encoder_hidden_size=64, encoder_num_hidden_layers=1,
        decoder_hidden_size=64, decoder_num_hidden_layers=1,
        decoder_num_attention_heads=1, decoder_intermediate_size=128,
    )
    snapshot = tmp_path / "snapshot"
    model.save_pretrained(snapshot)
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *args, **kwargs: str(snapshot))
    spec = RepresentationSpec("dinov2", "local-fixture", "local", (64, 16, 16), 1, (), "fixture")
    loaded = FrozenAutoencoder(spec, tmp_path)
    images = torch.rand(1, 3, 256, 256)
    first = loaded.encode(images)
    torch.testing.assert_close(first, loaded.encode(images), rtol=0, atol=0)
    assert loaded.decode(first).shape == (1, 3, 256, 256)
    assert not any(parameter.requires_grad for parameter in loaded.model.parameters())
    assert torch.nn.init.trunc_normal_(torch.empty(1)) is not None


def test_representation_cache_and_training_only_normalization(tmp_path):
    images = np.random.default_rng(4).random((4, 3, 16, 16), dtype=np.float32)
    targets, transform = encode_targets(Pixels(16), images, tmp_path, {"images": (1, 2, 3, 4)})
    restored, repeated = encode_targets(Pixels(16), images, tmp_path, {"images": (1, 2, 3, 4)})
    np.testing.assert_array_equal(targets, restored)
    assert transform == repeated
    np.testing.assert_allclose(transform.invert(transform.normalize(images)), images, atol=1e-7)
    assert abs(float(transform.normalize(images).mean())) < 1e-6
    with pytest.raises(ValueError, match="variance"):
        fit_scalar_transform(np.ones((4, 3, 16, 16)))


def test_image_runner_offline_completion_failure_and_download_gate(tmp_path):
    rng = np.random.default_rng(2)
    train = rng.random((4, 3, 16, 16), dtype=np.float32)
    heldout = rng.random((2, 3, 16, 16), dtype=np.float32)
    config = ImageConfig(
        n_train=4, n_eval=2, image_size=16, train_steps=2, batch_size=2,
        integration_steps=4, hidden_width=8, depth=1, heads=2,
        representations=("pixels",), grid_refinement=True,
    )
    report = run_image_experiment(
        config, tmp_path / "run", tmp_path / "cache", image_data=(train, heldout, {"fixture": True}),
    )
    assert report["status"] == "complete"
    assert set(report["methods"]) == {"pixels_independent", "pixels_ot"}
    import csv
    with (tmp_path / "run" / "comparison.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert {row["coupling"] for row in rows} == {"independent", "ot"}
    for row in rows:
        assert float(row["pixel_bending_median"]) == report["methods"][f"pixels_{row['coupling']}"]["pixel"]["bending_energy"]["median"]
    assert (tmp_path / "run" / "curvature_comparison.png").is_file()
    for method in report["methods"].values():
        assert method["native_path"] == method["pixel_path"]
        assert "grid_refinement" in method
    saved = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert saved["status"] == "complete"
    assert np.load(tmp_path / "run" / "pixels_ot" / "native.npy").shape == (5, 2, 3, 16, 16)
    with pytest.raises(FileExistsError):
        run_image_experiment(config, tmp_path / "run", tmp_path / "cache", image_data=(train, heldout, {}))
    with pytest.raises(ValueError, match="Dataset shapes"):
        run_image_experiment(config, tmp_path / "failed", tmp_path / "cache", image_data=(train[:1], heldout, {}))
    assert json.loads((tmp_path / "failed" / "summary.json").read_text())["status"] == "failed"
    with pytest.raises(PermissionError, match="approval"):
        download_assets(tmp_path / "cache", config)
