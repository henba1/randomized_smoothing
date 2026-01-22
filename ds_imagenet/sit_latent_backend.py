from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch

from .guided_diffusion.script_util import create_gaussian_diffusion


def _import_sit_models():
    """
    Import the vendored SiT repo (sibling of `randomized_smoothing/`).

    We keep this local (instead of requiring an installed package) to minimize project overhead.
    """
    rs_rd_dir = Path(__file__).resolve().parents[2]  # .../rs_rd
    if str(rs_rd_dir) not in sys.path:
        sys.path.insert(0, str(rs_rd_dir))

    from SiT.models import SiT_models  # type: ignore[import-not-found]

    return SiT_models


@dataclass(frozen=True)
class SiTLatentArgs:
    # SiT model (official checkpoints are trained in VAE latent space).
    image_size: int = 256
    sit_model_name: str = "SiT-XL/2"
    num_classes: int = 1000
    in_channels: int = 4
    learn_sigma: bool = True  # matches official 256x256 checkpoint behavior

    # We keep a guided-diffusion-style schedule for mapping sigma -> integer timestep `t`
    # (used by your RS scripts) and for a consistent comparison baseline.
    diffusion_steps: int = 1000
    noise_schedule: str = "linear"
    timestep_respacing: str | None = None
    use_kl: bool = False
    predict_xstart: bool = False
    rescale_timesteps: bool = False
    rescale_learned_sigmas: bool = False


def model_and_diffusion_defaults() -> dict:
    return SiTLatentArgs().__dict__.copy()


def args_to_dict(args: object, keys) -> dict:
    return {k: getattr(args, k) for k in keys}


def create_model_and_diffusion(
    *,
    image_size: int,
    sit_model_name: str,
    num_classes: int,
    in_channels: int,
    learn_sigma: bool,
    diffusion_steps: int,
    noise_schedule: str,
    timestep_respacing: str | None,
    use_kl: bool,
    predict_xstart: bool,
    rescale_timesteps: bool,
    rescale_learned_sigmas: bool,
):
    if image_size != 256:
        raise ValueError("This SiT latent backend currently targets ImageNet 256x256.")
    if in_channels != 4:
        raise ValueError("SiT latent backend expects in_channels=4 (Stable Diffusion VAE latent channels).")

    sit_models = _import_sit_models()
    if sit_model_name not in sit_models:
        raise ValueError(f"Unknown SiT model name: {sit_model_name}. Options: {sorted(sit_models.keys())}")

    latent_size = image_size // 8
    model = sit_models[sit_model_name](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=num_classes,
        learn_sigma=learn_sigma,
    )

    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=False,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing or "",
    )

    return model, diffusion

