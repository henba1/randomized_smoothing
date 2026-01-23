from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn

from .guided_diffusion.script_util import create_gaussian_diffusion


def _import_sit_models():
    """
    Import the vendored SiT repo.

    In this codebase, `randomized_smoothing/` lives in `rs_rd/randomized_smoothing/` and the SiT
    repo lives in `rs_rd/SiT/`, i.e. as a sibling directory. To keep integration minimal (no
    packaging/repo restructuring), we add `rs_rd/` to `sys.path` at runtime.
    """
    rs_rd_dir = Path(__file__).resolve().parents[2]
    if str(rs_rd_dir) not in sys.path:
        sys.path.insert(0, str(rs_rd_dir))

    from SiT.models import SiT_models 

    return SiT_models


class UnconditionalSiTDenoiser(nn.Module):
    """
    A minimal wrapper that makes SiT usable as an unconditional denoiser.

    - **Unconditional**: always passes the "null token" class label (index `num_classes`).
    - **guided-diffusion compatible**: accepts `(x, t, **kwargs)` and returns a tensor matching `x` shape.
    """

    def __init__(self, *, sit_model: nn.Module, num_classes: int):
        super().__init__()
        self.sit_model = sit_model
        self.num_classes = int(num_classes)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **_kwargs) -> torch.Tensor:
        # guided-diffusion calls the model with 1-D timestep tensor.
        if t.ndim != 1:
            raise ValueError(f"Expected t to have shape (B,), got {tuple(t.shape)}")
        if x.shape[0] != t.shape[0]:
            raise ValueError(f"Batch mismatch: x has B={x.shape[0]} but t has B={t.shape[0]}")

        # SiT uses a LabelEmbedder with an extra "null" embedding at index == num_classes for unconditional inference
        y_null = torch.full((x.shape[0],), fill_value=self.num_classes, device=x.device, dtype=torch.long)
        return self.sit_model(x, t.float(), y_null)


@dataclass(frozen=True)
class SiTArgs:
    image_size: int = 256
    sit_model_name: str = "SiT-B/8"
    num_classes: int = 1000
    in_channels: int = 3
    learn_sigma: bool = False  # keep diffusion simple (fixed variance) for compatibility

    # keep guided-diffusion-style schedule for mapping sigma to timestep t
    diffusion_steps: int = 1000
    noise_schedule: str = "linear"
    timestep_respacing: str | None = None
    use_kl: bool = False
    predict_xstart: bool = False
    rescale_timesteps: bool = False
    rescale_learned_sigmas: bool = False


def model_and_diffusion_defaults() -> dict:
    return SiTArgs().__dict__.copy()


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
    if learn_sigma:
        raise ValueError(
            "This SiT integration expects learn_sigma=False "
            "(fixed variance diffusion), because SiT's reference implementation "
            "returns only the first half of channels when learn_sigma=True."
        )
    if in_channels != 3:
        raise ValueError("This integration currently targets ImageNet RGB denoising (in_channels=3).")
    if image_size != 256:
        raise ValueError("This integration currently targets ImageNet 256x256 (image_size=256).")

    sit_models = _import_sit_models()
    if sit_model_name not in sit_models:
        raise ValueError(f"Unknown SiT model name: {sit_model_name}. Options: {sorted(sit_models.keys())}")

    sit = sit_models[sit_model_name](
        input_size=image_size,
        in_channels=in_channels,
        num_classes=num_classes,
        learn_sigma=learn_sigma,
    )
    model = UnconditionalSiTDenoiser(sit_model=sit, num_classes=num_classes)

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

