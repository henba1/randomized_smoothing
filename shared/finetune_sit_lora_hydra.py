from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from shared.lora import LoRAConfig, inject_lora_, load_lora_checkpoint, lora_parameters, save_lora_checkpoint
from shared.utils.diffusion_timestep import find_t_for_sigma


def _import_sit_models():
    rs_rd_dir = Path(__file__).resolve().parents[1]
    rs_rd_dir = rs_rd_dir.parent
    if str(rs_rd_dir) not in sys.path:
        sys.path.insert(0, str(rs_rd_dir))
    from SiT.models import SiT_models

    return SiT_models


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str
    output_path: str

    sit_model_name: str = "SiT-XL/2"
    sit_checkpoint_path: str | None = None  
    sit_vae_id: str = "stabilityai/sd-vae-ft-mse"
    resume_lora_path: str | None = None

    image_size: int = 256
    batch_size: int = 32
    num_workers: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.0
    max_steps: int = 10_000
    log_every: int = 50
    save_every: int = 1_000

    # RS noise matching
    sigma: float = 0.25
    diffusion_steps: int = 1000
    noise_schedule: str = "linear"
    tau_k: float = 1.0
    amp_dtype: str = "bf16"

    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = ("qkv", "proj", "fc1", "fc2")


def _amp_dtype(name: str) -> torch.dtype | None:
    name = name.lower()
    if name == "none":
        return None
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError("amp_dtype must be one of {'bf16','fp16','none'}")


def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    c = TrainConfig(**OmegaConf.to_container(cfg, resolve=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose(
        [
            transforms.Resize(c.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(c.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    ds = datasets.ImageFolder(c.data_dir, transform=tfm)
    loader = DataLoader(
        ds,
        batch_size=c.batch_size,
        shuffle=True,
        num_workers=c.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    sit_models = _import_sit_models()
    if c.sit_model_name not in sit_models:
        raise ValueError(f"Unknown SiT model: {c.sit_model_name}. Options: {sorted(sit_models.keys())}")

    latent_size = c.image_size // 8
    sit = sit_models[c.sit_model_name](input_size=latent_size, in_channels=4, num_classes=1000, learn_sigma=True).to(device)
    if c.sit_checkpoint_path is None:
        raise ValueError("sit_checkpoint_path must be set to the base SiT checkpoint (.pt).")
    base_sd = torch.load(c.sit_checkpoint_path, map_location=device)
    sit.load_state_dict(base_sd)
    sit.train()

    lora_cfg = LoRAConfig(r=c.lora_r, alpha=c.lora_alpha, dropout=c.lora_dropout, target_substrings=c.lora_targets)
    replaced = inject_lora_(sit, lora_cfg)
    if replaced == 0:
        raise RuntimeError("LoRA injection replaced 0 modules; adjust lora_targets.")

    
    if c.resume_lora_path is not None: # resume
        _, lora_sd = load_lora_checkpoint(c.resume_lora_path, device=device)
        _missing, unexpected = sit.load_state_dict(lora_sd, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading resume_lora_path: {unexpected}")

    for p in sit.parameters():
        p.requires_grad_(False)
    for p in lora_parameters(sit):
        p.requires_grad_(True)

    opt = torch.optim.AdamW(lora_parameters(sit), lr=c.lr, weight_decay=c.weight_decay)

    try:
        from diffusers.models import AutoencoderKL
    except ImportError as exc:
        raise ImportError("This training script requires `diffusers` (AutoencoderKL).") from exc

    vae = AutoencoderKL.from_pretrained(c.sit_vae_id).to(device).eval()
    vae.requires_grad_(False)
    vae_scaling = 0.18215

    from ds_imagenet.guided_diffusion.script_util import create_gaussian_diffusion

    diffusion = create_gaussian_diffusion(
        steps=c.diffusion_steps,
        learn_sigma=False,
        noise_schedule=c.noise_schedule,
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
    )

    t_int = find_t_for_sigma(diffusion=diffusion, sigma=c.sigma, target_multiplier=2.0)
    a = float(diffusion.sqrt_alphas_cumprod[t_int])
    b = float(diffusion.sqrt_one_minus_alphas_cumprod[t_int])
    tau = 1.0 / (1.0 + float(c.tau_k) * (b / a))
    tau_t = torch.full((c.batch_size,), tau, device=device, dtype=torch.float32)
    y_null = torch.full((c.batch_size,), 1000, device=device, dtype=torch.long)

    amp_dtype = _amp_dtype(c.amp_dtype)

    step = 0
    start = time.time()
    out_path = Path(c.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    while step < c.max_steps:
        for x, _y in loader:
            step += 1
            x = x.to(device, non_blocking=True)  # already in [-1,1]

            # Pixel-space RS noise (same as inference/certification).
            noise = torch.randn_like(x)
            t_batch = torch.full((x.shape[0],), t_int, device=device, dtype=torch.long)
            x_t = diffusion.q_sample(x_start=x, t=t_batch, noise=noise)

            with torch.no_grad():
                z1 = vae.encode(x).latent_dist.mode().mul_(vae_scaling)
                zt = vae.encode(x_t).latent_dist.mode().mul_(vae_scaling)

            opt.zero_grad(set_to_none=True)

            if device.type == "cuda" and amp_dtype is not None:
                with torch.autocast("cuda", dtype=amp_dtype):
                    u = sit(zt, tau_t[: zt.shape[0]], y_null[: zt.shape[0]])
                    z1_hat = zt + (1.0 - tau) * u
                    loss = F.mse_loss(z1_hat, z1)
                loss.backward()
            else:
                u = sit(zt, tau_t[: zt.shape[0]], y_null[: zt.shape[0]])
                z1_hat = zt + (1.0 - tau) * u
                loss = F.mse_loss(z1_hat, z1)
                loss.backward()

            opt.step()

            if step % c.log_every == 0:
                dt = time.time() - start
                print(f"step={step} loss={float(loss.detach().cpu()):.6f} ({dt:.1f}s)")

            if step % c.save_every == 0 or step >= c.max_steps:
                save_lora_checkpoint(
                    str(out_path),
                    cfg=lora_cfg,
                    model=sit,
                    extra={
                        "base_checkpoint": c.sit_checkpoint_path,
                        "sit_model_name": c.sit_model_name,
                        "sit_vae_id": c.sit_vae_id,
                        "sigma": c.sigma,
                        "t_int": int(t_int),
                        "tau": float(tau),
                        "tau_k": float(c.tau_k),
                    },
                )
                print(f"Saved LoRA checkpoint to {out_path}")

            if step >= c.max_steps:
                return

