import torch

from shared.drm_base import DiffusionRobustModelBase

from .guided_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from .sit_latent_backend import (
    SiTLatentArgs,
    args_to_dict as sit_latent_args_to_dict,
    create_model_and_diffusion as create_sit_latent_model_and_diffusion,
    model_and_diffusion_defaults as sit_latent_model_and_diffusion_defaults,
)


class Args:
    image_size = 256
    num_channels = 256
    num_res_blocks = 2
    num_heads = 4
    num_heads_upsample = -1
    num_head_channels = 64
    attention_resolutions = "32,16,8"
    channel_mult = ""
    dropout = 0.0
    class_cond = False
    use_checkpoint = False
    use_scale_shift_norm = True
    resblock_updown = True
    use_fp16 = False
    use_new_attention_order = False
    clip_denoised = True
    num_samples = 10000
    batch_size = 16
    use_ddim = False
    model_path = ""
    classifier_path = ""
    classifier_scale = 1.0
    learn_sigma = True
    diffusion_steps = 1000
    noise_schedule = "linear"
    timestep_respacing = None
    use_kl = False
    predict_xstart = False
    rescale_timesteps = False
    rescale_learned_sigmas = False


def _load_state_dict(diffusion_model_path: str, device: torch.device) -> dict:
    return torch.load(diffusion_model_path, map_location=device)


class DiffusionRobustModel(DiffusionRobustModelBase):
    def __init__(
        self,
        classifier_type: str,
        classifier_name: str,
        models_dir: str | None = None,
        dataset_name: str | None = "ImageNet",
        device: torch.device | None = None,
        image_size=None,
        pytorch_normalization: str = "none",
        *,
        denoiser_backend: str,
        model_subdir: str,
        sit_vae_id: str = "stabilityai/sd-vae-ft-mse",
    ):
        if denoiser_backend not in {"guided_diffusion", "sit_latent"}:
            raise ValueError("denoiser_backend must be one of {'guided_diffusion', 'sit_latent'}")

        if denoiser_backend == "guided_diffusion":
            super().__init__(
                diffusion_args=Args(),
                create_model_and_diffusion=create_model_and_diffusion,
                model_and_diffusion_defaults=model_and_diffusion_defaults,
                args_to_dict=args_to_dict,
                load_state_dict_fn=_load_state_dict,
                classifier_type=classifier_type,
                classifier_name=classifier_name,
                models_dir=models_dir,
                dataset_name=dataset_name,
                device=device,
                image_size=image_size,
                pytorch_normalization=pytorch_normalization,
                default_target_size=(512, 512),
                timm_target_size=(512, 512),
                verbose=False,
                list_missing_models=False,
                model_subdir=model_subdir,
            )
            self.denoiser_backend = denoiser_backend
            return

        super().__init__(
            diffusion_args=SiTLatentArgs(),
            create_model_and_diffusion=create_sit_latent_model_and_diffusion,
            model_and_diffusion_defaults=sit_latent_model_and_diffusion_defaults,
            args_to_dict=sit_latent_args_to_dict,
            load_state_dict_fn=_load_state_dict,
            classifier_type=classifier_type,
            classifier_name=classifier_name,
            models_dir=models_dir,
            dataset_name=dataset_name,
            device=device,
            image_size=image_size,
            pytorch_normalization=pytorch_normalization,
            default_target_size=(512, 512),
            timm_target_size=(512, 512),
            verbose=False,
            list_missing_models=False,
            model_subdir=model_subdir,
        )
        self.denoiser_backend = denoiser_backend

        from diffusers.models import AutoencoderKL

        self._sit_vae_scaling = 0.18215
        self._sit_num_classes = 1000
        self._sit_vae = AutoencoderKL.from_pretrained(sit_vae_id).to(self.device).eval()
        self._sit_vae.requires_grad_(False)
        self.sit_vae_id = sit_vae_id

    def denoise(self, x_start, t, multistep: bool = False, *, enable_grad: bool = False):
        """
        Backend switch:
        - guided_diffusion/sit: defer to base implementation (uses guided-diffusion p_sample)
        - sit_latent: do one-shot denoising in VAE latent space using the official SiT checkpoint.
        """
        if self.denoiser_backend != "sit_latent":
            return super().denoise(x_start, t, multistep=multistep, enable_grad=enable_grad)

        if multistep:
            raise ValueError("sit_latent backend supports one-shot denoising only (multistep=False).")

        # x_start is already in [-1, 1] (see DiffusionRobustModelBase.forward).
        grad_ctx = torch.enable_grad() if enable_grad else torch.no_grad()
        with grad_ctx:
            batch_size = x_start.shape[0]

            # The official SiT ImageNet 256x256 checkpoints operate in SD-VAE latent space with spatial size 32x32.
            # That corresponds to 256x256 inputs (since the VAE downsamples by 8).
            #
            # If the RS pipeline feeds 224x224 images, the VAE latents become 28x28 and SiT will fail with:
            # "Input height (28) doesn't match model (32)."
            #
            # We upsample to 256x256 here to keep the denoiser backend functional.
            if x_start.shape[-2:] != (256, 256):
                x_start = torch.nn.functional.interpolate(
                    x_start, size=(256, 256), mode="bicubic", antialias=True
                )

            # Map the RS timestep `t` (from guided-diffusion schedule) to a SiT linear interpolant time τ in (0, 1):
            #
            # guided-diffusion: x_t = a * x + b * eps  => effective sigma = b/a
            # SiT Linear path:  x_τ = τ * x + (1-τ) * z, z~N(0,1) => effective sigma = (1-τ)/τ
            # Match: (1-τ)/τ = b/a  => τ = a/(a+b)
            a = float(self.diffusion.sqrt_alphas_cumprod[t])
            b = float(self.diffusion.sqrt_one_minus_alphas_cumprod[t])
            tau = a / (a + b)

            tau_t = torch.full((batch_size,), tau, device=self.device, dtype=torch.float32)

            # Encode to latents (same scaling as the SiT repo).
            z1 = self._sit_vae.encode(x_start).latent_dist.sample().mul_(self._sit_vae_scaling)
            z0 = torch.randn_like(z1)

            # Linear interpolant in latent space.
            zt = tau * z1 + (1.0 - tau) * z0

            # Unconditional denoising: use null token y = num_classes (1000 for ImageNet).
            y_null = torch.full((batch_size,), self._sit_num_classes, device=self.device, dtype=torch.long)

            # SiT predicts velocity u_t. For Linear path:
            #   u = x1 - x0
            # and we can solve: x1 = x_t + (1 - t) * u
            u = self.model(zt, tau_t, y_null)
            z1_hat = zt + (1.0 - tau_t).view(batch_size, 1, 1, 1) * u

            x_hat = self._sit_vae.decode(z1_hat / self._sit_vae_scaling).sample
            return x_hat.clamp(-1, 1)