import torch
from .guided_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from shared.drm_base import DiffusionRobustModelBase


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
        classifier_type: str = "timm",
        classifier_name: str | None = "beit_large_patch16_512",
        models_dir: str | None = None,
        dataset_name: str | None = "ImageNet",
        device: torch.device | None = None,
        image_size=None,
        pytorch_normalization: str = "none",
    ):
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
            reject_state_dict=False,
        )