import torch

from .improved_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from shared.drm_base import DiffusionRobustModelBase

class Args:
    image_size=32
    num_channels=128
    num_res_blocks=3
    num_heads=4
    num_heads_upsample=-1
    attention_resolutions="16,8"
    dropout=0.3
    learn_sigma=True
    sigma_small=False
    class_cond=False
    diffusion_steps=4000
    noise_schedule="cosine"
    timestep_respacing=""
    use_kl=False
    predict_xstart=False
    rescale_timesteps=True
    rescale_learned_sigmas=True
    use_checkpoint=False
    use_scale_shift_norm=True


def _load_state_dict(diffusion_model_path: str, device: torch.device) -> dict:
    return torch.load(diffusion_model_path, weights_only=True, map_location=device)


class DiffusionRobustModel(DiffusionRobustModelBase):
    def __init__(self, 
        classifier_type: str = "huggingface", 
        classifier_name: str | None = "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10", 
        models_dir: str | None = None, 
        dataset_name: str | None = None, 
        device: torch.device | None = None, 
        image_size: tuple[int, int] | None = None,
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
            default_target_size=(224, 224),
            timm_target_size=None,
            verbose=True,
            list_missing_models=True,
            reject_state_dict=True,
        )