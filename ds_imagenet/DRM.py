from pathlib import Path

import timm
import torch
import torch.nn as nn
from .guided_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from transformers import AutoModelForImageClassification
from shared.classifiers.onnx_classifier import load_onnx_classifier
from shared.classifiers.pytorch_classifier import PyTorchClassifierWrapper
from shared.utils.utils import (
    get_diffusion_model_path_name_tuple,
    print_huggingface_device_status,
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


class DiffusionRobustModel(nn.Module):
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
        super().__init__()
        if pytorch_normalization not in {"none", "sdpcrown"}:
            raise ValueError(
                "pytorch_normalization must be one of {'none', 'sdpcrown'}"
            )
        self.pytorch_normalization = pytorch_normalization
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
         # instantiate the prepended DDPM
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(Args(), model_and_diffusion_defaults().keys())
        )
        diffusion_model_path, _ = get_diffusion_model_path_name_tuple(dataset_name)
        state_dict = torch.load(diffusion_model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval().to(device)

        self.model = model
        self.diffusion = diffusion
        self.model.requires_grad_(False)

        if classifier_type == "onnx":
            if classifier_name is None or models_dir is None:
                raise ValueError("classifier_name and models_dir required for ONNX")
            provider_device = "cuda" if device.type == "cuda" else "cpu"
            classifier = load_onnx_classifier(
                classifier_name, str(models_dir), device=provider_device
            )
            classifier.eval()
            self.classifier_type = "onnx"
            self.classifier_name = classifier_name
        elif classifier_type == "pytorch":
            if classifier_name is None or models_dir is None:
                raise ValueError("classifier_name and models_dir required for PyTorch")
            if image_size is None:
                raise ValueError("image_size must be provided for PyTorch classifier")
            model_path = Path(models_dir) / f"{classifier_name}.pth"
            if not model_path.exists():
                raise FileNotFoundError(f"PyTorch model not found: {model_path}")
            pytorch_model = torch.load(model_path, map_location=device, weights_only=False)
            if not isinstance(pytorch_model, nn.Module):
                raise ValueError(f"Loaded object from {model_path} is not a torch.nn.Module")
            expected_height = image_size[1]
            expected_width = image_size[0]
            classifier = PyTorchClassifierWrapper(
                pytorch_model,
                expected_height=expected_height,
                expected_width=expected_width,
            )
            classifier.eval().to(device)
            self.classifier_type = "pytorch"
            self.classifier_name = classifier_name
        elif classifier_type == "huggingface":
            if classifier_name is None:
                raise ValueError("classifier_name must be specified for HuggingFace")
            model_id = classifier_name
            try:
                classifier = AutoModelForImageClassification.from_pretrained(
                    model_id, local_files_only=True
                )
            except Exception:
                classifier = AutoModelForImageClassification.from_pretrained(model_id)
            classifier.eval().to(device)
            print_huggingface_device_status(classifier, model_id)
            self.classifier_type = "huggingface"
            self.classifier_name = model_id
        else:
            if classifier_name is None:
                raise ValueError("classifier_name must be specified for timm")
            classifier = timm.create_model(classifier_name, pretrained=True)
            classifier.eval().to(device)
            self.classifier_type = "timm"
            self.classifier_name = classifier_name

        self.classifier = classifier
        if isinstance(self.classifier, nn.Module):
            self.classifier.requires_grad_(False)

    def _apply_pytorch_normalization(self, imgs: torch.Tensor) -> torch.Tensor:
        if self.pytorch_normalization == "sdpcrown":
            means = torch.tensor([125.3, 123.0, 113.9], device=imgs.device, dtype=imgs.dtype) / 255
            stds = torch.tensor([0.225, 0.225, 0.225], device=imgs.device, dtype=imgs.dtype)
            return (imgs - means.view(1, 3, 1, 1)) / stds.view(1, 3, 1, 1)
        return imgs

    def forward(self, x, t, *, enable_grad: bool = False):
        grad_ctx = torch.enable_grad() if enable_grad else torch.no_grad()
        with grad_ctx:
            x_in = x * 2 - 1
            imgs = self.denoise(x_in, t, enable_grad=enable_grad)

            if self.classifier_type == "onnx":
                target_size = (self.classifier.expected_height, self.classifier.expected_width)
            elif self.classifier_type == "pytorch":
                target_size = (self.classifier.expected_height, self.classifier.expected_width)
            elif self.classifier_type == "timm":
                target_size = (512, 512)
            else:
                target_size = (512, 512)

            imgs = torch.nn.functional.interpolate(
                imgs, target_size, mode="bicubic", antialias=True
            )

            if self.classifier_type in ["onnx", "pytorch"]:
                imgs = imgs * 0.5 + 0.5  # convert [-1,1] to [0,1]
                if self.classifier_type == "pytorch":
                    imgs = self._apply_pytorch_normalization(imgs)

            out = self.classifier(imgs)
            return out.logits if hasattr(out, "logits") else out

    def denoise(self, x_start, t, multistep: bool = False, *, enable_grad: bool = False):
        grad_ctx = torch.enable_grad() if enable_grad else torch.no_grad()
        with grad_ctx:
            t_batch = torch.tensor([t] * len(x_start), device=self.device)
            noise = torch.randn_like(x_start)
            x_t_start = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)

            if multistep:
                out = x_t_start
                for i in range(t)[::-1]:
                    t_batch = torch.tensor([i] * len(x_start), device=self.device)
                    out = self.diffusion.p_sample(self.model, out, t_batch, clip_denoised=True)["sample"]
            else:
                out = self.diffusion.p_sample(self.model, x_t_start, t_batch, clip_denoised=True)["pred_xstart"]

            return out