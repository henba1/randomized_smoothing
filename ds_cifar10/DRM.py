from pathlib import Path

import torch
import torch.nn as nn

from .improved_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from shared.classifiers.onnx_classifier import load_onnx_classifier
from transformers import AutoModelForImageClassification
from shared.classifiers.pytorch_classifier import PyTorchClassifierWrapper
from shared.utils.utils import get_diffusion_model_path_name_tuple, print_huggingface_device_status

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


class DiffusionRobustModel(nn.Module):
    def __init__(self, 
        classifier_type: str = "huggingface", 
        classifier_name: str | None = "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10", 
        models_dir: str | None = None, 
        dataset_name: str | None = None, 
        device: torch.device | None = None, 
        image_size: tuple[int, int] | None = None,
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
        model.load_state_dict(
            torch.load(diffusion_model_path, weights_only=True, map_location=device)
        )
        model.eval().to(device)

        self.model = model 
        self.diffusion = diffusion 
        self.model.requires_grad_(False)

        # Load classifier based on type
        if classifier_type == "onnx":
            if classifier_name is None:
                raise ValueError("classifier_name must be specified when using ONNX classifier")
            if models_dir is None:
                raise ValueError("models_dir must be specified when using ONNX classifier")
            print(f"\n{'='*70}")
            print(f"Loading ONNX classifier: {classifier_name} from {models_dir}")
            device_str = "cuda" if device.type == "cuda" else "cpu"
            classifier = load_onnx_classifier(classifier_name, str(models_dir), device=device_str)
            classifier.eval()
            print(f"{'='*70}\n")
            self.classifier_type = "onnx"
            self.classifier_name = classifier_name
        elif classifier_type == "pytorch":
            if classifier_name is None:
                raise ValueError("classifier_name must be specified when using PyTorch classifier")
            if models_dir is None:
                raise ValueError("models_dir must be specified when using PyTorch classifier")
            print(f"\n{'='*70}")
            print(f"Loading PyTorch classifier: {classifier_name} from {models_dir}")
            model_path = Path(models_dir) / f"{classifier_name}.pth"
            if not model_path.exists():
                available_models = list(Path(models_dir).glob("*.pth"))
                raise FileNotFoundError(
                    f"PyTorch model {model_path} not found. "
                    f"Available models: {[m.name for m in available_models]}"
                )
            pytorch_model = torch.load(model_path, map_location=device, weights_only=False)
            # Handle case where .pth file contains state_dict vs full model
            if isinstance(pytorch_model, dict) and 'state_dict' in pytorch_model:
                raise ValueError(
                    f"Model {model_path} is a state_dict. "
                    "Please provide a full model (architecture + weights) saved with torch.save(model, ...)"
                )
            if not isinstance(pytorch_model, nn.Module):
                raise ValueError(f"Loaded object from {model_path} is not a torch.nn.Module")
            if image_size is None:
                raise ValueError("image_size must be provided when using PyTorch classifier")
            expected_height = image_size[1] 
            expected_width = image_size[0] #width
            classifier = PyTorchClassifierWrapper(pytorch_model, expected_height=expected_height, expected_width=expected_width)
            classifier.eval().to(device)
            print(f"{'='*70}\n")
            self.classifier_type = "pytorch"
            self.classifier_name = classifier_name
        else: #HF
            if classifier_name is None:
                raise ValueError("classifier_name must be specified when using HuggingFace classifier")
            
            model_id = classifier_name
            print(f"\n{'='*70}")
            print(f"Loading HuggingFace classifier: {model_id}")
            
            try:
                print(f"Loading from default cache: ~/.cache/huggingface")
                classifier = AutoModelForImageClassification.from_pretrained(
                    model_id,
                    local_files_only=True
                )
            except Exception as e:
                print(f"Model not found in local cache, downloading from HuggingFace Hub...")
                classifier = AutoModelForImageClassification.from_pretrained(model_id)
            classifier.eval().to(device)
            print_huggingface_device_status(classifier, model_id)
            print(f"{'='*70}\n")
            self.classifier_type = "huggingface"
            self.classifier_name = model_id

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
            x_in = x * 2 - 1  # output is in [-1,1]
            imgs = self.denoise(x_in, t, enable_grad=enable_grad)

            # Resize images based on classifier type
            if self.classifier_type == "onnx":
                # Use the ONNX model's expected input size
                target_size = (self.classifier.expected_height, self.classifier.expected_width)
            elif self.classifier_type == "pytorch":
                # Use the PyTorch model's expected input size
                target_size = (self.classifier.expected_height, self.classifier.expected_width)
            else:
                # HuggingFace ViT expects 224x224 as it was trained on ImageNet, #TODO hardcoded for now
                target_size = (224, 224)

            # Upscale the images to  target size
            imgs = torch.nn.functional.interpolate(imgs, target_size, mode="bicubic", antialias=True)
            
            # Convert back to [0,1] for ONNX and PyTorch models (assume trained on [0,1] data)
            if self.classifier_type in ["onnx", "pytorch"]:
                imgs = imgs * 0.5 + 0.5  # Convert [-1, 1] to [0, 1]
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