from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
from ada_verona import apply_pytorch_normalization
from transformers import AutoModelForImageClassification

from shared.classifiers.onnx_classifier import load_onnx_classifier
from shared.classifiers.pytorch_classifier import PyTorchClassifierWrapper
from shared.utils.utils import get_diffusion_model_path_name_tuple, print_huggingface_device_status


class DiffusionRobustModelBase(nn.Module):
    def __init__(
        self,
        *,
        diffusion_args: object,
        create_model_and_diffusion: Callable[..., tuple[nn.Module, object]],
        model_and_diffusion_defaults: Callable[[], dict],
        args_to_dict: Callable[[object, object], dict],
        load_state_dict_fn: Callable[[str, torch.device], dict],
        classifier_type: str,
        classifier_name: str | None,
        models_dir: str | None,
        dataset_name: str | None,
        device: torch.device | None,
        image_size: tuple[int, int] | None,
        pytorch_normalization: str,
        default_target_size: tuple[int, int],
        timm_target_size: tuple[int, int] | None = None,
        verbose: bool = False,
        list_missing_models: bool = False,
    ):
        super().__init__()
        if pytorch_normalization not in {"none", "sdpcrown"}:
            raise ValueError("pytorch_normalization must be one of {'none', 'sdpcrown'}")
        self.pytorch_normalization = pytorch_normalization

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.default_target_size = default_target_size
        self.timm_target_size = timm_target_size

        model, diffusion = create_model_and_diffusion(
            **args_to_dict(diffusion_args, model_and_diffusion_defaults().keys())
        )
        diffusion_model_path, _ = get_diffusion_model_path_name_tuple(dataset_name)
        state_dict = load_state_dict_fn(diffusion_model_path, device)
        model.load_state_dict(state_dict)
        model.eval().to(device)

        self.model = model
        self.diffusion = diffusion
        self.model.requires_grad_(False)

        self.classifier_type = classifier_type
        self.classifier_name = classifier_name
        self.classifier = self._load_classifier(
            classifier_type=classifier_type,
            classifier_name=classifier_name,
            models_dir=models_dir,
            image_size=image_size,
            device=device,
            verbose=verbose,
            list_missing_models=list_missing_models,
        )
        if isinstance(self.classifier, nn.Module):
            self.classifier.requires_grad_(False)

    def _load_classifier(
        self,
        *,
        classifier_type: str,
        classifier_name: str | None,
        models_dir: str | None,
        image_size: tuple[int, int] | None,
        device: torch.device,
        verbose: bool,
        list_missing_models: bool,
    ) -> nn.Module:
        if classifier_type == "onnx":
            if classifier_name is None:
                raise ValueError("classifier_name must be specified when using ONNX classifier")
            if models_dir is None:
                raise ValueError("models_dir must be specified when using ONNX classifier")
            if verbose:
                print(f"\n{'='*70}")
                print(f"Loading ONNX classifier: {classifier_name} from {models_dir}")
            device_str = "cuda" if device.type == "cuda" else "cpu"
            classifier = load_onnx_classifier(classifier_name, str(models_dir), device=device_str)
            classifier.eval()
            if verbose:
                print(f"{'='*70}\n")
            return classifier

        if classifier_type == "pytorch":
            if classifier_name is None:
                raise ValueError("classifier_name must be specified when using PyTorch classifier")
            if models_dir is None:
                raise ValueError("models_dir must be specified when using PyTorch classifier")
            if image_size is None:
                raise ValueError("image_size must be provided when using PyTorch classifier")
            if verbose:
                print(f"\n{'='*70}")
                print(f"Loading PyTorch classifier: {classifier_name} from {models_dir}")
            model_path = Path(models_dir) / f"{classifier_name}.pth"
            if not model_path.exists():
                if list_missing_models:
                    available_models = list(Path(models_dir).glob("*.pth"))
                    raise FileNotFoundError(
                        f"PyTorch model {model_path} not found. "
                        f"Available models: {[m.name for m in available_models]}"
                    )
                raise FileNotFoundError(f"PyTorch model not found: {model_path}")
            pytorch_model = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(pytorch_model, (OrderedDict, dict)) and not isinstance(
                pytorch_model, nn.Module
            ):
                if self.pytorch_normalization == "sdpcrown":
                    from ada_verona import load_sdpcrown_pytorch_model

                    pytorch_model = load_sdpcrown_pytorch_model(model_path, device)
                else:
                    raise ValueError(
                        f"Cannot load state_dict for {model_path}"
                    )
            expected_height = image_size[1]
            expected_width = image_size[0]
            classifier = PyTorchClassifierWrapper(
                pytorch_model,
                expected_height=expected_height,
                expected_width=expected_width,
            )
            classifier.eval().to(device)
            if verbose:
                print(f"{'='*70}\n")
            return classifier

        if classifier_type == "huggingface":
            if classifier_name is None:
                raise ValueError("classifier_name must be specified when using HuggingFace classifier")
            model_id = classifier_name
            if verbose:
                print(f"\n{'='*70}")
                print(f"Loading HuggingFace classifier: {model_id}")
                print("Loading from default cache: ~/.cache/huggingface")
            try:
                classifier = AutoModelForImageClassification.from_pretrained(
                    model_id, local_files_only=True
                )
            except Exception:
                if verbose:
                    print("Model not found in local cache, downloading from HuggingFace Hub...")
                classifier = AutoModelForImageClassification.from_pretrained(model_id)
            classifier.eval().to(device)
            print_huggingface_device_status(classifier, model_id)
            if verbose:
                print(f"{'='*70}\n")
            return classifier

        if classifier_type == "timm":
            if classifier_name is None:
                raise ValueError("classifier_name must be specified for timm")
            try:
                import timm
            except ImportError as exc:
                raise ImportError("timm is required for classifier_type='timm'") from exc
            classifier = timm.create_model(classifier_name, pretrained=True)
            classifier.eval().to(device)
            return classifier

        raise ValueError(f"Unknown classifier_type: {classifier_type}")

    def _apply_pytorch_normalization(self, imgs: torch.Tensor) -> torch.Tensor:
        return apply_pytorch_normalization(imgs, self.pytorch_normalization)

    def _resolve_target_size(self) -> tuple[int, int]:
        if self.classifier_type in {"onnx", "pytorch"}:
            return (self.classifier.expected_height, self.classifier.expected_width)
        if self.classifier_type == "timm" and self.timm_target_size is not None:
            return self.timm_target_size
        return self.default_target_size

    def forward(
        self,
        x: torch.Tensor,
        t: int,
        *,
        enable_grad: bool = False,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ):
        grad_ctx = torch.enable_grad() if enable_grad else torch.no_grad()
        with grad_ctx:
            x_in = x * 2 - 1  # output is in [-1,1]
            imgs = self.denoise(x_in, t, enable_grad=enable_grad, noise=noise, generator=generator)

            target_size = self._resolve_target_size()
            imgs = torch.nn.functional.interpolate(
                imgs, target_size, mode="bicubic", antialias=True
            )

            if self.classifier_type in {"onnx", "pytorch"}:
                imgs = imgs * 0.5 + 0.5  # Convert [-1, 1] to [0, 1]
                if self.classifier_type == "pytorch":
                    imgs = self._apply_pytorch_normalization(imgs)

            out = self.classifier(imgs)
            return out.logits if hasattr(out, "logits") else out

    def denoise(
        self,
        x_start: torch.Tensor,
        t: int,
        multistep: bool = False,
        *,
        enable_grad: bool = False,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ):
        grad_ctx = torch.enable_grad() if enable_grad else torch.no_grad()
        with grad_ctx:
            t_batch = torch.tensor([t] * len(x_start), device=self.device)
            if noise is None:
                if generator is None:
                    noise = torch.randn_like(x_start)
                else:
                    noise = torch.randn(
                        x_start.shape, dtype=x_start.dtype, device=x_start.device, generator=generator
                    )
            x_t_start = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)

            if multistep:
                out = x_t_start
                for i in range(t)[::-1]:
                    t_batch = torch.tensor([i] * len(x_start), device=self.device)
                    out = self.diffusion.p_sample(self.model, out, t_batch, clip_denoised=True)["sample"]
            else:
                out = self.diffusion.p_mean_variance(
                    self.model, x_t_start, t_batch, clip_denoised=True
                )["pred_xstart"]

            return out

