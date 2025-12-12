import os

import torch
import torch.nn as nn

from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
from transformers import AutoModelForImageClassification
from onnx_classifier import load_onnx_classifier
from utils import print_huggingface_device_status, get_diffusion_model_path_name_tuple


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
    def __init__(self, classifier_type="huggingface", classifier_name=None, models_dir=None, dataset_name=None):
        super().__init__()
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(Args(), model_and_diffusion_defaults().keys())
        )
        diffusion_model_path, _ = get_diffusion_model_path_name_tuple(dataset_name)
        model.load_state_dict(
            torch.load(diffusion_model_path, weights_only=True)
        )
        model.eval().cuda()

        self.model = model 
        self.diffusion = diffusion 

        # Load classifier based on type
        if classifier_type == "onnx":
            if classifier_name is None:
                raise ValueError("classifier_name must be specified when using ONNX classifier")
            if models_dir is None:
                raise ValueError("models_dir must be specified when using ONNX classifier")
            print(f"\n{'='*70}")
            print(f"Loading ONNX classifier: {classifier_name} from {models_dir}")
            classifier = load_onnx_classifier(classifier_name, models_dir)
            classifier.eval().cuda()
            print(f"{'='*70}\n")
            self.classifier_type = "onnx"
            self.classifier_name = classifier_name
        else:  # default to huggingface
            try:
                model_id = classifier_name
            except Exception as e:
                raise ValueError(f"Error loading HuggingFace classifier: {e}. Please check the model ID and if the model is available on HuggingFace.")
            print(f"\n{'='*70}")
            print(f"Loading HuggingFace classifier: {model_id}")
            if models_dir:
                # Load from local directory
                print(f"Loading from local cache: {models_dir}")
                os.environ['HF_HOME'] = models_dir
                classifier = AutoModelForImageClassification.from_pretrained(
                    model_id,
                    local_files_only=True
                )
            else:
                # Load remote model from HuggingFace
                classifier = AutoModelForImageClassification.from_pretrained(model_id)
            classifier.eval().cuda()
            print_huggingface_device_status(classifier, model_id)
            print(f"{'='*70}\n")
            self.classifier_type = "huggingface"
            self.classifier_name = model_id

        self.classifier = classifier

    def forward(self, x, t):
        x_in = x * 2 -1 #Output is in [-1,1]
        imgs = self.denoise(x_in, t)

        # Resize images based on classifier type
        if self.classifier_type == "onnx":
            # Use the ONNX model's expected input size
            target_size = (self.classifier.expected_height, self.classifier.expected_width)
        else:
            # HuggingFace ViT expects 224x224 as it was trained on ImageNet
            target_size = (224, 224)
        #upscale the images to the target size
        imgs = torch.nn.functional.interpolate(imgs, target_size, mode='bicubic', antialias=True)
        
        # Convert back to [0,1] for ONNX models (trained on [0,1] data from JAIR_code)
        if self.classifier_type == "onnx":
            imgs = imgs * 0.5 + 0.5  # Convert [-1, 1] to [0, 1]   !!! This is not needed for the hf vit, investigate exactly why before submitting

        with torch.no_grad():
            out = self.classifier(imgs)

        return out.logits

    def denoise(self, x_start, t, multistep=False):
        t_batch = torch.tensor([t] * len(x_start)).cuda()

        noise = torch.randn_like(x_start)

        x_t_start = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)

        with torch.no_grad():
            if multistep:
                out = x_t_start
                for i in range(t)[::-1]:
                    t_batch = torch.tensor([i] * len(x_start)).cuda()
                    out = self.diffusion.p_sample(
                        self.model,
                        out,
                        t_batch,
                        clip_denoised=True
                    )['sample']
            else:
                out = self.diffusion.p_sample(
                    self.model,
                    x_t_start,
                    t_batch,
                    clip_denoised=True
                )['pred_xstart']

        return out