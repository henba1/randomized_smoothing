import os

import numpy as np
import torch
from torch.utils.data import Dataset


def print_huggingface_device_status(model: torch.nn.Module, model_id: str) -> str:
    """Print and return the actual device where HuggingFace model is located.

    Args:
        model: The PyTorch model
        model_id: The model identifier string

    Returns:
        String indicating the actual device being used
    """
    device = next(model.parameters()).device
    device_str = str(device)
    actual_device = "GPU (CUDA)" if device.type == "cuda" else "CPU"
    status = "Passed:" if device.type == "cuda" else "Warning: CPU Fallback"
    print(
        f"{status} HuggingFace Model {model_id} put on Device: {actual_device} "
        f"(PyTorch device: {device_str})"
    )
    return actual_device


def get_diffusion_model_path_name_tuple(dataset_name: str) -> tuple[str, str]:
    """Get the path to the diffusion model using $PRJS environment variable.

    Args:
        dataset_name: Name of the dataset

    Returns:
        tuple: (Path to the diffusion model .pt file, Name of the diffusion model)

    Raises:
        ValueError: If PRJS is not set, directory doesn't exist,
                   or no/multiple .pt files are found
    """
    prjs = os.getenv("PRJS")
    if not prjs:
        raise ValueError("PRJS environment variable not set")

    model_dir = os.path.join(prjs, "models", dataset_name, "DDPM")

    if not os.path.isdir(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")

    # Find the .pt file in the directory #TODO: this assumes only one file in dir
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]

    if len(model_files) == 0:
        raise ValueError(f"No .pt files found in {model_dir}")
    if len(model_files) > 1:
        raise ValueError(
            f"Multiple .pt files found in {model_dir}: {model_files}. "
            f"Expected only one for the dataset {dataset_name}"
        )

    return os.path.join(model_dir, model_files[0]), str(model_files[0].replace(".pt", ""))


def print_onnx_device_status(provider_list: list, device_requested: str) -> str:
    """Print and return the actual device being used by ONNX Runtime.

    Args:
        provider_list: List of ONNX Runtime execution providers
        device_requested: The device that was requested ("cuda" or "cpu")

    Returns:
        String indicating the actual device being used
    """
    actual_device = "GPU (CUDA)" if "CUDAExecutionProvider" in provider_list else "CPU"
    status = (
        "Passed:"
        if actual_device == "GPU (CUDA)" and device_requested == "cuda"
        else "Warning: CPU Fallback"
    )
    print(f"{status} ONNX Model Device: {actual_device} (requested: {device_requested})")
    return actual_device


def override_args_with_cli(defaults: dict, args) -> dict:
    """Override default values with command-line arguments if provided.

    Args:
        defaults: Dictionary of default parameter values
        args: Parsed command-line arguments object (or None)

    Returns:
        Dictionary with overridden values (same keys as defaults, in same order)
    """
    if args is None:
        return defaults

    params = defaults.copy()
    params.update({k: v for k, v in vars(args).items() if v is not None and k in params})
    return params


