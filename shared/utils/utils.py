import os
import subprocess

import torch


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


def get_device_with_diagnostics() -> torch.device:
    """Get CUDA device if available, with detailed diagnostics.

    Checks CUDA availability and prints diagnostic information including:
    - Device availability and count
    - Device name
    - CUDA_VISIBLE_DEVICES environment variable
    - PyTorch version and CUDA compilation info
    - nvidia-smi output if CUDA is not available but GPU is visible

    Returns:
        torch.device: CUDA device if available, otherwise CPU device
    """
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = torch.device("cuda")
        print(f"CUDA is available. Using device: {device}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    else:
        device = torch.device("cpu")
        print(f"WARNING: CUDA is not available. Using device: {device}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch CUDA compiled: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")
        try:
            nvidia_smi = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if nvidia_smi.returncode == 0:
                print(f"nvidia-smi output: {nvidia_smi.stdout.strip()}")
                print("WARNING: GPU is visible via nvidia-smi but PyTorch cannot access it!")
                print("This may indicate a PyTorch-CUDA version mismatch or missing CUDA libraries.")
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            print("nvidia-smi not available or failed")
    
    return device


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


