import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset


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

    # Find the .pt file in the directory #TODO: ASSUMING ONE FILE IN DIR 
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


def get_balanced_sample(dataset, train_bool: bool = False, seed: int = 42, sample_size: int = 100):
    """Get a balanced sample from the dataset using StratifiedShuffleSplit.

    Args:
        dataset: The dataset to sample from
        train_bool: Whether this is a training set (True) or test set (False)
        seed: Random seed for reproducibility
        sample_size: Number of samples to select

    Returns:
        tuple: (Subset of the dataset, array of original indices)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Extract the labels
    labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])

    # Use StratifiedShuffleSplit to create balanced subsets
    if train_bool:
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=seed)
        for train_idx, _ in splitter.split(np.zeros(len(labels)), labels):
            balanced_sample_idx = train_idx
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=sample_size, random_state=seed
        )
        for _, test_idx in splitter.split(np.zeros(len(labels)), labels):
            balanced_sample_idx = test_idx

    # Create a subset of the original dataset using the balanced indices
    balanced_dataset = Subset(dataset, balanced_sample_idx)

    return balanced_dataset, balanced_sample_idx


def get_sample(dataset, seed: int = 42, sample_size: int = 100):
    """Get a random sample from the dataset without stratification.

    Unlike get_balanced_sample(), this function does not ensure balanced class distribution
    and simply returns a random subset of the data.

    Args:
        dataset: The dataset to sample from
        seed: Random seed for reproducibility
        sample_size: Number of samples to select

    Returns:
        tuple: (Subset of the dataset, array of original indices)

    Raises:
        ValueError: If sample_size exceeds dataset size
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Random sampling without stratification
    dataset_length = len(dataset)
    if sample_size > dataset_length:
        raise ValueError(
            f"Requested sample size {sample_size} exceeds dataset size {dataset_length}"
        )

    # Generate random indices
    all_indices = np.arange(dataset_length)
    sample_idx = np.random.choice(all_indices, size=sample_size, replace=False)

    # Create subset of original dataset using the sampled indices
    sampled_dataset = Subset(dataset, sample_idx)

    return sampled_dataset, sample_idx


def create_experiment_folder(
    results_dir: str,
    safe_classifier_name: str,
    sigma: float,
    alpha: float,
    N0: int,
    N: int,
    batch_size: int,
) -> str:
    """Create experiment folder for given configuration and return the folder path.

    This function creates a folder structure to organize experiment results by
    configuration. All files with the same hyperparameters are stored in the
    same folder, simplifying file naming.

    Args:
        results_dir: Base results directory
        safe_classifier_name: Safe classifier name (slashes/hyphens replaced)
        sigma: Noise level parameter
        alpha: Failure probability
        N0: Number of initial samples
        N: Number of samples for certification
        batch_size: Batch size for processing

    Returns:
        str: Path to the experiment folder
    """
    # Ensure base results directory exists
    os.makedirs(results_dir, exist_ok=True)

    experiment_folder_name = (
        f"{safe_classifier_name}_{sigma}_{alpha}_{N0}_{N}_{batch_size}"
    )
    experiment_folder_path = os.path.join(results_dir, experiment_folder_name)

    os.makedirs(experiment_folder_path, exist_ok=True)

    return experiment_folder_path

