from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ada_verona import get_dataset_config
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset


@dataclass(frozen=True)
class CertifiedSample:
    image_id: int
    original_label: int
    predicted_class: int
    epsilon_value: float


@dataclass(frozen=True)
class CertifyRunData:
    samples: list[CertifiedSample]
    summary_sigma: float | None


def _maybe_float(x: str | None) -> float | None:
    if x is None:
        return None
    x = x.strip()
    if not x:
        return None
    return float(x)


def load_certify_run(run_path: Path) -> CertifyRunData:
    """
    Load the output of a randomized smoothing certify run.

    Expected files:
    - result_df.csv: contains per-sample certified radii (epsilon_value) and predicted_class.
    - summary_df.csv (optional): contains run-level metadata, including sigma.

    Notes:
    - `epsilon_value` is interpreted as the certified L2 radius in the input space (pixel space).
    - `image_id` is assumed to refer to the *original dataset index*.
    """
    run_path = Path(run_path)
    if run_path.is_dir():
        result_path = run_path / "result_df.csv"
        summary_path = run_path / "summary_df.csv"
    else:
        result_path = run_path
        summary_path = result_path.parent / "summary_df.csv"

    if not result_path.exists():
        raise FileNotFoundError(f"Could not find certify results file at: {result_path}")

    summary_sigma = None
    if summary_path.exists():
        with summary_path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows:
            summary_sigma = _maybe_float(rows[0].get("sigma"))

    with result_path.open(newline="") as f:
        reader = csv.DictReader(f)
        samples: list[CertifiedSample] = []
        for row in reader:
            samples.append(
                CertifiedSample(
                    image_id=int(row["image_id"]),
                    original_label=int(row["original_label"]),
                    predicted_class=int(row["predicted_class"]),
                    epsilon_value=float(row["epsilon_value"]),
                )
            )

    if not samples:
        raise ValueError(f"No rows found in certify results: {result_path}")

    seen: set[int] = set()
    deduped: list[CertifiedSample] = []
    for s in samples:
        if s.image_id in seen:
            continue
        seen.add(s.image_id)
        deduped.append(s)

    return CertifyRunData(samples=deduped, summary_sigma=summary_sigma)


def make_dataset_subset_from_image_ids(
    *,
    dataset_name: str,
    dataset_dir: Path | str,
    train_bool: bool,
    image_ids: list[int],
    image_size: tuple[int, int] | None,
    flatten: bool,
) -> tuple[Subset, np.ndarray]:
    """
    Create a torch.utils.data.Subset using explicit dataset indices (image_ids).

    This mirrors the basic dataset/transform behavior of `rs_rd.utils.experiment_utils.get_sample`,
    but uses a provided list of indices instead of random sampling.
    """
    dataset_config = get_dataset_config()
    if dataset_name not in dataset_config:
        raise ValueError(f"Unsupported dataset: '{dataset_name}'. Supported: {', '.join(dataset_config.keys())}")

    config = dataset_config[dataset_name]
    target_size = image_size if image_size is not None else config["default_size"]

    transform_list = [transforms.Resize(target_size), transforms.ToTensor()]
    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.flatten()))
    data_transforms = transforms.Compose(transform_list)

    dataset_class = config["class"]
    if dataset_name == "ImageNet":
        split_dir = "train" if train_bool else "val"
        dataset_path = Path(dataset_dir) / split_dir
        if not dataset_path.exists():
            raise FileNotFoundError(f"ImageNet {split_dir} directory not found at {dataset_path}")
        if dataset_class is not ImageFolder:
            dataset_class = ImageFolder
        torch_dataset = dataset_class(root=str(dataset_path), transform=data_transforms)
    else:
        torch_dataset = dataset_class(root=str(dataset_dir), train=train_bool, download=False, transform=data_transforms)

    dataset_length = len(torch_dataset)
    bad = [idx for idx in image_ids if idx < 0 or idx >= dataset_length]
    if bad:
        raise ValueError(f"Some image_id indices are out of bounds for {dataset_name} (len={dataset_length}): {bad[:10]}")

    original_indices = np.asarray(image_ids, dtype=np.int64)
    return Subset(torch_dataset, original_indices), original_indices

