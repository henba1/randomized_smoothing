"""CIFAR-10 Hydra certification wrapper."""

from omegaconf import DictConfig

from hydra import main as hydra_main
from shared.certify_hydra import main as shared_main


@hydra_main(version_base=None, config_path="../hydra/conf", config_name="certify_cifar10")
def hydra_entry(cfg: DictConfig) -> None:
    """Hydra entry point for CIFAR-10."""
    shared_main(cfg)


if __name__ == "__main__":
    try:
        hydra_entry()
    except ImportError:
        print("Error: hydra-core not installed. Install with: pip install hydra-core hydra-submitit-launcher")
        print("Falling back to manual config loading...")
        from hydra import compose, initialize
        with initialize(config_path="../hydra/conf", version_base=None):
            cfg = compose(config_name="certify_cifar10")
            shared_main(cfg)
