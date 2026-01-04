"""ImageNet Hydra certification wrapper."""

from hydra import main as hydra_main
from omegaconf import DictConfig

from shared.certify_hydra import main as shared_main


@hydra_main(version_base=None, config_path="../hydra/conf", config_name="certify_imagenet")
def hydra_entry(cfg: DictConfig) -> None:
    """Hydra entry point for ImageNet."""
    shared_main(cfg)


if __name__ == "__main__":
    try:
        hydra_entry()
    except ImportError:
        print("Error: hydra-core not installed.")
        from hydra import compose, initialize
        with initialize(config_path="../hydra/conf", version_base=None):
            cfg = compose(config_name="certify_imagenet")
            shared_main(cfg)

