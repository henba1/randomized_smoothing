"""CIFAR-10 Hydra PGD-EOT attack wrapper."""

from omegaconf import DictConfig

from hydra import main as hydra_main
from shared.attack_hydra import main as shared_main


@hydra_main(version_base=None, config_path="../hydra/conf", config_name="attack_cifar10")
def hydra_entry(cfg: DictConfig) -> None:
    shared_main(cfg)


if __name__ == "__main__":
    hydra_entry()







