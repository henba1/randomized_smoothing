"""ImageNet Hydra entry point for LoRA fine-tuning SiT as a purifier."""

from omegaconf import DictConfig

from hydra import main as hydra_main
from shared.finetune_sit_lora_hydra import main as shared_main


@hydra_main(version_base=None, config_path="../hydra/conf", config_name="finetune_sit_lora_imagenet")
def hydra_entry(cfg: DictConfig) -> None:
    shared_main(cfg)


if __name__ == "__main__":
    hydra_entry()

