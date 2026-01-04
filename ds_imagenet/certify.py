"""ImageNet certification wrapper - loads defaults from YAML and calls shared certify function."""

import yaml
from pathlib import Path

from shared.certify import main, create_argument_parser

if __name__ == "__main__":
    # Load defaults from YAML file
    defaults_path = Path(__file__).parent / "imagenet_params.yaml"
    with open(defaults_path, "r") as f:
        defaults = yaml.safe_load(f)
    
    parser = create_argument_parser()
    args = parser.parse_args()
    main(args=args, defaults=defaults)
