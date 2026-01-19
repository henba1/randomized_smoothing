"""CIFAR-10 certification wrapper - loads experimentparameters from YAML and calls shared certify function."""

from pathlib import Path

import yaml

from shared.certify import create_argument_parser, main

if __name__ == "__main__":
    defaults_path = Path(__file__).parent / "cifar10_params.yaml"
    with open(defaults_path) as f:
        defaults = yaml.safe_load(f)
    
    parser = create_argument_parser()
    args = parser.parse_args()
    main(args=args, defaults=defaults)
