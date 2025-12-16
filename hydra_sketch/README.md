# Sweep Configuration Files

This directory contains predefined sweep configurations for running certification experiments.

## Usage

All sweep configs require the `-m` (multirun) flag to enable parameter sweeping:

```bash
python certify_hydra.py --config-name sweeps/<config_name> -m
```

## Available Sweep Configs

### `onnx_all_models.yaml`
Sweeps all 9 ONNX models across 4 sigma values (36 jobs total).

**Models:**
- conv_big_standard, conv_big_fgsm, conv_big_pgd
- resnet_4b_standard, resnet_4b_fgsm, resnet_4b_pgd  
- cifar_7_1024_standard, cifar_7_1024_fgsm, cifar_7_1024_pgd

**Sigmas:** 0.25, 0.5, 0.75, 1.0

```bash
python certify_hydra.py --config-name sweeps/onnx_all_models -m
```

### `vit_model.yaml`
Sweeps ViT HuggingFace model across 4 sigma values (4 jobs total).

**Model:** aaraki/vit-base-patch16-224-in21k-finetuned-cifar10

**Sigmas:** 0.25, 0.5, 0.75, 1.0

```bash
python certify_hydra.py --config-name sweeps/vit_model -m
```

### `conv_big_models.yaml`
Sweeps Conv Big models (standard, fgsm, pgd) across sigmas (12 jobs).

```bash
python certify_hydra.py --config-name sweeps/conv_big_models -m
```

### `resnet_4b_models.yaml`
Sweeps ResNet 4B models (standard, fgsm, pgd) across sigmas (12 jobs).

```bash
python certify_hydra.py --config-name sweeps/resnet_4b_models -m
```

### `cifar_7_1024_models.yaml`
Sweeps CIFAR 7 1024 models (standard, fgsm, pgd) across sigmas (12 jobs).

```bash
python certify_hydra.py --config-name sweeps/cifar_7_1024_models -m
```

### `sigma_sweep.yaml`
Sweeps default classifier across sigma values (4 jobs).

```bash
python certify_hydra.py --config-name sweeps/sigma_sweep -m
```

### `full_grid_search.yaml`
Example full grid search over multiple parameters.

```bash
python certify_hydra.py --config-name sweeps/full_grid_search -m
```

## Overriding Parameters

You can still override parameters from the command line:

```bash
# Use sweep config but override sample_size
python certify_hydra.py --config-name sweeps/onnx_all_models -m sample_size=200

# Use sweep config but override sigma range
python certify_hydra.py --config-name sweeps/vit_model -m sigma=0.25,0.5
```

## Note on `-m` Flag

The `-m` (multirun) flag is required to enable parameter sweeping. Without it, Hydra will run a single configuration. The sweep configs define the parameters to sweep over, and `-m` tells Hydra to create multiple runs.
