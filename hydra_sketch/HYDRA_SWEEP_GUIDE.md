# Hydra Parameter Sweep Guide for certify.py

This guide explains how to use Hydra for running parameter sweeps and scheduling jobs on SLURM.

## Installation

```bash
pip install hydra-core hydra-submitit-launcher
```

## Basic Usage

### Single Run (with defaults from config)
```bash
python certify_hydra.py
```

### Single Run (override parameters)
```bash
python certify_hydra.py sigma=0.5 sample_size=200 dataset_name="MNIST"
```

### Parameter Sweep (Local - runs sequentially)
```bash
# Sweep over sigma values
python certify_hydra.py -m sigma=0.25,0.5,0.75

# Sweep over multiple parameters (grid search)
python certify_hydra.py -m sigma=0.25,0.5 sample_size=100,200

# Sweep with different classifiers
python certify_hydra.py -m \
  sigma=0.25,0.5 \
  sample_size=100,200 \
  classifier_name="model1,model2,model3"
```

### Parameter Sweep (SLURM - parallel jobs)
```bash
# This will submit each configuration as a separate SLURM job
python certify_hydra.py -m sigma=0.25,0.5,0.75 sample_size=100,200
```

## Sweep Examples

### Example 1: Sigma Sweep
```bash
# Test different noise levels
python certify_hydra.py -m sigma=0.1,0.25,0.5,0.75,1.0
```

### Example 2: Sample Size Sweep
```bash
# Test different sample sizes
python certify_hydra.py -m sample_size=50,100,200,500
```

### Example 3: Full Grid Search
```bash
# Grid search over sigma and sample_size
python certify_hydra.py -m \
  sigma=0.25,0.5,0.75 \
  sample_size=100,200 \
  N0=100,200
```

### Example 4: Classifier Comparison
```bash
# Compare different classifiers
python certify_hydra.py -m \
  classifier_name="aaraki/vit-base-patch16-224-in21k-finetuned-cifar10,model2,model3" \
  sigma=0.25,0.5
```

## SLURM Configuration

The SLURM launcher is configured in `conf/hydra/launcher/submitit_slurm.yaml`:

```yaml
partition: gpu_a100
gpus_per_node: 1
cpus_per_task: 18
mem_gb: 120
timeout_min: 30
```

### Customizing SLURM Resources

You can override SLURM settings per sweep:

```bash
# Use different partition
python certify_hydra.py -m sigma=0.25,0.5 \
  hydra/launcher=submitit_slurm \
  hydra.launcher.partition=cpu_partition \
  hydra.launcher.gpus_per_node=0
```

## Output Organization

Hydra organizes outputs automatically:

```
multirun/
└── 2024-01-15/
    └── 10-30-45/
        ├── 0/  # First job
        │   └── outputs/
        ├── 1/  # Second job
        │   └── outputs/
        └── multirun.yaml  # Summary of all runs
```

Each job gets its own directory with:
- Output files
- Logs
- Config used
- Results

## Monitoring Sweeps

### Check Job Status
```bash
# View submitted jobs
squeue -u $USER

# View job details
scontrol show job <job_id>
```

### View Sweep Summary
```bash
# After sweep completes, check the multirun.yaml
cat multirun/2024-01-15/10-30-45/multirun.yaml
```

## Advanced: Custom Sweep Configs

Create custom sweep configurations in `conf/sweeps/`:

### conf/sweeps/my_sweep.yaml
```yaml
defaults:
  - /certify@_here_

# Define default sweep parameters
# Override via command line: python certify_hydra.py --config-name my_sweep -m sigma=0.5
```

Usage:
```bash
python certify_hydra.py --config-name my_sweep -m sigma=0.25,0.5
```

## Integration with Existing SLURM Scripts

If you want to keep using your existing `run_certification.slurm`, you can:

1. **Option A**: Use Hydra inside the SLURM script
```bash
# In run_certification.slurm
python certify_hydra.py -m sigma=$SIGMA sample_size=$SAMPLE_SIZE
```

2. **Option B**: Use Hydra's SLURM launcher (recommended)
```bash
# Submit directly from command line - no SLURM script needed!
python certify_hydra.py -m sigma=0.25,0.5,0.75
```

## Troubleshooting

### Jobs not submitting
- Check that `submitit` is installed: `pip install submitit`
- Verify SLURM is accessible: `sinfo`
- Check launcher config: `conf/hydra/launcher/submitit_slurm.yaml`

### Config not found
- Ensure config files are in `conf/` directory
- Check config name matches: `--config-name certify`

### Parameter not overriding
- Use `=` syntax: `sigma=0.5` not `--sigma 0.5`
- For nested configs: `hydra.launcher.partition=gpu_a100`

## Best Practices

1. **Start Small**: Test with 2-3 configurations first
2. **Use Descriptive Names**: Set `hydra.job.name` for easy identification
3. **Monitor Resources**: Check SLURM queue limits before large sweeps
4. **Save Configs**: Hydra automatically saves used configs in output directories
5. **Use Comet ML**: Already integrated for tracking all runs

## Example: Complete Workflow

```bash
# 1. Test single run
python certify_hydra.py sigma=0.25

# 2. Small sweep (local)
python certify_hydra.py -m sigma=0.25,0.5

# 3. Large sweep (SLURM)
python certify_hydra.py -m \
  sigma=0.1,0.25,0.5,0.75,1.0 \
  sample_size=50,100,200 \
  classifier_name="model1,model2"

# 4. Monitor jobs
watch -n 5 'squeue -u $USER'

# 5. Check results
ls multirun/*/multirun.yaml
```

