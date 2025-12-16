#!/bin/bash
# Example script for running Hydra sweeps
# This demonstrates different sweep patterns

echo "=== Hydra Parameter Sweep Examples ==="
echo ""

# Example 1: Simple sigma sweep
echo "Example 1: Sigma sweep (0.25, 0.5, 0.75)"
echo "Command: python certify_hydra.py -m sigma=0.25,0.5,0.75"
echo ""

# Example 2: Grid search
echo "Example 2: Grid search (sigma x sample_size)"
echo "Command: python certify_hydra.py -m sigma=0.25,0.5 sample_size=100,200"
echo ""

# Example 3: Full parameter sweep
echo "Example 3: Full parameter sweep"
echo "Command:"
cat << 'EOF'
python certify_hydra.py -m \
  sigma=0.25,0.5,0.75 \
  sample_size=50,100,200 \
  N0=100,200 \
  classifier_name="aaraki/vit-base-patch16-224-in21k-finetuned-cifar10,model2"
EOF

echo ""
echo "=== To run a sweep, uncomment one of the examples above ==="

# Uncomment to run:
# python certify_hydra.py -m sigma=0.25,0.5,0.75

