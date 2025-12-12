#!/bin/bash
#SBATCH --job-name=certification
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=7:00:00
#SBATCH --output=/projects/prjs1681/runs/slurm_outputs/certification_%j.out
#SBATCH --error=/projects/prjs1681/runs/slurm_outputs/certification_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

CLASSIFIER=$(echo "${2:-cifar10}" | tr -c '[:alnum:]_-' '_')
DATASET=$(echo "${1:-default}" | tr -c '[:alnum:]_-' '_')
JOB_NAME="${CLASSIFIER}_certification_${DATASET}"
scontrol update job=$SLURM_JOB_ID name=$JOB_NAME 2>/dev/null || echo "Could not update job name to $JOB_NAME"

echo "Loaded modules:"
module list 2>&1

cd /home/jvrijn/code/rs/diffusion_denoised_smoothing

echo "Activating conda environment verona_new..."
source /gpfs/home2/jvrijn/miniforge3/etc/profile.d/conda.sh
conda activate verona_new
# 
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Comet ML version: $(python -c 'import comet_ml; print(comet_ml.__version__)')"
echo "ONNX Runtime CUDA providers: $(python -c 'import onnxruntime as ort; print(ort.get_available_providers())')"

# Run the certification experiment

DATASET=${1}
CLASSIFIER_NAME=${2}
CLASSIFIER_TYPE=${3}
SIGMA=${4:-0.25}
SAMPLE_SIZE=${5:-100}
N0=${6:-100}
N=${7:-100000}
BATCH_SIZE=${8:-200}
ALPHA=${9:-0.001}
RANDOM_SEED=${10:-42}
SAMPLE_CORRECT_PREDICTIONS=${11:-True}
STRATIFIED=${12:-True}


# Validate dataset
if [[ "$DATASET" != "cifar10" && "$DATASET" != "mnist" && "$DATASET" != "imagenet" ]]; then
    echo "Error: Unsupported dataset '$DATASET'"
    echo "Supported datasets: cifar10, mnist, imagenet"
    exit 1
fi

# Set dataset-specific configurations
case "$DATASET" in
    "cifar10")
        SCRIPT_DIR="cifar10"
        ;;
    "mnist")
        SCRIPT_DIR="mnist"
        ;;
    "imagenet")
        SCRIPT_DIR="imagenet"
        ;;
esac

python ${SCRIPT_DIR}/certify.py \
    --sigma $SIGMA \
    --sample_size $SAMPLE_SIZE \
    --N0 $N0 \
    --N $N \
    --batch_size $BATCH_SIZE \
    --alpha $ALPHA \
    --sample_correct_predictions $SAMPLE_CORRECT_PREDICTIONS \
    --random_seed $RANDOM_SEED \
    --stratified $STRATIFIED \
    --classifier_type $CLASSIFIER_TYPE \
    --classifier_name "$CLASSIFIER_NAME"

# Print completion information
echo "Job completed at: $(date)"
echo "Exit code: $?"

# Clean up resources
echo "Cleaning up resources..."
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Deactivate conda environment
conda deactivate

# Print final resource usage
echo "Final resource usage:"
echo "Memory usage: $(free -h | grep '^Mem:' | awk '{print $3 "/" $2}')"
echo "GPU memory: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo 'N/A')"

echo "Job finished successfully."

# Example runs:
# ONNX model (only correct predictions):
#   sbatch run_certification.slurm cifar10 conv_big_standard onnx 0.25 10 100 100000 200 0.001 certification_run 42 True
# ONNX model (all predictions):
#   sbatch run_certification.slurm cifar10 conv_big_standard onnx 0.25 10 100 100000 200 0.001 certification_run 42 False
# HuggingFace model (vit-base with aaraki organization):
#   sbatch run_certification.slurm cifar10 aaraki/vit-base-patch16-224-in21k-finetuned-cifar10 huggingface 0.25 10 100 100000 200 0.001 certification_run 42 True
# HuggingFace model (custom from any organization):
#   sbatch run_certification.slurm cifar10 organization/model-name huggingface 0.25 10 100 100000 200 0.001 certification_run 42 True