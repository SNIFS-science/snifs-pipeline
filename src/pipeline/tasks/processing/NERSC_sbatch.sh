#!/bin/bash
#SBATCH -A <YOUR_PROJECT_ACCOUNT>     #CHANGE HERE!!!!
#SBATCH -C "gpu&hbm80g"                # Request Perlmutter 80GB A100 GPU nodes
#SBATCH -q debug                        # debug because testing for now 30 min limit
#SBATCH -t 00:15:00                     # the cpu portion takes abt 5.7 mins but everything needs to sync between iters
#SBATCH -N 1                            # 1 full GPU node
#SBATCH --gpus-per-node=4               
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=snifs_test_group9
#SBATCH --output=logs/snifs_test_%j.out
#SBATCH --error=logs/snifs_test_%j.err

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================
module load cudatoolkit #CHANGE HERE!
module load python


export CUDA_PATH=$CUDA_HOME

# Memory and Worker configuration:
export MODEL_CPU_WORKERS= 32              
export MODEL_CPU_MEM="3.5GiB"            # 56 workers * 4.2 = 235 GB (Safely under 256GB limit)
export MODEL_GPU_MEM="10GiB"             # Per-task VRAM allocation (4 workers per 80GB GPU)

# To saturate 56 CPU cores, we need ~10-15 GPU matrices waiting in the queue.
# 12 inflight matrices * 5.5 perturbations = ~66 CPU tasks ready to go.
export MODEL_INFLIGHT=12                 

# Threading control irrelevant not using pypardiso right now
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ==============================================================================
# NVIDIA MPS SETUP
# ==============================================================================
# Highly recommended to leave this ON. It forces the 4 Dask workers on each 
# GPU to share memory contexts and execute concurrent kernels safely.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

#CHANGE HERE
GROUP_NUM=9 #0-14
FITS_PATH="" #ADDPATH

OUTPUT_DIR="$SCRATCH/snifs-pipeline/output/make_matrix_GPU_accelerated_test"
mkdir -p "$OUTPUT_DIR"
# ==============================================================================
# EXECUTION
# ==============================================================================
echo "Forward Modeling group ${GROUP_NUM} on Perlmutter GPU Node"

srun --gpu-bind=none uv python laptop_run.py \
    --group "$GROUP_NUM" \
    --output-dir "$OUTPUT_DIR" \
    --fits-path "$FITS_PATH"

echo "Job completed successfully."

# ==============================================================================
# CLEANUP
# ==============================================================================
echo quit | nvidia-cuda-mps-control