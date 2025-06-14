#!/bin/bash
#SBATCH -J d1_llada_math_diffu-grpo_genlen-256_confthres-0.9_blocklen-32_epoch-1_run-2
#SBATCH -o ../../watch_folder/%x_%j.out
#SBATCH -N 1 
#SBATCH -t 72:00:00  
#SBATCH --get-user-env
#SBATCH --mem=32000  
#SBATCH --partition=cornell,priority
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=50


export TRITON_CACHE_DIR=/tmp


DATASET="math"
RUN_NAME=${DATASET}_diffu-grpo_genlen-256_confthres-0.9_blocklen-32_epoch-1_empireai_run-2
MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"
SAMPLER="llada"
NUM_ITER=12
srun --label accelerate launch \
    --config_file ./accelerate_a100.yaml \
    --num_processes 8 \
    --main_process_port 12346 ../diffu_grpo_train.py \
    --config ./train.yaml \
    --save_steps 100 \
    --num_train_epochs 1 \
    --model_path $MODEL_PATH \
    --num_iterations $NUM_ITER \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --sampler $SAMPLER \
    --output_dir ../outputs/d1_llada_gsm8k_diffu-grpo_genlen-256_confthres-0.9_blocklen-32_epoch-1_run-2