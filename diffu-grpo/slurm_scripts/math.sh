#!/bin/bash
#SBATCH -J d1_llada_math_diffu-grpo-fastdllm_genlen-256_confthres-0.6_blocklen-32_epoch-1_train-1
#SBATCH -o ../../watch_folder/%x_%j.out
#SBATCH -N 1 
#SBATCH -t 960:00:00  
#SBATCH --get-user-env 
#SBATCH --mem=32000  
#SBATCH --partition=gpu
#SBATCH --constraint="[a6000|6000ada|a100|h100]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=50
#SBATCH --exclude=""


export TRITON_CACHE_DIR=/tmp


DATASET="math"
RUN_NAME=${DATASET}_diffu-grpo-fastdllm_genlen-256_confthres-0.6_blocklen-32_epoch-1_train-1
MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"
#CHECKPOINT_PATH="/share/kuleshov/gw354/ICLR-2026/d1/diffu-grpo/outputs/d1_llada_math_diffu-grpo-fastdllm_genlen-256_confthres-0.8_blocklen-32_epoch-1_train-1/checkpoint-2856"
SAMPLER="fast_dllm"
NUM_ITER=12
srun --label accelerate launch \
    --config_file ./accelerate_a100.yaml \
    --num_processes 4 \
    --main_process_port 12346 ../diffu_grpo_train.py \
    --config ./train.yaml \
    --save_steps 204 \
    --num_train_epochs 1 \
    --model_path $MODEL_PATH \
    --num_iterations $NUM_ITER \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --sampler $SAMPLER \
    --conf_thres 0.6 \
    --output_dir ../outputs/d1_llada_math_diffu-grpo-fastdllm_genlen-256_confthres-0.6_blocklen-32_epoch-1_train-1
    #--checkpoint_path $CHECKPOINT_PATH \