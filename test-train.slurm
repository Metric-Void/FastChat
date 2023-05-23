#!/bin/bash
#SBATCH --job-name=vicuna-bp-trainer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=zl1111@princeton.edu

# export OMP_NUM_THREADS=4

module purge
module load anaconda3/2023.3
module load cudatoolkit/11.7

conda activate vicuna
cd /scratch/network/zl1111/FastChat

export WANDB_MODE=offline

torchrun --nproc_per_node=1 --master_port=20001 fastchat/train/train_mem.py \
    --model_name_or_path /scratch/network/zl1111/pyllama_data/7B \
    --data_path /scratch/network/zl1111/SHORT_MOBILE-19.json \
    --bf16 True \
    --output_dir /scratch/network/zl1111/llama-bp-7b \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True

