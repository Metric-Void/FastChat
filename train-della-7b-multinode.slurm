#!/bin/bash
#SBATCH --job-name=vicuna-multinode
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=zl1111@princeton.edu

# export OMP_NUM_THREADS=4

module purge
module load anaconda3/2023.3
module load cudatoolkit/11.7

conda activate /scratch/gpfs/$USER/.conda/fastchat
cd /scratch/gpfs/$USER/FastChat

export WANDB_MODE=offline

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Head Node: $head_node
echo Node IP: $head_node_ip
echo Global PID: $SLURM_PROCID

export LOGLEVEL=INFO

srun torchrun \
    --nnodes=4 \
    --nproc_per_node=1 \
    --node_rank=$SLURM_PROCID \
    --rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint $head_node_ip:20001 \
    fastchat/train/train_mem.py \
    --model_name_or_path /scratch/gpfs/zl1111/llama-7b \
    --data_path /scratch/gpfs/$USER/FastChat/playground/train-data/dummy.json \
    --bf16 True \
    --output_dir /scratch/gpfs/$USER/llama-7b-finetuned \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
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

