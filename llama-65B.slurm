#!/bin/bash
#SBATCH --job-name=llama-65B-inference
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --mem=180G
#SBATCH --time=00:10:00
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
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=$SLURM_PROCID \
    --rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint $head_node_ip:20001 \
    llama/example.py \
    --ckpt_dir /scratch/gpfs/zl1111/llama-models/65B/ \
    --tokenizer_path /scratch/gpfs/zl1111/llama-models/tokenizer.model
