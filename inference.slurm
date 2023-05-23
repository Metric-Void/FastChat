#!/bin/bash
#SBATCH --job-name=vicuna-inference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:20:00
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

python -m fastchat.eval.get_model_answer \
  --model-path /scratch/gpfs/zl1111/llama-7b \
  --model-id llama-7b \
  --question-file /scratch/gpfs/$USER/FastChat/playground/test-data/chats.jsonl