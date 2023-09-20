#!/bin/bash
#SBATCH --job-name=baseline_pretrained_2
#SBATCH --account=def-ssanner
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:4
#SBATCH --mem-per-gpu=24G
#SBATCH --time=01-00:00:00
#SBATCH --cpus-per-task=16
nvidia-smi
export OMP_NUM_THREADS=16

source /home/gfloto/env_store/diff_env/bin/activate
cd /home/gfloto/projects/def-ssanner/gfloto/projects/diffuseq/scripts

module load gcc/9.3.0 arrow cuda/11

torchrun --nproc_per_node=4 run_train.py \
--data_dir /home/gfloto/scratch/datasets/detox \
--folder_name /home/gfloto/scratch/diffusion_models \
--resume_checkpoint /home/gfloto/scratch/diffusion_models/qqp/model000999.pt \
--diff_steps 2000 \
--lr 0.0001 \
--learning_steps 100000 \
--save_interval 1000 \
--seed 102 \
--noise_schedule sqrt \
--hidden_dim 128 \
--bsz 2048 \
--microbatch 256 \
--dataset detox \
--vocab bert \
--seq_len 64 \
--schedule_sampler lossaware \
--notes detox
