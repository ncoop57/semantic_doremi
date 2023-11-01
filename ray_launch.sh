#!/bin/bash
#SBATCH --partition=g40
#SBATCH --job-name=semantic_doremi
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=6
#SBATCH --exclusive
#SBATCH --mem=64g
#SBATCH --output=raylogs/%x_%j.out
#SBATCH --account stablegpt
#SBATCH --exclude=ip-26-0-152-47
#SBATCH --time=05:00:00

srun --account stablegpt sh $PWD/ray_worker.sh
