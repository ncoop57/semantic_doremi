#!/bin/bash
#SBATCH --partition=g40
#SBATCH --job-name=applied_ablation
#SBATCH --nodes 12
#SBATCH --ntasks-per-node 1
#SBATCH --exclusive
#SBATCH --mem=999g
#SBATCH --output=raylogs/%x_%j.out
#SBATCH --account stablegpt

srun --account stablegpt sh $(PWD)/ray_worker.sh
