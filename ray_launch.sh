#!/bin/bash
#SBATCH --partition=cpu24
#SBATCH --job-name=semantic_doremi
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1
#SBATCH --exclusive
#SBATCH --mem=64g
#SBATCH --output=raylogs/%x_%j.out
#SBATCH --account stability

srun --account stability sh $PWD/ray_worker.sh
