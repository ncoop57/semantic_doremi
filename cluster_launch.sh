#!/bin/bash
#SBATCH --partition=cpu24
#SBATCH --job-name=semantic_doremi
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --exclusive
#SBATCH --mem=64g
#SBATCH --output=raylogs/%x_%j.out
#SBATCH --account stability

$PWD/venv/bin/python clustering.py --n_components <num_umap_components> --n_clusters <num_clusters>