#!/bin/bash
#SBATCH --partition=g40
#SBATCH --job-name=kmeans
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=6
#SBATCH --exclusive
#SBATCH --mem=999g
#SBATCH --output=raylogs/%x_%j.out
#SBATCH --account stablegpt
#SBATCH --exclude=ip-26-0-152-47
#SBATCH --time=24:00:00

ulimit -n 100000
# pip install torch --index-url https://download.pytorch.org/whl/cu118
# eval "$(/admin/home-nathan/miniconda3/bin/conda shell.bash hook)"
# conda activate semantic_doremi

# export CONDA_PREFIX=/admin/home-nathan/miniconda3/envs/semantic_doremi

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
source /admin/home-nathan/semantic_doremi/venv/bin/activate
python kmeans_training.py \
    --n_components 50 \
    --n_clusters 10 \
    --train_percent 0.1 \
    --s3_path s3://pile-everything-west/redpajama_processed/c4/ \
    --extension .jsonl

deactivate