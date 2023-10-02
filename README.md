# Semantic DoReMi

This project provides tools to embed and cluster hundreds of millions of text documents using TF-IDF for embedding, UMAP for dimensionality reduction, and FAISS for clustering. The process is split into two main parts: embedding and clustering, each handled by a separate script. The workflow is optimized for execution on a Slurm cluster.

## Setup

### 1. Virtual Environment

Before starting, ensure you have set up a Python virtual environment:

```bash
python -m venv /path/to/your/venv
source /path/to/your/venv/bin/activate
pip install -r requirements.txt
```

Replace `/path/to/your/venv` with your desired directory for the virtual environment.

### 2. Ray Cluster Initialization

Start a Ray cluster on your Slurm environment by first modifying the file `ray_launch.sh` to modify the number of nodes, partition name, memory of each node, job name, and account. Once you are happy with it, you can launch using standard sbatch:

```bash
sbatch ray_launch.sh
```

### 3. Logging into the Ray Head Node

Once your Ray cluster is up, identify the head node's IP (which is typically printed on the console). You can then SSH into this head node:

```bash
ssh ray_head_node_ip
```

Ensure that you have `ray` initialized and running on this head node.

### 4. Running the Embedding Script

With your Ray cluster active and while logged into the head node, run the embedding script:

```bash
source /path/to/your/venv/bin/activate
python embedding.py --dataset <huggingface_dataset_name> --max_features <max_features_for_tfidf>
```

Replace <huggingface_dataset_name> with the name or path of the dataset you want to load from the Hugging Face library. The script will save the embedded dataset to ./embedded_dataset.

### 5. Running the Clustering Script

You no longer need the ray cluster since we won't be using any distributed UMAP and FAISS. Therefore, launch a separate CPU job that will just be one node by modifying the file `cluster_launch.sh` script to your liking. Then run it similar to the ray launching script:

```bash
sbatch cluser_launch.sh
```

This script will load the previously embedded dataset, perform UMAP reduction, and cluster the data using FAISS. The resulting dataset with cluster labels will be saved to ./clustered_dataset.

## Additional Notes

- Ensure that you have appropriate permissions and necessary resources (nodes, memory, etc.) allocated for your Slurm jobs.
- Ensure your dataset is in a format compatible with HuggingFace's `datasets` library.
- Monitor your Ray cluster's health and logs for any anomalies during execution.
