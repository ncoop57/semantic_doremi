# Semantic DoReMi

This project provides tools to embed and cluster hundreds of millions of text documents using TF-IDF for embedding, UMAP for dimensionality reduction, and FAISS for clustering. The process is split into two main parts: embedding and clustering, each handled by a separate script. The workflow is optimized for execution on a Slurm cluster using the `submitit` library.

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

Start a Ray cluster on your Slurm environment using the following:

```bash
python ray_slurm_launcher.py --venv_path /path/to/your/venv --nodes 10 --partition your_partition --mem_gb 500 --job_name your_job_name --account your_account_name
```

Adjust the arguments based on your requirements.

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

## Embedding & Clustering

1. **Embedding**: Uses Ray for distributed processing to embed text documents using TF-IDF and UMAP reduction.

2. **Clustering**: Once you have the embeddings, you can proceed to cluster them using FAISS:

```bash
python clustering.py --n_components <num_umap_components> --n_clusters <num_clusters>
```

This script will load the previously embedded dataset, perform UMAP reduction, and cluster the data using FAISS. The resulting dataset with cluster labels will be saved to ./clustered_dataset.

## Additional Notes

- Ensure that you have appropriate permissions and necessary resources (nodes, memory, etc.) allocated for your Slurm jobs.
- Ensure your dataset is in a format compatible with HuggingFace's `datasets` library.
- Monitor your Ray cluster's health and logs for any anomalies during execution.
