# Semantic DoReMi

This project provides tools to embed and cluster hundreds of millions of text documents using TF-IDF for embedding, UMAP for dimensionality reduction, and FAISS for clustering. The process is split into two main parts: embedding and clustering, each handled by a separate script. The workflow is optimized for execution on a Slurm cluster using the `submitit` library.

### Requirements

- Python 3.8+
- A Slurm cluster (if running on a cluster)
- Virtual Environment (recommended)

### Setup

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/ncoop57/semantic_doremi
    cd semantic_doremi
    ```

2. **Set Up a Virtual Environment:**

    ```bash
    python -m venv venv_name
    source venv_name/bin/activate  # Use 'venv_name\Scripts\activate' on Windows
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Running the Workflow

1. **Embedding Documents:**

    This step transforms text documents into embedded vectors. The embedding is performed using TF-IDF.

    ```bash
    python slurm_launcher.py --job embedding --dataset <huggingface_dataset_name> --max_features <max_features_for_tfidf>
    ```

2. **Clustering Embedded Documents:**

    This step first reduces the embeddings using UMAP and then clusters the embedded vectors using FAISS.

    ```bash
    python slurm_launcher.py --job clustering --n_components <num_umap_components> --n_clusters <num_clusters>
    ```

### Slurm Configuration

The `slurm_launcher.py` script provides arguments for configuring your Slurm job requirements:

- `time`: Max runtime in minutes for Slurm job. Default is `120`.
- `cpus_per_task`: Number of CPUs per task. Default is `2`.
- `tasks_per_node`: Number of tasks per node. Default is `1`.
- `gpus_per_node`: Number of GPUs per node. Default is `2`.
- `nodes`: Number of nodes. Default is `1`.
- `mem_gb`: Memory in GB. Default is `32`.
- `slurm_partition`: Slurm partition name. Default is `main`.

These arguments can be provided to the `slurm_launcher.py` script when launching your jobs.

### Note on Virtual Environments

When submitting jobs to Slurm using the provided job template (`job_template.sh`), the virtual environment you set up in the **Setup** section is automatically activated on the Slurm nodes. Ensure the path to the virtual environment in the `job_template.sh` script is correctly set.
