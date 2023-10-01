# Semantic DoReMi

This project provides a pipeline to preprocess, embed, and cluster large sets of text documents utilizing Ray for parallel processing, Hugging Face's `datasets` library for efficient data management, TF-IDF for embeddings, UMAP for dimensionality reduction, and FAISS for clustering.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Workflow Overview](#workflow-overview)
3. [Usage](#usage)

## Prerequisites

- Python 3.x
- Ray
- Hugging Face's `datasets`
- Scikit-learn
- UMAP
- FAISS

Install these dependencies using:

```bash
pip install ray datasets scikit-learn umap-learn faiss-cpu
```

## Workflow Overview

1. **Embedding** (`embedding.py`):
    - Loads a dataset using the `datasets` library from Hugging Face.
    - Preprocesses and embeds the texts using TF-IDF.
    - Stores the embeddings back into the dataset and saves it to disk.

2. **Clustering** (`clustering.py`):
    - Loads the dataset with embeddings.
    - Uses UMAP for dimensionality reduction.
    - Clusters the reduced embeddings using FAISS's KMeans.
    - Stores the cluster labels back to the dataset and saves it.

## Usage

1. **Embedding**:
   
   ```bash
   python embedding.py --max_features 5000 --dataset <huggingface_dataset_name>
   ```

   Replace `<huggingface_dataset_name>` with the name or path of the dataset you want to load from the Hugging Face library. The script will save the embedded dataset to `./embedded_dataset`.

2. **Clustering**:

   ```bash
   python clustering.py --n_components 50 --n_clusters 100
   ```

   This script will load the previously embedded dataset, perform UMAP reduction, and cluster the data using FAISS. The resulting dataset with cluster labels will be saved to `./clustered_dataset`.
