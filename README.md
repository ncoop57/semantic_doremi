# Semantic DoReMi

This project provides a scalable solution to embed and cluster a large number of text documents using Ray for parallelization, TF-IDF for embedding, UMAP for dimensionality reduction, and KMeans for clustering.

## Requirements

Make sure to install the required packages using:

```
pip install -r requirements.txt
```

## How to Run

1. First, place your documents in an appropriate data structure (e.g., a list) within the main script or load them from an external source.
2. Use the following command to execute the clustering:

```
python embed_and_cluster.py --max_features [NUM_FEATURES] --n_components [NUM_COMPONENTS] --n_clusters [NUM_CLUSTERS]
```

## Parameters

- `--max_features`: Maximum number of features for TF-IDF (default: 5000).
- `--n_components`: Number of components for UMAP reduction (default: 50).
- `--n_clusters`: Number of clusters for KMeans (default: 100).

## Output

The script will output a list of dictionaries, where each dictionary represents a document and its associated cluster label. For instance:

```
[
    {"doc": "sample doc 1", "cluster": 2},
    {"doc": "sample doc 2", "cluster": 5},
    ...
]
```

## Notes

- The preprocessing function currently only lowercases the document. You might want to extend this with more complex preprocessing steps like tokenization, stop-word removal, etc.
- Adjust the hyperparameters for the models as needed. The provided ones serve as reasonable defaults for general purposes.
