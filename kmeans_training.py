import argparse
import fsspec
import pandas as pd
import numpy as np
import umap
import faiss

from tqdm.auto import tqdm


def train_kmeans(embedded_docs, n_clusters, seed=115):
    """Cluster the embedded documents using FAISS's KMeans."""
    embedded_docs = embedded_docs.astype("float32")
    kmeans = faiss.Kmeans(
        embedded_docs.shape[1], n_clusters, verbose=True, gpu=True, seed=seed
    )
    kmeans.train(embedded_docs)
    return kmeans.centroids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reduce dimensions using UMAP and cluster using FAISS."
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=10,
        help="Number of clusters for KMeans in FAISS.",
    )
    parser.add_argument(
        "--train_percent",
        type=float,
        default=0.1,
        help="Percent of data to use for training kmeans.",
    )
    parser.add_argument(
        "--s3_path",
        type=str,
        default="s3://pile-everything-west/redpajama_processed/c4/",
        help="Path to the S3 directory containing the parquet files.",
    )
    parser.add_argument(
        "--extension",
        type=str,
        default=".parquet",
        help="Path to the S3 directory containing the parquet files.",
    )

    args = parser.parse_args()
    seed = 115
    np.random.seed(seed)

    fs = fsspec.filesystem(
        "s3",
        config_kwargs={"retries": {"max_attempts": 10}, "max_pool_connections": 512},
    )
    all_files = fs.glob(f"{args.s3_path}/*{args.extension}")
    # select random subset of files
    n = int(len(all_files) * args.train_percent)
    embedding_files = np.random.choice(all_files, n, replace=False)
    embds = []
    for file in tqdm(embedding_files, total=len(embedding_files), desc="Loading files"):
        with fs.open(file, "rb") as f:
            df = pd.read_parquet(
                f
            )  # if args.extension == ".parquet" else pd.read_json(f, lines=True)
        embds.extend(df["embeddings"].tolist())

    # h/t to: https://github.com/facebookresearch/faiss/issues/2087
    embds = np.array(embds)
    centroids = train_kmeans(embds, args.n_clusters, seed=seed)
    with fs.open(args.s3_path + "centroids.npy", "wb") as f:
        np.save(f, centroids)
    with open("kmeans_embeddings.npy", "wb") as f:
        np.save(f, embds)
