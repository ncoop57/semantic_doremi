import argparse
import fsspec
from tqdm import tqdm
import ray
import os
import pandas as pd
import numpy as np
import faiss

@ray.remote(num_gpus=1,num_cpus=6,max_retries=10)
def worker(file_list, seed=115):
    out_fs = fsspec.filesystem(
        "s3",
        config_kwargs={
            "retries": {"max_attempts": 10},
            "max_pool_connections": 512
        }
    )
    s3_path = "s3://pile-everything-west/redpajama_processed/c4/"
    print("Loading centroids and data")
    with out_fs.open(s3_path + "centroids.npy", "rb") as f:
        centroids = np.load(f)
    with open("kmeans_embeddings.npy", "rb") as f:
        data = np.load(f)
    input_dim = data.shape[1]
    num_clusters = centroids.shape[0]
    # h/t to: https://github.com/facebookresearch/faiss/issues/2087
    kmeans = faiss.Kmeans(input_dim, num_clusters, verbose=True, gpu=False, niter=0, nredo=0, seed=seed)
    kmeans.train(data, init_centroids=centroids)
    # this ensures that kmeans.index is created
    assert np.sum(kmeans.centroids - centroids) == 0, "centroids are not the same" # sanity check

    for path in tqdm(file_list, total=len(file_list), desc="Labeling clusters"):
        with out_fs.open(path, "rb") as f:
            df = pd.read_parquet(f)
        embds = np.array(df["embeddings"].tolist()).astype("float32")
        try:
            # get the labels
            _, labels = kmeans.assign(embds)
            df["cluster"] = labels.reshape(-1)
            # assert they are not -1
            assert np.sum(df["cluster"].values == -1) == 0, "There are -1 labels"
        except Exception as e:
                print(e)
                slurm_job_id = os.environ["SLURM_JOB_ID"]
                # dump the batch to a txt file
                with open(f"error_{slurm_job_id}.txt", "w") as f:
                    f.write(str(df['text'].tolist()))
                raise ValueError(f"Error with batch at {path}")
        with out_fs.open(
            path.replace(
                "redpajama_processed",
                "redpajama_clustered",
            ).replace(".jsonl", ".parquet"),
            "wb",
        ) as f:
            df.to_parquet(f)
        print("Done with file: " + path.replace(
                "redpajama_processed",
                "redpajama_clustered",
            ).replace(".jsonl", ".parquet"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reduce dimensions using UMAP and cluster using FAISS."
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=50,
        help="Number of components for UMAP reduction.",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=10,
        help="Number of clusters for KMeans in FAISS.",
    )

    args = parser.parse_args()

    ray.init(address="auto", ignore_reinit_error=True)

    fs = fsspec.filesystem("s3")
    file_list = fs.glob("s3://pile-everything-west/redpajama_processed/c4/*.jsonl")
    file_list = ['s3://' + string for string in file_list]

    file_list.sort()
    gpus = 8
    nodes = 4
    partitions = gpus * nodes # 8 is gpus and 1 is nodes

    file_list = np.array_split(file_list, partitions)
    workers = [worker.remote(x) for x in file_list]

    ray.get(workers)