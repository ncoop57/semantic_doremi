import fsspec
import os
import argparse
import ray

import pandas as pd
from ray import data as ray_data
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

import numpy as np

fs = fsspec.filesystem("s3")
# ray.init(include_dashboard=True, dashboard_host='0.0.0.0', dashboard_port=8265)
ray.init(address="auto", ignore_reinit_error=True, _temp_dir='./ray_tmp')


model_name = "jinaai/jina-embeddings-v2-small-en"

def preprocess(batch):
    """Preprocess a batch of documents by lowercasing them."""
    return [doc['text'].lower() for doc in batch]

def tfidf_embedding_partial(docs, max_features, shared_vectorizer):
    """Embed a portion of the documents using TF-IDF."""
    vectorizer = TfidfVectorizer(vocabulary=shared_vectorizer, max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(docs).toarray()
    return tfidf_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Embed documents using TF-IDF.")
    parser.add_argument('--max_features', type=int, default=5000, help="Maximum number of features for TF-IDF.")
    # parser.add_argument('--dataset_path', type=str, required=True, help="Path to the Parquet dataset.")

    args = parser.parse_args()

    # Read Parquet dataset using Ray
    file_list = fs.glob("s3://pile-everything-west/redpajama_raw/c4/*.jsonl")
    file_list = ['s3://' + string for string in file_list][:2]
    dataset = ray_data.read_json(file_list)

    # Preprocess docs using Ray's .map() function
    preprocessed_docs = dataset.map(preprocess).flatten()

    # Collect a sample for constructing the shared vocabulary
    sample_docs = preprocessed_docs.take(1000)
    global_vectorizer = TfidfVectorizer(max_features=args.max_features)
    global_vectorizer.fit(sample_docs)
    shared_vocabulary = global_vectorizer.vocabulary_

    # Embed documents using TF-IDF and Ray
    tfidf_matrices = preprocessed_docs.map_partitions(
        lambda batch: tfidf_embedding_partial(batch, args.max_features, shared_vocabulary)
    )
    
    # Save the dataset
    dataset = dataset.add_column("tfidf_embedding", tfidf_matrices)
    dataset.save_to_disk('./embedded_dataset')

    # Ensure that your output path ends with ".parquet" if you're saving in the Parquet format.
    # output_path = "path_to_save/your_embedded_dataset.parquet"
    # tfidf_matrices.write_parquet(output_path)

    # dataset = dataset.add_column("tfidf_embedding", tfidf_matrices)
    # dataset.save_to_disk('./embedded_dataset')

    ray.shutdown()