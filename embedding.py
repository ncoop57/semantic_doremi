import argparse
import ray
from datasets import load_dataset, concatenate_datasets
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle

ray.init()

@ray.remote
def preprocess(doc):
    """Preprocess the document by lowercasing it."""
    return doc.lower()

@ray.remote
def tfidf_embedding_partial(docs, max_features, shared_vectorizer):
    """Embed a portion of the documents using TF-IDF."""
    vectorizer = TfidfVectorizer(vocabulary=shared_vectorizer, max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(docs).toarray()
    return tfidf_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Embed documents using TF-IDF.")
    parser.add_argument('--max_features', type=int, default=5000, help="Maximum number of features for TF-IDF.")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the huggingface dataset to load.")
    
    args = parser.parse_args()

    # Load dataset from Hugging Face
    dataset = load_dataset(args.dataset)
    docs = [example['text'] for example in dataset]

    # Preprocess docs in parallel using Ray
    preprocessed_docs = ray.get([preprocess.remote(doc) for doc in docs])

    # Construct a shared vocabulary based on the entire document set
    global_vectorizer = TfidfVectorizer(max_features=args.max_features)
    global_vectorizer.fit(preprocessed_docs)
    shared_vocabulary = global_vectorizer.vocabulary_

    # Split documents into chunks and process in parallel using Ray
    chunk_size = len(docs) // ray.available_resources()['CPU']
    doc_chunks = [docs[i:i + chunk_size] for i in range(0, len(docs), chunk_size)]
    tfidf_matrices = ray.get([tfidf_embedding_partial.remote(chunk, args.max_features, shared_vocabulary) for chunk in doc_chunks])

    # Combine the chunks back into a full matrix
    tfidf_matrix = np.vstack(tfidf_matrices)
    dataset = dataset.add_column("tfidf_embedding", tfidf_matrix.tolist())
    dataset.save_to_disk('./embedded_dataset')

    ray.shutdown()
