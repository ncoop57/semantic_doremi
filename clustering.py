import argparse
import numpy as np
import umap
import faiss
import pickle

def umap_reduce(tfidf_matrix, n_components):
    """Reduce dimensionality of the embedded documents using UMAP."""
    reducer = umap.UMAP(n_components=n_components)
    embeddings = reducer.fit_transform(tfidf_matrix)
    return embeddings

def cluster_docs(embedded_docs, n_clusters):
    """Cluster the embedded documents using FAISS's KMeans."""
    embedded_docs = embedded_docs.astype('float32')
    kmeans = faiss.Kmeans(embedded_docs.shape[1], n_clusters, niter=20, verbose=True)
    kmeans.train(embedded_docs)
    _, labels = kmeans.index.search(embedded_docs, 1)
    return labels.reshape(-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reduce dimensions using UMAP and cluster using FAISS.")
    parser.add_argument('--n_components', type=int, default=50, help="Number of components for UMAP reduction.")
    parser.add_argument('--n_clusters', type=int, default=100, help="Number of clusters for KMeans in FAISS.")
    
    args = parser.parse_args()

    with open('embedded_docs.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)

    # Reduce dimensions using UMAP
    embedded_docs = umap_reduce(tfidf_matrix, args.n_components)

    # Cluster using FAISS
    labels = cluster_docs(embedded_docs, args.n_clusters)

    # Add cluster labels to documents as metadata
    docs_with_metadata = [{"doc": doc, "cluster": label} for doc, label in zip(docs, labels)]

    print(docs_with_metadata)
