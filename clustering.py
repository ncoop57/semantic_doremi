import argparse
import numpy as np
import umap
import faiss
from datasets import load_from_disk

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

    # Load the embedded dataset
    dataset = load_from_disk('./embedded_dataset')
    tfidf_matrix = np.array(dataset['tfidf_embedding'])

    # Reduce dimensions using UMAP
    embedded_docs = umap_reduce(tfidf_matrix, args.n_components)

    # Cluster using FAISS
    labels = cluster_docs(embedded_docs, args.n_clusters)
    
    dataset = dataset.add_column("cluster", labels)
    dataset.save_to_disk('./clustered_dataset')
