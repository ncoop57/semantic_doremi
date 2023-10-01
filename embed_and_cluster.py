import argparse
import ray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import umap

def preprocess(doc):
    """Preprocess the document by lowercasing it. Can be expanded for more complex preprocessing."""
    return doc.lower()

def tfidf_embedding(docs, max_features):
    """Embed the documents using TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(docs)
    return tfidf_matrix

def umap_reduce(tfidf_matrix, n_components):
    """Reduce dimensionality of the embedded documents using UMAP."""
    reducer = umap.UMAP(n_components=n_components)
    embeddings = reducer.fit_transform(tfidf_matrix)
    return embeddings

def cluster_docs(embedded_docs, n_clusters):
    """Cluster the embedded documents using KMeans."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embedded_docs)
    return kmeans.labels_

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Embed and cluster documents.")
    parser.add_argument('--max_features', type=int, default=5000, help="Maximum number of features for TF-IDF.")
    parser.add_argument('--n_components', type=int, default=50, help="Number of components for UMAP reduction.")
    parser.add_argument('--n_clusters', type=int, default=100, help="Number of clusters for KMeans.")
    
    args = parser.parse_args()

    ray.init()

    # Replace this with your document loading code
    docs = ["sample doc 1", "sample doc 2"]  # Dummy data

    # Preprocess docs
    preprocessed_docs = [preprocess(doc) for doc in docs]

    # Embed using TF-IDF
    tfidf_matrix = tfidf_embedding(preprocessed_docs, args.max_features)

    # Reduce dimensions using UMAP
    embedded_docs = umap_reduce(tfidf_matrix, args.n_components)

    # Cluster
    labels = cluster_docs(embedded_docs, args.n_clusters)

    # Add cluster labels to documents as metadata
    docs_with_metadata = [{"doc": doc, "cluster": label} for doc, label in zip(docs, labels)]

    print(docs_with_metadata)

    ray.shutdown()
