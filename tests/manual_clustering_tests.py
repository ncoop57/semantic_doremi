#Run this test from the base repo directory
# python tests/manual_clustering_tests.py

import faiss
from clustering import cluster_docs
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colormaps

DIM = 2 #Each vector is 2D 
N_CLUSTERS = 150 #Let's say there are three clusters
VECTORS_PER_CENTROID = 100 
MAX_NOISE = 5

#Generation 
np.random.seed(1234)
centroids = np.random.randint(-100, 100, size = N_CLUSTERS * DIM).reshape(N_CLUSTERS, DIM)
D = np.stack([ np.random.randint(1, MAX_NOISE) * np.random.normal( size = (VECTORS_PER_CENTROID, DIM)) + centroids[i, :] for i in range(centroids.shape[0])]).reshape((-1, 2))

#Plot the centers 
D_full = np.vstack([D, centroids]) #Add in the actual centroids

#The K-means method under test
def cluster_docs(embedded_docs, n_clusters):
    """Cluster the embedded documents using FAISS's KMeans."""
    embedded_docs = embedded_docs.astype('float32')

    kmeans = faiss.Kmeans(embedded_docs.shape[1], n_clusters, niter=20, verbose=True)
    kmeans.train(embedded_docs)

    _, labels = kmeans.index.search(embedded_docs, 1)
    return labels.reshape(-1)

labels = cluster_docs(D_full, N_CLUSTERS)

#Creating a rainbow colormap
plt.title(f"{N_CLUSTERS} clusters + {VECTORS_PER_CENTROID} vectors, max noise {MAX_NOISE}")
cmap = plt.get_cmap('rainbow')

# Generating a colormap that maps each label to a specific color
colors = cmap(np.linspace(0, 1, N_CLUSTERS)) 

for i in range(N_CLUSTERS):
    indices = np.where(labels == i)[0]
    plt.scatter(D_full[indices, 0], D_full[indices, 1], color=colors[i], label=f'Label {i}')

plt.show()