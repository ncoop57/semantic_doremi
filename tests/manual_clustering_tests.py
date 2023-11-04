from clustering import cluster_docs
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colormaps

import faiss 
#Generate some data within R of a center mean


DIM = 2 #Each vector is 2D 
n_clusters = 150 #Let's say there are three clusters
VECTORS_PER_CENTROID = 100 

#Generation 
centroids = np.random.randint(-100, 100, size = n_clusters * DIM).reshape(n_clusters, DIM)
D = np.stack([np.random.normal( size = (VECTORS_PER_CENTROID, DIM)) + centroids[i, :] for i in range(centroids.shape[0])]).reshape((-1, 2))

#Plot the centers 
D_full = np.vstack([D, centroids]) #Add in the actual centroids

#The K-means method under test
def cluster_docs(embedded_docs, n_clusters):
    """Cluster the embedded documents using FAISS's KMeans."""
    embedded_docs = embedded_docs.astype('float32')

    print(embedded_docs, 'searching for shape', embedded_docs.shape[1])

    kmeans = faiss.Kmeans(embedded_docs.shape[1], n_clusters, niter=20, verbose=True)
    kmeans.train(embedded_docs)

    print('Searching along ')
    _, labels = kmeans.index.search(embedded_docs, 1)
    return labels.reshape(-1)

labels = cluster_docs(D_full, n_clusters)

#Creating a rainbow colormap
cmap = plt.get_cmap('rainbow')

# Generating a colormap that maps each label to a specific color
colors = cmap(np.linspace(0, 1, n_clusters)) 

print(labels.shape[0])
print(labels)
for i in range(n_clusters):
    indices = np.where(labels == i)[0]
    plt.scatter(D_full[indices, 0], D_full[indices, 1], color=colors[i], label=f'Label {i}')

plt.show()