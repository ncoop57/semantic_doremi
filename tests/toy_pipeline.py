import umap
import faiss
import sklearn 

import numpy as np
import matplotlib.pyplot as plt


#Step 1 - Load a dataset from SKLearn, define parameters
from sklearn.datasets import fetch_20newsgroups
dataset = fetch_20newsgroups(subset='train')

#Parameters
dataset_name = 'fetch_20newsgroups'
max_feature_dim = 5000 #Pre-embedding, feature dim
n_clusters = 250
embedding_dim = 2 #Post UMAP Dimensionality


from pprint import pprint

pprint(list(dataset.target_names))

#Step 2 - Vectorize the dataset using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

document_vectors = vectorizer.fit_transform(dataset.data)
print(document_vectors.shape)

#Assess sparsity  -- number of non negative- 
pprint(document_vectors.nnz / float(document_vectors.shape[0])) #TODO: Not too sure what .nnz is, 

#Step 3 - Perform dimensionality using UMAP
reducer = umap.UMAP(n_components=embedding_dim)
embedded_docs = reducer.fit_transform(document_vectors)

print('Embedding Shape:', embedded_docs.shape)
print(type(embedded_docs))

#Step 4 - Perform Clustering
embedded_docs = embedded_docs.astype('float32')
kmeans = faiss.Kmeans(embedded_docs.shape[1], n_clusters, niter=20, verbose=True)
kmeans.train(embedded_docs)
_, labels = kmeans.index.search(embedded_docs, 1)

#Step 5 - Plot Results
labels = labels.reshape(-1)

plt.title(f"{n_clusters} clusters, {dataset_name} dataset")
cmap = plt.get_cmap('rainbow')

# Generating a colormap that maps each label to a specific color
colors = cmap(np.linspace(0, 1, n_clusters)) 

for i in range(n_clusters):
    indices = np.where(labels == i)[0]
    plt.scatter(embedded_docs[indices, 0], embedded_docs[indices, 1], color=colors[i], label=f'Label {i}')

plt.show()
