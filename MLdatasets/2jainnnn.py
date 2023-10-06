import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import sklearn

from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn import metrics

# Load the .mat file
mat = scipy.io.loadmat('jain.mat')

# Print the keys in the loaded .mat file
print(mat.keys())

# Extract the data from the loaded .mat file
x = mat['jain2']
df = pd.DataFrame(x)
df.to_csv("jain2.csv", index=False)  # Save DataFrame to CSV

# Standardize and normalize the data
X = StandardScaler().fit_transform(x)
X = normalize(X)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.3, min_samples=4)
labels = dbscan.fit_predict(X_pca)

# Identify the core points
sample_cores = np.zeros_like(labels, dtype=bool)
sample_cores[dbscan.core_sample_indices_] = True

# Calculate the number of clusters
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print("Number of clusters:", n_clusters)
print("Silhouette score:", metrics.silhouette_score(X_pca, labels))

# Configuration options
num_samples_total = 746
cluster_centers = [(3,7)]
num_classes = len(cluster_centers)
epsilon = 0.3
min_samples = 4

# Generate data
X, y = make_blobs(n_samples = num_samples_total, centers = cluster_centers, n_features = num_classes, center_box=(0, 1), cluster_std = 0.5)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clustering Visualization')
plt.colorbar(label='Cluster Label')
plt.show()