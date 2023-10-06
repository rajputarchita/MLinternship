import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler

# Load the dataset from a CSV file 
df = pd.read_csv('line.csv')

# Extracting the features (assuming our dataset has columns 'x' and 'y')
#  column names as per our dataset.
X = df[['0', '1']].values

# If our dataset is not pre-scaled, we might want to standardize it.
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Definining the OPTICS model
optics = OPTICS(min_samples=20, xi=0.2, min_cluster_size=0.15)  # You can adjust parameters as needed

# Fitting the OPTICS model to the dataset
optics.fit(X)

# Getting the cluster labels
labels = optics.labels_                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

# Number of clusters found (excluding noise points)
n_clusters_ = len(set(labels)) - 1  # Subtracting 1 to exclude noise points

# Print the number of clusters found
print(f'Number of clusters: {n_clusters_}')

# Plot the results
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]  # Black for noise points

    class_member_mask = (labels == k)

    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

plt.title('OPTICS Clustering on Line Dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
