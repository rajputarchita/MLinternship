import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# Load the dataset from a CSV file (replace 'jain.csv' with your file path)
df = pd.read_csv('LINEDataset9.csv')

# Extract the features (assuming your dataset has columns 'feature1' and 'feature2')
# Adjust column names as per your dataset.
X = df[['1', '2']].values

# Load the true labels (replace 'true_labels_column' with the actual column name containing the true labels)
true_labels = df['3'].values

# If your dataset is not pre-scaled, you might want to standardize it.
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the OPTICS model
optics = OPTICS(min_samples=5, xi=0.07, min_cluster_size=0.07)  # You can adjust parameters as needed

# Fit the OPTICS model to the dataset
optics.fit(X)

# Get the cluster labels
labels = optics.labels_

# Number of clusters found (excluding noise points)
n_clusters_ = len(set(labels)) - 1  # Subtract 1 to exclude noise points

# Calculate the silhouette score
silhouette_avg = silhouette_score(X, labels)

# Calculate the adjusted mutual information (AMI) score without true labels
ami_score = adjusted_mutual_info_score(true_labels, labels)

# Calculate the adjusted Rand index (ARI) without true labels
ari_score = adjusted_rand_score(true_labels, labels)



# Print the number of clusters found
print(f'Number of clusters: {n_clusters_}')
print(f'Adjusted Mutual Information (AMI) Score: {ami_score}')
# Print  the ARI score
print(f'Adjusted Rand Index (ARI) Score: {ari_score}')
print(f'Silhouette Score: {silhouette_avg}')
print(f'Number of detected clusters: {n_clusters_}')
plt.title('OPTICS Clustering on "line" Dataset')
plt.title('OPTICS Clustering on "line" Dataset with AMI Score')
plt.title('OPTICS Clustering on "line" Dataset with Rand Index')

# Plot the results
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]  # Black for noise points

    class_member_mask = (labels == k)

    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

plt.xlabel('1')
plt.ylabel('2')
plt.show()