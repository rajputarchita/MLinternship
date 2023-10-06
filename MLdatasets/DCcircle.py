import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler

# Load the dataset from a CSV file (replace 'circle_dataset.csv' with your file path)
df = pd.read_csv('circle.csv')

# Extract the features (assuming your dataset has appropriate column names)
# Adjust column names as per your dataset.
X = df[['0', '1']].values

# If your dataset is not pre-scaled, you might want to standardize it.
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the OPTICS model
optics = OPTICS(min_samples=5, xi=0.5, min_cluster_size=0.05)  # You can adjust parameters as needed

# Fit the OPTICS model to the dataset
optics.fit(X)

# Get the cluster labels
labels = optics.labels_

# Number of clusters found (excluding noise points)
n_clusters_ = len(set(labels)) - 1  # Subtract 1 to exclude noise points

# Print the number of detected clusters
print(f'Number of detected clusters: {n_clusters_}')

# Plot the results
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]  # Black for noise points

    class_member_mask = (labels == k)

    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

plt.title('OPTICS Clustering on "circle" Dataset')
plt.xlabel('0')
plt.ylabel('1')
plt.show()
