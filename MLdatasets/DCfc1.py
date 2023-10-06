import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler

# Load the dataset from a CSV file (replace 'fc1_dataset.csv' with your file path)
df = pd.read_csv('fc1.csv')

# Extract the features (assuming your dataset has appropriate column names)
# Adjust column names as per your dataset.
X = df[['feature1', 'feature2']].values

# If your dataset is not pre-scaled, you might want to standardize it.
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Define the OPTICS model
optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05, metric='minkowski', p=2)  # You can adjust parameters as needed

# Fit the OPTICS model to the dataset
optics.fit(X)

# Get the cluster labels
labels = optics.labels_

# Number of clusters found (excluding noise points)
n_clusters_ = len(set(labels)) - 1  # Subtract 1 to exclude noise points

# Print the number of detected clusters
print(f'Number of detected clusters: {n_clusters_}')
