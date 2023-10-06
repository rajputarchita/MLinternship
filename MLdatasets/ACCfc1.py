import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
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
optics = OPTICS(min_samples=5, xi=0.1, min_cluster_size=0.05, metric='minkowski', p=2)  # You can adjust parameters as needed

# Fit the OPTICS model to the dataset
optics.fit(X)

# Get the cluster labels
labels = optics.labels_

# Calculate the silhouette score
silhouette_avg = silhouette_score(X, labels)

# Print the silhouette score
print(f'Silhouette Score: {silhouette_avg}')
