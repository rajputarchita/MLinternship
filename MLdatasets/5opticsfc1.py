import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler

# Load data from the MATLAB file
mat_data = scipy.io.loadmat('fc1.mat')  # Replace 'fc1.mat' with your MATLAB file path

# Assuming the data is in a variable named 'fc1_data' in the MATLAB file
data = mat_data['fc1']

# Convert the data to a DataFrame
df = pd.DataFrame(data, columns=['feature1', 'feature2'])  # Adjust column names as needed

# If your dataset is not pre-scaled, you might want to standardize it.
scaler = StandardScaler()
df[['feature1', 'feature2']] = scaler.fit_transform(df[['feature1', 'feature2']])

# Define the OPTICS model
optics = OPTICS(min_samples=5, xi=0.1, min_cluster_size=0.09)  # You can adjust parameters as needed

# Fit the OPTICS model to the DataFrame
optics.fit(df[['feature1', 'feature2']])

# Get the cluster labels
labels = optics.labels_

# Number of clusters found (excluding noise points)
n_clusters_ = len(set(labels)) - 1  # Subtract 1 to exclude noise points

# Print the number of clusters found
print(f'Number of clusters: {n_clusters_}')

# Plot the results (optional)
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]  # Black for noise points

    class_member_mask = (labels == k)

    xy = df[class_member_mask][['feature1', 'feature2']]
    plt.plot(xy['feature1'], xy['feature2'], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

plt.title('OPTICS Clustering on "fc1" Dataset')
plt.show()

# Export the clustered data to a CSV file (optional)
df['cluster_label'] = labels  # Add cluster labels to the DataFrame
df.to_csv('fc1_clustered.csv', index=False)  # Replace 'fc1_clustered.csv' with your desired file name
