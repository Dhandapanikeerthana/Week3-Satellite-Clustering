from ucimlrepo import fetch_ucirepo
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Download the real Landsat Satellite dataset
landsat = fetch_ucirepo(id=146)

# Get features and target
X = landsat.data.features
y = landsat.data.targets

# Create dataset folder path
dataset_folder = "dataset"

# Save the actual dataset files
X.to_csv(os.path.join(dataset_folder, "landsat_features.csv"), index=False)
y.to_csv(os.path.join(dataset_folder, "landsat_target.csv"), index=False)

print("Landsat dataset downloaded successfully!")
print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\nFirst 5 rows:")
print(X.head())

print("\nDataset information:")
print(X.info())

print("\nMissing values:")
print(X.isnull().sum().sum())

print("\nNumber of duplicate rows:")
print(X.duplicated().sum())

print("\nFeature ranges:")
print(X.describe().T[['min', 'max']])

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled data shape:")
print(X_scaled.shape)

print("\nFirst 5 scaled rows:")
print(X_scaled[:5])

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

inertia = []
silhouette_scores = []

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

print("\nInertia values:")
print(inertia)

print("\nSilhouette scores:")
print(silhouette_scores)

# Final K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

cluster_labels = kmeans.fit_predict(X_scaled)

print("\nCluster counts:")
print(pd.Series(cluster_labels).value_counts().sort_index())

# Add cluster labels to the original feature data
X_clustered = X.copy()
X_clustered["Cluster"] = cluster_labels

print("\nCluster characteristics:")
print(X_clustered.groupby("Cluster").mean())

import matplotlib.pyplot as plt

# -----------------------------
# Elbow Method Graph
# -----------------------------

plt.figure(figsize=(8, 5))
plt.plot(range(2, 9), inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.xticks(range(2, 9))
plt.grid(True)

plt.savefig("visualization/elbow_method.png")
plt.show()


# -----------------------------
# Silhouette Score Graph
# -----------------------------

plt.figure(figsize=(8, 5))
plt.plot(range(2, 9), silhouette_scores, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different K Values")
plt.xticks(range(2, 9))
plt.grid(True)

plt.savefig("visualization/silhouette_scores.png")
plt.show()


from sklearn.decomposition import PCA

# -----------------------------
# PCA for 2D Visualization
# -----------------------------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nPCA shape:")
print(X_pca.shape)

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal variance explained:")
print(pca.explained_variance_ratio_.sum())

# -----------------------------
# PCA Cluster Visualization
# -----------------------------

plt.figure(figsize=(8, 6))

for cluster in range(3):
    plt.scatter(
        X_pca[cluster_labels == cluster, 0],
        X_pca[cluster_labels == cluster, 1],
        label=f"Cluster {cluster}",
        alpha=0.6
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clusters Visualized using PCA")
plt.legend()
plt.grid(True)

plt.savefig("visualization/pca_clusters.png")
plt.show()

# -----------------------------
# Compare Clusters with Original Classes
# -----------------------------

comparison = pd.crosstab(
    pd.Series(cluster_labels, name="Cluster"),
    y.iloc[:, 0]
)

print("\nCluster vs Original Class:")
print(comparison)

# -----------------------------
# Save Final Clustered Dataset
# -----------------------------

final_data = X.copy()
final_data["Cluster"] = cluster_labels
final_data["Original_Class"] = y.iloc[:, 0].values

final_data.to_csv(
    "dataset/landsat_clustered.csv",
    index=False
)

print("\nFinal clustered dataset saved successfully!")

# -----------------------------
# Compare Clusters with Original Classes
# -----------------------------

comparison = pd.crosstab(
    pd.Series(cluster_labels, name="Cluster"),
    y.iloc[:, 0]
)

print("\nCluster vs Original Class:")
print(comparison)


# -----------------------------
# Cluster vs Class Heatmap
# -----------------------------

import seaborn as sns

plt.figure(figsize=(8, 5))

sns.heatmap(
    comparison,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("K-Means Clusters vs Original Land-Cover Classes")
plt.xlabel("Original Class")
plt.ylabel("K-Means Cluster")

plt.savefig("visualization/cluster_class_heatmap.png")
plt.show()




# -----------------------------
# Compare Clusters with Original Classes
# -----------------------------

comparison = pd.crosstab(
    pd.Series(cluster_labels, name="Cluster"),
    y.iloc[:, 0]
)

print("\nCluster vs Original Class:")
print(comparison)


# -----------------------------
# Cluster Percentage Distribution
# -----------------------------

cluster_percentage = comparison.div(
    comparison.sum(axis=1),
    axis=0
) * 100

print("\nCluster Percentage Distribution:")
print(cluster_percentage.round(2))


# -----------------------------
# Cluster vs Class Heatmap
# -----------------------------

import seaborn as sns

plt.figure(figsize=(8, 5))

sns.heatmap(
    comparison,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("K-Means Clusters vs Original Land-Cover Classes")
plt.xlabel("Original Class")
plt.ylabel("K-Means Cluster")

plt.savefig("visualization/cluster_class_heatmap.png")
plt.show()


# -----------------------------
# Save Final Clustered Dataset
# -----------------------------

final_data = X.copy()
final_data["Cluster"] = cluster_labels
final_data["Original_Class"] = y.iloc[:, 0].values

final_data.to_csv(
    "dataset/landsat_clustered.csv",
    index=False
)

# Save cluster feature characteristics
cluster_profile = final_data.groupby("Cluster").mean()

cluster_profile.to_csv(
    "dataset/cluster_profiles.csv"
)

print("\nFinal clustered dataset saved successfully!")
print("Cluster profile saved successfully!")

print("\nCluster Profiles:")
print(cluster_profile)