import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.mixture import GaussianMixture

# 📌 Read the CSV file
df = pd.read_csv("DatasetFileName.csv")

# 📌 Drop irrelevant columns
df = df.drop(columns=["filename", "adsorption", "All Metals"], errors='ignore')

# 📌 Dimensionality reduction with PCA for 2D visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(df)

# 📌 📌 📌 1️⃣ K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["KMeans_Cluster"] = kmeans.fit_predict(df)

# 📌 📌 📌 2️⃣ DBSCAN clustering
dbscan = DBSCAN(eps=0.3, min_samples=3)
df["DBSCAN_Cluster"] = dbscan.fit_predict(df)

# 📌 📌 📌 3️⃣ Hierarchical clustering
linkage_matrix = linkage(df, method="ward")
df["Hierarchical_Cluster"] = fcluster(linkage_matrix, t=4, criterion='maxclust')

# 📌 📌 📌 4️⃣ Gaussian Mixture Model (GMM) clustering
gmm = GaussianMixture(n_components=3, random_state=42)
df["GMM_Cluster"] = gmm.fit_predict(df)

# 📌 📊 📊 📊 Plot clustering results
fig, axes = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={'hspace': 0.3, 'wspace': 0.2})
algorithms = ["KMeans_Cluster", "DBSCAN_Cluster", "Hierarchical_Cluster", "GMM_Cluster"]
titles = ["K-Means Clustering", "DBSCAN Clustering", "Hierarchical Clustering", "GMM Clustering"]

for i, (ax, algo, title) in enumerate(zip(axes.flatten(), algorithms, titles)):
    scatter = sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=df[algo], palette="viridis", ax=ax)
    ax.set_title(title, fontname="Times New Roman", fontsize=14)  # Change title font

    # 🟢 Smaller legend in top-left for DBSCAN
    if algo == "DBSCAN_Cluster":
        ax.legend(title="Cluster", loc='upper left', fontsize=9)
    else:
        ax.legend(title="Cluster")

plt.tight_layout()

# 📌 Save the figure
plt.savefig("output.svg", dpi=1200, bbox_inches='tight')

plt.show()
