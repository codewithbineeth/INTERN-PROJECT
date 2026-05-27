import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Configuration & Loading Data
file_path = (
    r"C:\Users\SWAMY\Desktop\project\level2-intermediate\task3\churn-bigml-80.csv"
)

if not os.path.exists(file_path):
    print(f"Error: Could not locate dataset file at: {file_path}")
    exit()

df = pd.read_csv(file_path)

print("=" * 65)
print("📌 INTERMEDIATE LEVEL 2 TASK 3: CUSTOMER SEGMENTATION CLUSTERING")
print("=" * 65)

# --- PREPROCESSING ---
# Select core numerical metrics tracking customer usage activity
numerical_cols = [
    "Account length",
    "Number vmail messages",
    "Total day minutes",
    "Total day calls",
    "Total day charge",
    "Total eve minutes",
    "Total eve calls",
    "Total eve charge",
    "Total night minutes",
    "Total night calls",
    "Total night charge",
    "Total intl minutes",
    "Total intl calls",
    "Total intl charge",
    "Customer service calls",
]

X = df[numerical_cols].copy()

# Scale features so that distance-based metrics (like K-Means uses) are accurate
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==========================================
#  OBJECTIVE 1 & 2: THE ELBOW METHOD FOR OPTIMAL K
# ==========================================
print("\ Step 1: Calculating Within-Cluster Sum of Squares (WCSS)...")
wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)  # Inertia is the WCSS value

# Plot the Elbow Curve to find the "bend" point
plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker="o", linestyle="--", color="b", linewidth=2)
plt.title("The Elbow Method to Find Optimal Clusters", fontsize=12, fontweight="bold")
plt.xlabel("Number of Clusters (K)", fontsize=10)
plt.ylabel("WCSS (Inertia)", fontsize=10)
plt.xticks(k_range)
plt.grid(True, linestyle=":", alpha=0.6)

elbow_chart = "clustering_elbow_method.png"
plt.savefig(elbow_chart, dpi=300, bbox_inches="tight")
plt.close()
print(f"    Saved Elbow Selection Plot: '{elbow_chart}'")


# ==========================================
#  OBJECTIVE 3: APPLY FINAL K-MEANS CLUSTERING
# ==========================================
# Based on common data variance, 3 or 4 groups represent clean segments. We will use K=3.
OPTIMAL_K = 3
print(f"\n Step 2: Running K-Means with K={OPTIMAL_K} segments...")
final_kmeans = KMeans(n_clusters=OPTIMAL_K, init="k-means++", random_state=42)
cluster_labels = final_kmeans.fit_predict(X_scaled)

# Append cluster designations back to the original dataframe for business analysis
df["Cluster_ID"] = cluster_labels


# ==========================================
#  OBJECTIVE 4: DIMENSIONALITY REDUCTION VIA PCA
# ==========================================
print("\n Step 3: Reducing dimensions using PCA for 2D visualization...")
# PCA squashes 15 dimensions of data into 2 meta-components while retaining structural variance
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(data=X_pca, columns=["PCA_Component_1", "PCA_Component_2"])
df_pca["Cluster"] = cluster_labels

# Plot the clusters in 2D Space
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df_pca,
    x="PCA_Component_1",
    y="PCA_Component_2",
    hue="Cluster",
    palette="Set1",
    s=60,
    alpha=0.8,
    edgecolor="w",
)
plt.title(
    "Customer Segments Visualized in 2D Space via PCA", fontsize=13, fontweight="bold"
)
plt.xlabel("Principal Component 1", fontsize=11)
plt.ylabel("Principal Component 2", fontsize=11)
plt.legend(title="Customer Group", loc="upper right")
plt.grid(True, linestyle=":", alpha=0.5)

scatter_chart = "clustering_pca_segments.png"
plt.savefig(scatter_chart, dpi=300, bbox_inches="tight")
plt.close()
print(f"   Saved 2D Cluster Visual: '{scatter_chart}'")


# ==========================================
#  OBJECTIVE 5: INTERPRET CLUSTERING FINDINGS
# ==========================================
print("\n Step 4: Group Summary Profiles (Mean Performance Metrics):")
cluster_summary = df.groupby("Cluster_ID")[
    ["Total day minutes", "Total night minutes", "Customer service calls"]
].mean()
print(cluster_summary)

print("\n" + "=" * 65)
print(" Unsupervised Learning Tasks Complete!")
print("=" * 65)