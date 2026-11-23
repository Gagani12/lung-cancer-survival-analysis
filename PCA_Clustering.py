# PCA DATA PREPARATION

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("\n========== PCA ANALYSIS ==========")

# Select numerical variables
pca_numerical = [
    "Age",
    "BMI",
    "Cigarettes_Per_Day",
    "Years_Smoking",
    "Tumor_Size_cm",
    "Diagnosis_Year"
]

# Select categorical variables
pca_categorical = [
    "Gender",
    "Smoking_Status",
    "Cancer_Stage",
    "Metastasis",
    "Treatment"
]

# Select features
X_pca = df[
    pca_numerical + pca_categorical
].copy()

# Convert categorical variables into dummy variables
X_pca = pd.get_dummies(
    X_pca,
    drop_first=False,
    dtype=int
)

# Standardize all features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_pca)

print("Original Feature Count:", X_pca.shape[1])

print("Scaled Data Shape:", X_scaled.shape)

# APPLY PCA

pca = PCA()

pca_result = pca.fit_transform(X_scaled)

# Explained variance
explained_variance = pca.explained_variance_ratio_

cumulative_variance = np.cumsum(
    explained_variance
)

print("\nExplained Variance Ratio:")

for i, variance in enumerate(
    explained_variance,
    start=1
):

    print(
        f"PC{i}: {variance:.4f} "
        f"({variance * 100:.2f}%)"
    )

print(
    "\nCumulative Variance of First 2 PCs:",
    cumulative_variance[1] * 100
)

# Save PCA results
pca_results_df = pd.DataFrame(
    pca_result,
    columns=[
        f"PC{i+1}"
        for i in range(pca_result.shape[1])
    ]
)

pca_results_df.to_csv(
    OUTPUT_DIR / "pca_results.csv",
    index=False
)


# PCA SCREE PLOT

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, len(explained_variance) + 1),
    cumulative_variance * 100,
    marker="o"
)

plt.axhline(
    y=80,
    linestyle="--",
    label="80% Variance"
)

plt.xlabel("Number of Principal Components")

plt.ylabel("Cumulative Explained Variance (%)")

plt.title("PCA Cumulative Explained Variance")

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "pca_scree_plot.png",
    dpi=300
)

plt.close()

print("PCA scree plot saved!")

# PCA 2D VISUALIZATION

pca_2d = pd.DataFrame({
    "PC1": pca_result[:, 0],
    "PC2": pca_result[:, 1],
    "Cancer_Stage": df["Cancer_Stage"].values
})

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=pca_2d,
    x="PC1",
    y="PC2",
    hue="Cancer_Stage",
    alpha=0.7
)

plt.title("PCA Visualization of Lung Cancer Dataset")

plt.xlabel(
    f"PC1 ({explained_variance[0]*100:.2f}% variance)"
)

plt.ylabel(
    f"PC2 ({explained_variance[1]*100:.2f}% variance)"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "pca_2d_visualization.png",
    dpi=300
)

plt.close()

# FIND OPTIMAL CLUSTERS

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("\n========== CLUSTERING ANALYSIS ==========")

inertia_values = []
silhouette_values = []

k_values = range(2, 8)

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    inertia_values.append(
        kmeans.inertia_
    )

    silhouette_values.append(
        silhouette_score(X_scaled, labels)
    )

# Create evaluation table
cluster_evaluation = pd.DataFrame({
    "K": list(k_values),
    "Inertia": inertia_values,
    "Silhouette_Score": silhouette_values
})

print(cluster_evaluation)

cluster_evaluation.to_csv(
    OUTPUT_DIR / "cluster_evaluation.csv",
    index=False
)

# ELBOW METHOD

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertia_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")

plt.title("Elbow Method for Optimal K")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "elbow_method.png",
    dpi=300
)

plt.close()

# SILHOUETTE SCORE

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    silhouette_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")

plt.title("Silhouette Score Analysis")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "silhouette_analysis.png",
    dpi=300
)

plt.close()

# FINAL K-MEANS CLUSTERING

best_k = cluster_evaluation.loc[
    cluster_evaluation["Silhouette_Score"].idxmax(),
    "K"
]

best_k = int(best_k)

print("\nBest Number of Clusters:", best_k)

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = final_kmeans.fit_predict(
    X_scaled
)

print("\nCluster Distribution:")

print(df["Cluster"].value_counts().sort_index())

# Save cluster assignments
df.to_csv(
    OUTPUT_DIR / "clustered_lung_cancer_data.csv",
    index=False
)

# CLUSTER VISUALIZATION

cluster_plot = pd.DataFrame({
    "PC1": pca_result[:, 0],
    "PC2": pca_result[:, 1],
    "Cluster": df["Cluster"].astype(str)
})

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=cluster_plot,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="Set2",
    alpha=0.7
)

plt.title("K-Means Clusters Visualized Using PCA")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "kmeans_cluster_visualization.png",
    dpi=300
)

plt.close()

# CLUSTER PROFILE ANALYSIS

cluster_profile = df.groupby("Cluster").agg({

    "Age": "mean",

    "BMI": "mean",

    "Tumor_Size_cm": "mean",

    "Survival_Months": "median",

    "Cigarettes_Per_Day": "mean",

    "Cancer_Stage": lambda x: x.mode().iloc[0],

    "Survived": lambda x: x.mode().iloc[0]

})

print("\n========== CLUSTER PROFILES ==========")

print(cluster_profile)

cluster_profile.to_csv(
    OUTPUT_DIR / "cluster_profiles.csv"
)

