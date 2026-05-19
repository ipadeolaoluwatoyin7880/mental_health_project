"""
Mental Health in Tech - Unsupervised Learning Analysis
Course: DLBDSMLUSL01
"""

# ============================================================================
# IMPORT LIBRARIES
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import VarianceThreshold
import warnings
from math import pi

warnings.filterwarnings('ignore')

print("✅ Libraries imported!")

# ============================================================================
# SETUP PATHS (IMPORTANT FOR ORGANIZED STRUCTURE)
# ============================================================================

# Get the project root (go up one level from 'code' folder)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Define paths
data_path = os.path.join(project_root, 'data', 'survey.csv')
outputs_path = os.path.join(project_root, 'outputs')
figures_path = os.path.join(outputs_path, 'figures')

# Create folders if they don't exist
os.makedirs(figures_path, exist_ok=True)

print(f"📁 Project: {project_root}")
print(f"📁 Data: {data_path}")
print(f"📁 Outputs: {outputs_path}")

# ============================================================================
# LOAD DATA
# ============================================================================

try:
    df = pd.read_csv(data_path)
    print(f"\n✅ Data loaded! Shape: {df.shape}")
except FileNotFoundError:
    print(f"\n❌ Error: survey.csv not found in data folder")
    print(f"   Please put the file here: {data_path}")
    exit()

# ============================================================================
# HANDLE MISSING VALUES
# ============================================================================

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

if len(numeric_cols) > 0:
    num_imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    print("✅ Missing values imputed (numeric)")

if len(categorical_cols) > 0:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    print("✅ Missing values imputed (categorical)")

# ============================================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================================

skip_cols = ['timestamp', 'Timestamp', 'time', 'Date', 'comments', 'state']
categorical_to_encode = [col for col in categorical_cols if col not in skip_cols]

df_encoded = pd.get_dummies(df, columns=categorical_to_encode, drop_first=True)
df_numeric = df_encoded.select_dtypes(include=[np.number])

print(f"✅ Features after encoding: {df_numeric.shape[1]}")

# ============================================================================
# REMOVE LOW VARIANCE FEATURES
# ============================================================================

selector = VarianceThreshold(threshold=0)
df_filtered = pd.DataFrame(selector.fit_transform(df_numeric))
print(f"✅ After removing zero-variance: {df_filtered.shape[1]} features")

# ============================================================================
# SCALE FEATURES
# ============================================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_filtered)
print(f"✅ Feature scaling complete: {X_scaled.shape}")

# ============================================================================
# PCA
# ============================================================================

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
print(f"\n✅ PCA: {n_components_90} components for 90% variance")

# 2D for visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# ============================================================================
# FIND OPTIMAL K
# ============================================================================

silhouette_scores = []
K_range = range(2, 11)

print("\n📊 Finding optimal clusters...")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)
    print(f"   k={k}: Silhouette={score:.3f}")

optimal_k = K_range[np.argmax(silhouette_scores)]
best_silhouette = max(silhouette_scores)
print(f"\n✅ OPTIMAL k = {optimal_k} (Score: {best_silhouette:.3f})")

# ============================================================================
# FINAL CLUSTERING
# ============================================================================

final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = final_kmeans.fit_predict(X_scaled)
df['Cluster'] = cluster_labels

print("\n📊 Cluster sizes:")
for i in range(optimal_k):
    size = np.sum(cluster_labels == i)
    print(f"   Cluster {i}: {size} ({size/len(df)*100:.1f}%)")

# ============================================================================
# FIGURE 1: PCA VARIANCE
# ============================================================================

plt.figure(figsize=(10, 6))
plt.bar(range(1, min(16, len(explained_variance)+1)), explained_variance[:15], alpha=0.7)
plt.plot(range(1, min(16, len(cumulative_variance)+1)), cumulative_variance[:15], 'ro-', linewidth=2)
plt.axhline(y=0.90, color='g', linestyle='--', label='90% Variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance')
plt.title('Figure 1: PCA Variance Analysis')
plt.legend(['Cumulative', 'Individual', '90%'])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'figure1_pca_variance.png'), dpi=300)
plt.close()
print("\n✅ Figure 1 saved")

# ============================================================================
# FIGURE 2: CLUSTERS IN PCA SPACE
# ============================================================================

plt.figure(figsize=(12, 8))
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6, s=50)

centroids_pca = pca_2d.transform(final_kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=300, c='red', edgecolors='black')

plt.colorbar(label='Cluster')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title(f'Figure 2: Clusters in PCA Space (k={optimal_k})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'figure2_clusters_pca.png'), dpi=300)
plt.close()
print("✅ Figure 2 saved")

# ============================================================================
# GET CLUSTER PERCENTAGES (FOR YOUR WRITE-UP)
# ============================================================================

print("\n" + "="*60)
print("CLUSTER PERCENTAGES - COPY FOR YOUR WRITE-UP")
print("="*60)

# Look for relevant columns
for col in ['treatment', 'family_history', 'remote_work', 'tech_company']:
    if col in df.columns:
        print(f"\n📊 {col.upper()}:")
        for i in range(optimal_k):
            cluster_df = df[df['Cluster'] == i]
            if df[col].dtype == 'object':
                pct = (cluster_df[col] == 'Yes').mean() * 100
            else:
                pct = cluster_df[col].mean() * 100
            print(f"   Cluster {i}: {pct:.1f}%")

# ============================================================================
# SAVE RESULTS
# ============================================================================

df.to_csv(os.path.join(outputs_path, 'clustered_data.csv'), index=False)
print(f"\n✅ Results saved to: {outputs_path}/clustered_data.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print(f"\n📊 SUMMARY:")
print(f"   Participants: {df.shape[0]}")
print(f"   Features: {X_scaled.shape[1]}")
print(f"   Optimal k: {optimal_k}")
print(f"   Silhouette: {best_silhouette:.3f}")
print(f"   Variance (2 PC): {cumulative_variance[1]:.2%}")

print(f"\n📁 OUTPUT FILES:")
print(f"   {figures_path}/figure1_pca_variance.png")
print(f"   {figures_path}/figure2_clusters_pca.png")
print(f"   {outputs_path}/clustered_data.csv")
print("\n" + "="*60)