import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch

# Set random seed for reproducibility
np.random.seed(42)

def load_and_prepare_data(file_path):
    """Load and prepare the ADNI progression data for clustering."""
    df = pd.read_csv(file_path)
    
    # Select features for clustering
    feature_columns = [col for col in df.columns if col.endswith('_Rate_Scaled')]
    X = df[feature_columns].values
    
    return df, X

def find_optimal_k(X, max_k=7):
    """Find optimal number of clusters using silhouette score."""
    silhouette_scores = []
    for k in range(2, max_k):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        score = silhouette_score(X, cluster_labels)
        silhouette_scores.append(score)
        print(f"k={k}: Silhouette Score = {score:.4f}")
    
    optimal_k = np.argmax(silhouette_scores) + 2
    return optimal_k

def perform_clustering(X, k):
    """Perform multiple clustering methods."""
    # K-Means
    kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_labels = kmeans_model.fit_predict(X)
    
    # Hierarchical Clustering
    hiera_model = AgglomerativeClustering(n_clusters=k, linkage='ward')
    hiera_labels = hiera_model.fit_predict(X)
    
    # DBSCAN
    dbscan_model = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan_model.fit_predict(X)
    
    return kmeans_labels, hiera_labels, dbscan_labels, kmeans_model

def visualize_clusters(X, df, kmeans_labels, hiera_labels, dbscan_labels, k):
    """Create visualizations for clustering results."""
    # PCA for dimensionality reduction
    pca = PCA(n_components=2, random_state=42)
    principal_components = pca.fit_transform(X)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    
    # Add clustering results
    pca_df['KMeans_Cluster'] = kmeans_labels
    pca_df['Hierarchical_Cluster'] = hiera_labels
    pca_df['DBSCAN_Cluster'] = dbscan_labels
    pca_df['DX_bl'] = df['DX_bl']
    
    # Plot 1: K-Means Clustering Results
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=pca_df,
        x='PC1',
        y='PC2',
        hue='KMeans_Cluster',
        style='DX_bl',
        palette='Set1',
        s=100
    )
    plt.title(f'Cognitive Decline Profiles Clusters (K-Means, k={k})')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.grid(True)
    plt.show()
    
    # Plot 2: Hierarchical Clustering Dendrogram
    plt.figure(figsize=(15, 8))
    Z = sch.linkage(X, method='ward')
    sch.dendrogram(
        Z,
        truncate_mode='lastp',
        p=30,
        show_leaf_counts=True,
        leaf_rotation=90.,
        leaf_font_size=10.,
    )
    plt.title('Hierarchical Clustering of Cognitive Decline Profiles')
    plt.xlabel('Patient Index')
    plt.ylabel('Distance')
    plt.show()
    
    return pca_df

def analyze_clusters(df, kmeans_labels, kmeans_model):
    """Analyze the characteristics of each cluster."""
    df['Cluster'] = kmeans_labels
    
    # Analyze progression rates by cluster
    rate_columns = [col for col in df.columns if col.endswith('_Rate') and not col.endswith('_Rate_Scaled')]
    cluster_profiles = df.groupby('Cluster')[rate_columns].mean()
    
    # Analyze diagnosis distribution in each cluster
    diagnosis_dist = pd.crosstab(df['Cluster'], df['DX_bl'], normalize='index') * 100
    
    print("\nCluster Profiles (Mean Annual Change):")
    print(cluster_profiles.round(2))
    print("\nDiagnosis Distribution in Clusters (%):")
    print(diagnosis_dist.round(1))
    
    return cluster_profiles, diagnosis_dist

if __name__ == "__main__":
    # Load and prepare data
    df, X = load_and_prepare_data('ADNI_Progression_Data.csv')
    
    # Find optimal number of clusters
    print("Finding optimal number of clusters...")
    k = find_optimal_k(X)
    print(f"\nOptimal number of clusters: {k}")
    
    # Perform clustering
    kmeans_labels, hiera_labels, dbscan_labels, kmeans_model = perform_clustering(X, k)
    
    # Visualize results
    pca_df = visualize_clusters(X, df, kmeans_labels, hiera_labels, dbscan_labels, k)
    
    # Analyze cluster characteristics
    cluster_profiles, diagnosis_dist = analyze_clusters(df, kmeans_labels, kmeans_model)