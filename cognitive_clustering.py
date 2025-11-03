# Import necessary libraries
import pandas as pd              # For data handling
import numpy as np              # For numerical operations
from sklearn.preprocessing import StandardScaler  # For scaling data
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA            # For visualization
import matplotlib.pyplot as plt                  # For creating plots
import seaborn as sns                           # For better plots
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Step 1: Load the patient data
print("Reading patient data...")
df = pd.read_csv('ADNIMERGE_23Jun2024.csv', low_memory=False)

# Define the cognitive tests we'll use for analysis
# These are standard tests used to assess memory and thinking abilities
cognitive_measures = [
    'MMSE',    # Mini-Mental State Examination (0-30, higher is better)
    'CDRSB',   # Clinical Dementia Rating Sum of Boxes (0-18, lower is better)
    'ADAS11',  # Alzheimer's Disease Assessment Scale (higher score = more impairment)
    'ADAS13'   # Extended version of ADAS11 with additional tasks
]

# Function to prepare data for analysis
def prepare_data(df, measures):
    """
    Prepare the patient data for analysis by:
    1. Getting the first visit data for each patient
    2. Removing any incomplete records
    3. Making sure all measurements are on the same scale
    """
    # Step 1: Get first visit data (baseline) for each patient
    print("Getting first visit data for each patient...")
    baseline_data = df.groupby('RID')[measures + ['DX']].first().reset_index()
    
    # Step 2: Remove incomplete records
    print("Removing incomplete records...")
    clean_data = baseline_data.dropna(subset=measures)
    
    # Step 3: Scale all measurements to be comparable
    print("Scaling measurements...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clean_data[measures])
    
    return clean_data, scaled_features

# Function to find the best number of patient groups
def find_optimal_clusters(data, max_clusters=10):
    """
    Find the best number of groups to divide patients into by:
    1. Trying different numbers of groups (2 to max_clusters)
    2. Measuring how well each grouping works
    3. Using both elbow method and silhouette score
    """
    print("Finding the best number of patient groups...")
    inertias = []          # List to store how tight each grouping is
    silhouette_scores = [] # List to store how well-separated groups are
    
    # Try different numbers of groups
    for k in range(2, max_clusters + 1):
        # Create and fit the model
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        
        # Store the results
        inertias.append(kmeans.inertia_)  # Lower is better
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))  # Higher is better
    
    return range(2, max_clusters + 1), inertias, silhouette_scores

# Function to create visual representation of patient groups
def visualize_clusters(data, labels, centers=None):
    """
    Create a 2D visualization of how patients group together:
    1. Reduce the complex data to 2D using PCA
    2. Plot each patient as a point
    3. Color points based on their group
    4. Mark group centers with X
    """
    print("Creating visual representation of patient groups (PCA)...")

    # 1) Reduce data to 2D using PCA
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)

    # 2) Scatter plot of patients in PCA space
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', s=20, alpha=0.7)

    # 3) If cluster centers are provided (in original scaled space), project them to PCA space and mark
    if centers is not None:
        centers_2d = pca.transform(centers)
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='x', s=200, linewidths=3)

    # 4) Variance explained by each principal component (as percent)
    variance_explained = pca.explained_variance_ratio_ * 100

    plt.title('Patient Groups Based on Cognitive Tests')
    plt.xlabel(f'Combined Measure 1 ({variance_explained[0]:.1f}% variance)')
    plt.ylabel(f'Combined Measure 2 ({variance_explained[1]:.1f}% variance)')
    plt.colorbar(scatter, label='Group Number')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('cluster_visualization.png', dpi=150)
    plt.close()

    # 5) Compute how much each original test contributes to each combined measure
    contributions = pd.DataFrame(
        pca.components_,
        columns=cognitive_measures,
        index=['Combined Measure 1', 'Combined Measure 2']
    )

    # Save contributions as CSV for later inspection
    contributions.to_csv('pca_contributions.csv')

    # 6) Plot contributions (bar chart)
    plt.figure(figsize=(10, 6))
    contributions.T.plot(kind='bar')
    plt.title('Contribution of Original Cognitive Tests to Combined Measures')
    plt.xlabel('Original Cognitive Tests')
    plt.ylabel('Component Weight (signed)')
    plt.legend(title='Combined Measures')
    plt.tight_layout()
    plt.savefig('pca_contributions.png', dpi=150)
    plt.close()

    # 7) Produce a short human-readable explanation file with the numeric contributions
    explanation_lines = []
    explanation_lines.append('# Explanation of Combined Measures (PCA)')
    explanation_lines.append('')
    explanation_lines.append(f'Combined Measure 1 explains {variance_explained[0]:.1f}% of the data variance.')
    explanation_lines.append(f'Combined Measure 2 explains {variance_explained[1]:.1f}% of the data variance.')
    explanation_lines.append('')
    explanation_lines.append('Component weights (how each original test contributes to each combined measure):')
    explanation_lines.append('')
    # Build a simple markdown table of contributions (avoid external optional dependency)
    contrib_rounded = contributions.round(3)
    md_table_lines = []
    header = ['Measure'] + contrib_rounded.columns.tolist()
    md_table_lines.append('| ' + ' | '.join(header) + ' |')
    md_table_lines.append('| ' + ' | '.join(['---'] * len(header)) + ' |')
    for idx in contrib_rounded.index:
        row = [idx] + [str(x) for x in contrib_rounded.loc[idx].values]
        md_table_lines.append('| ' + ' | '.join(row) + ' |')
    explanation_lines.append('\n'.join(md_table_lines))
    explanation_lines.append('')
    explanation_lines.append('Interpretation hints:')
    explanation_lines.append("- A positive weight means the original test increases the combined measure value.")
    explanation_lines.append("- A negative weight means the original test decreases the combined measure value.")
    explanation_lines.append("- If Combined Measure 1 has large positive weights on impairment tests (ADAS) and negative on MMSE, higher Combined Measure 1 indicates worse impairment.")

    with open('combined_measures_explanation.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(explanation_lines))

    # 8) Print a short console summary for user clarity
    print('\n--- PCA Summary ---')
    print(f'Combined Measure 1 variance explained: {variance_explained[0]:.1f}%')
    print(f'Combined Measure 2 variance explained: {variance_explained[1]:.1f}%')
    print('\nContributions (rows=Combined Measures, columns=Original tests):')
    print(contributions.round(3))
    print('\nCreated files: cluster_visualization.png, pca_contributions.png, pca_contributions.csv, combined_measures_explanation.md')

# Function to analyze what makes each patient group unique
def analyze_clusters(data, measures, cluster_labels):
    """
    Analyze each patient group to understand:
    1. Average test scores for each group
    2. How test scores vary within groups
    3. Distribution of diagnoses in each group
    """
    print("Analyzing characteristics of each patient group...")
    # Add group labels to the data
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = cluster_labels
    
    # Calculate average scores and variation for each group
    print("Calculating test score statistics...")
    numeric_profiles = data_with_clusters.groupby('Cluster')[measures].agg(['mean', 'std'])
    
    # Calculate percentage of each diagnosis in each group
    print("Analyzing diagnosis distributions...")
    dx_distribution = pd.crosstab(data_with_clusters['Cluster'], 
                                data_with_clusters['DX'], 
                                normalize='index') * 100  # Convert to percentages
    
    return numeric_profiles, dx_distribution

def run_clustering_algorithms(data):
    """Run multiple clustering algorithms and return their results"""
    results = {}
    
    # Define clustering algorithms
    algorithms = {
        'K-Means': KMeans(n_clusters=4, random_state=42),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
        'Hierarchical': AgglomerativeClustering(n_clusters=4),
        'Spectral': SpectralClustering(n_clusters=4, random_state=42)
    }
    
    # Run each algorithm
    for name, algo in algorithms.items():
        print(f"\nRunning {name} clustering...")
        labels = algo.fit_predict(data)
        
        # Calculate evaluation metrics
        sil_score = silhouette_score(data, labels) if len(np.unique(labels)) > 1 else 0
        cal_score = calinski_harabasz_score(data, labels)
        
        results[name] = {
            'labels': labels,
            'silhouette': sil_score,
            'calinski': cal_score
        }
        
        print(f"{name} - Silhouette Score: {sil_score:.3f}, Calinski-Harabasz Score: {cal_score:.3f}")
    
    return results

def visualize_all_clusters(data, clustering_results):
    """Visualize results from all clustering algorithms"""
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    
    # Create subplot for each algorithm
    n_algorithms = len(clustering_results)
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    for idx, (name, result) in enumerate(clustering_results.items()):
        ax = axes[idx]
        scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], 
                           c=result['labels'], cmap='viridis')
        ax.set_title(f'{name}\nSilhouette: {result["silhouette"]:.3f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('all_clustering_results.png', dpi=150)
    plt.close()

# Main analysis function
def main():
    """
    Main function that runs the entire analysis:
    1. Prepare the data
    2. Find optimal number of groups
    3. Create visualizations
    4. Analyze group characteristics
    """
    # Step 1: Prepare the data
    print("\n=== Step 1: Preparing Patient Data ===")
    clean_data, scaled_features = prepare_data(df, cognitive_measures)
    
    # Step 2: Find best number of groups
    print("\n=== Step 2: Determining Optimal Number of Groups ===")
    k_range, inertias, silhouette_scores = find_optimal_clusters(scaled_features)
    
    # Create plots to help visualize the optimal number of groups
    print("Creating plots to show optimal number of groups...")
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Elbow curve
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, 'bx-')
    plt.xlabel('Number of Groups')
    plt.ylabel('Within-Group Variance')
    plt.title('Elbow Method: Finding Optimal Number of Groups')
    
    # Plot 2: Silhouette scores
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'rx-')
    plt.xlabel('Number of Groups')
    plt.ylabel('Silhouette Score')
    plt.title('Group Separation Quality')
    
    plt.tight_layout()
    plt.savefig('optimal_clusters.png')
    plt.close()
    
    # Step 3: Group the patients
    print("\n=== Step 3: Creating Patient Groups ===")
    optimal_k = 4  # We use 4 groups based on medical relevance and statistical analysis
    print(f"Creating {optimal_k} patient groups...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # Step 4: Create visualizations
    print("\n=== Step 4: Creating Visualizations ===")
    visualize_clusters(scaled_features, cluster_labels, kmeans.cluster_centers_)
    
    # Step 5: Analyze each group
    print("\n=== Step 5: Analyzing Group Characteristics ===")
    numeric_profiles, dx_distribution = analyze_clusters(clean_data, cognitive_measures, cluster_labels)
    
    # Step 6: Run and visualize multiple clustering algorithms
    print("\n=== Running Multiple Clustering Algorithms ===")
    clustering_results = run_clustering_algorithms(scaled_features)
    
    print("\n=== Visualizing All Clustering Results ===")
    visualize_all_clusters(scaled_features, clustering_results)
    
    # Analyze the best performing algorithm's results (based on silhouette score)
    best_algorithm = max(clustering_results.items(), 
                        key=lambda x: x[1]['silhouette'])[0]
    best_labels = clustering_results[best_algorithm]['labels']
    
    print(f"\nUsing results from {best_algorithm} for detailed analysis (best silhouette score)")
    numeric_profiles, dx_distribution = analyze_clusters(clean_data, 
                                                       cognitive_measures, 
                                                       best_labels)
    
    # Save all results
    print("\n=== Saving Results ===")
    numeric_profiles.to_csv('cluster_numeric_profiles.csv')
    dx_distribution.to_csv('cluster_dx_distribution.csv')
    
    # Print summary of findings
    print("\n=== Summary of Findings ===")
    print("\nSize of each patient group:")
    group_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    for group in range(len(group_sizes)):
        print(f"Group {group}: {group_sizes[group]} patients")
    
    print("\nDiagnosis distribution in each group (%):")
    print("CN = Cognitively Normal")
    print("MCI = Mild Cognitive Impairment")
    print(dx_distribution)
    
    print("\nFiles created:")
    print("1. cluster_visualization.png - Visual map of patient groups")
    print("2. optimal_clusters.png - Analysis of optimal group numbers")
    print("3. cluster_numeric_profiles.csv - Detailed test scores for each group")
    print("4. cluster_dx_distribution.csv - Diagnosis patterns in each group")
    print("5. all_clustering_results.png - Comparison of clustering algorithms")

if __name__ == "__main__":
    main()