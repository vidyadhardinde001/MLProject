# Import necessary libraries
import pandas as pd              # For data handling
import numpy as np              # For numerical operations
from sklearn.preprocessing import StandardScaler  # For scaling data
from sklearn.cluster import KMeans               # For grouping patients
from sklearn.decomposition import PCA            # For visualization
import matplotlib.pyplot as plt                  # For creating plots
import seaborn as sns                           # For better plots
from sklearn.metrics import silhouette_score     # For evaluating clusters

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
    print("Creating visual representation of patient groups...")
    # Reduce data to 2D for visualization
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis')
    
    # Add group centers if provided
    if centers is not None:
        centers_2d = pca.transform(centers)
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='x', s=200, linewidths=3)
    
    # Add labels and save
    plt.title('Patient Groups Based on Cognitive Tests')
    plt.xlabel('Combined Measure 1')
    plt.ylabel('Combined Measure 2')
    plt.colorbar(scatter, label='Group Number')
    plt.savefig('cluster_visualization.png')
    plt.close()

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

if __name__ == "__main__":
    main()