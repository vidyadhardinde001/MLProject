import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def calculate_progression_rates(df):
    """Calculate progression rates for cognitive measures."""
    # Get unique patients
    patients = df['RID'].unique()
    progression_data = []
    
    for patient in patients:
        patient_data = df[df['RID'] == patient].sort_values('Years_bl')
        
        if len(patient_data) >= 2:  # Need at least 2 timepoints
            baseline = patient_data.iloc[0]
            final = patient_data.iloc[-1]
            years = final['Years_bl'] - baseline['Years_bl']
            
            if years > 0:
                # Calculate progression rates
                progression = {
                    'RID': patient,
                    'DX_bl': baseline['DX_bl'],
                    'AGE': baseline['AGE'],
                    'Years_Follow_up': years,
                    'MMSE_Rate': (final['MMSE'] - baseline['MMSE']) / years,
                    'ADAS11_Rate': (final['ADAS11'] - baseline['ADAS11']) / years,
                    'ADAS13_Rate': (final['ADAS13'] - baseline['ADAS13']) / years,
                    'CDRSB_Rate': (final['CDRSB'] - baseline['CDRSB']) / years,
                    'RAVLT_immediate_Rate': (final['RAVLT_immediate'] - baseline['RAVLT_immediate']) / years,
                    'RAVLT_learning_Rate': (final['RAVLT_learning'] - baseline['RAVLT_learning']) / years,
                    'FAQ_Rate': (final['FAQ'] - baseline['FAQ']) / years
                }
                
                # Add baseline values
                for measure in ['MMSE', 'ADAS11', 'ADAS13', 'CDRSB', 'RAVLT_immediate', 'RAVLT_learning', 'FAQ']:
                    progression[f'{measure}_bl'] = baseline[measure]
                
                progression_data.append(progression)
    
    return pd.DataFrame(progression_data)

def prepare_clustering_features(progression_df):
    """Prepare features for clustering analysis."""
    # Select features for clustering
    feature_columns = [
        'MMSE_Rate', 'ADAS11_Rate', 'ADAS13_Rate', 'CDRSB_Rate',
        'RAVLT_immediate_Rate', 'RAVLT_learning_Rate', 'FAQ_Rate'
    ]
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(progression_df[feature_columns])
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create DataFrame with scaled features
    scaled_df = pd.DataFrame(
        X_scaled,
        columns=[f'{col}_Scaled' for col in feature_columns],
        index=progression_df.index
    )
    
    return pd.concat([progression_df, scaled_df], axis=1)

if __name__ == "__main__":
    # Load ADNI data
    adni_df = pd.read_csv('ADNIMERGE_23Jun2024.csv')
    
    # Calculate progression rates
    progression_df = calculate_progression_rates(adni_df)
    
    # Prepare features for clustering
    final_df = prepare_clustering_features(progression_df)
    
    # Save processed data
    final_df.to_csv('ADNI_Progression_Data.csv', index=False)
    
    # Print summary statistics
    print("\nProgression Data Summary:")
    print(f"Total patients: {len(final_df)}")
    print("\nBaseline Diagnosis Distribution:")
    print(final_df['DX_bl'].value_counts())
    print("\nProgression Rates Summary (per year):")
    rate_columns = ['MMSE_Rate', 'ADAS11_Rate', 'ADAS13_Rate', 'CDRSB_Rate']
    print(final_df[rate_columns].describe().round(2))