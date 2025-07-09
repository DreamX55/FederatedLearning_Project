import os
import numpy as np
import pandas as pd

def check_data_integrity(data_dir='data/processed'):
    '''
    Check data integrity of processed UCI HAR dataset.
    Validates features and labels for consistency.
    '''
    print("=== DATA INTEGRITY CHECK ===")
    
    if not os.path.exists(data_dir):
        print(f"ERROR: {data_dir} does not exist!")
        return False
    
    # Get all client files
    X_files = [f for f in os.listdir(data_dir) if f.endswith('_X.npy')]
    y_files = [f for f in os.listdir(data_dir) if f.endswith('_y.npy')]
    
    print(f"Found {len(X_files)} feature files and {len(y_files)} label files")
    
    if len(X_files) != len(y_files):
        print("ERROR: Mismatch between feature and label files!")
        return False
    
    # Check each client's data
    for i, x_file in enumerate(X_files):
        client_id = x_file.split('_')[1]
        y_file = f'client_{client_id}_y.npy'
        
        if y_file not in y_files:
            print(f"ERROR: Missing label file for client {client_id}")
            return False
        
        # Load data
        X = np.load(os.path.join(data_dir, x_file))
        y = np.load(os.path.join(data_dir, y_file))
        
        # Check dimensions
        print(f"Client {client_id}: X shape = {X.shape}, y shape = {y.shape}")
        
        # Check if samples match
        if X.shape[0] != y.shape[0]:
            print(f"ERROR: Sample count mismatch for client {client_id}")
            return False
        
        # Check feature dimensions (should be 561 for UCI HAR)
        if X.shape[1] != 561:
            print(f"WARNING: Expected 561 features, got {X.shape[1]} for client {client_id}")
        
        # Check label range (should be 0-5 for 6 classes)
        min_label, max_label = y.min(), y.max()
        if min_label < 0 or max_label > 5:
            print(f"ERROR: Invalid label range [{min_label}, {max_label}] for client {client_id}")
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print(f"ERROR: NaN or infinite values in features for client {client_id}")
            return False
            
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            print(f"ERROR: NaN or infinite values in labels for client {client_id}")
            return False
    
    print("✓ All data integrity checks passed!")
    return True

def check_normalization(data_dir='data/processed'):
    '''
    Check if normalization is applied correctly.
    Features should be in [0,1] range after normalization.
    '''
    print("\n=== NORMALIZATION CHECK ===")
    
    # Load all client data and combine to check overall normalization
    all_X = []
    
    X_files = [f for f in os.listdir(data_dir) if f.endswith('_X.npy')]
    
    for x_file in X_files:
        X = np.load(os.path.join(data_dir, x_file))
        all_X.append(X)
    
    # Combine all data
    combined_X = np.vstack(all_X)
    
    # Check normalization statistics
    min_vals = combined_X.min(axis=0)
    max_vals = combined_X.max(axis=0)
    mean_vals = combined_X.mean(axis=0)
    
    print(f"Feature range: [{min_vals.min():.6f}, {max_vals.max():.6f}]")
    print(f"Mean of all features: {mean_vals.mean():.6f}")
    print(f"Std of all features: {combined_X.std():.6f}")
    
    # Check if data is properly normalized to [0,1]
    if min_vals.min() >= 0 and max_vals.max() <= 1:
        print("✓ Data is properly normalized to [0,1] range!")
        return True
    else:
        print("⚠ WARNING: Data may not be properly normalized!")
        print(f"Expected range: [0, 1], Got range: [{min_vals.min():.6f}, {max_vals.max():.6f}]")
        return False

if __name__ == "__main__":
    # Run checks
    integrity_ok = check_data_integrity()
    normalization_ok = check_normalization()
    
    if integrity_ok and normalization_ok:
        print("\n✓ All data validation checks passed!")
    else:
        print("\n⚠ Some data validation checks failed!")
