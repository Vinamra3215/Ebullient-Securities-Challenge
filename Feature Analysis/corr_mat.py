import numpy as np
import cupy as cp
import dask_cudf
import cudf
import pandas as pd
from scipy.stats import pearsonr
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION PARAMETERS - MODIFIED
# ============================================================================
DATA_FILE = '/data/quant14/EBX/combined.csv'
BLOCKSIZE = '256MB'  # Size of each chunk for processing
OUTPUT_DIR = '.'  # Directory to save results

# FEATURE SELECTION:
# <--- IMPORTANT: List the exact T11 feature names as they appear in your CSV
features_to_analyze = []
vb_count = 6
t_count = 12

# Outer loop for VB numbers (1 to 6)
for vb_num in range(1, vb_count + 1):
    # Inner loop for T numbers (1 to 12)
    for t_num in range(1, t_count + 1):
        # Create the feature string using an f-string
        feature_name = f"VB{vb_num}_T{t_num}"
        features_to_analyze.append(feature_name)
# ============================================================================


def gpu_pearson_corr(series1, series2):
    """GPU-accelerated Pearson correlation using CuPy."""
    try:
        x = cp.asarray(series1)
        y = cp.asarray(series2)
        
        mask = ~(cp.isnan(x) | cp.isnan(y))
        x = x[mask]
        y = y[mask]
        
        if len(x) < 2:
            return np.nan, 0
        
        corr_matrix = cp.corrcoef(x, y)
        corr = float(corr_matrix[0, 1])
        return corr, int(cp.sum(mask))
    except Exception as e:
        print(f"Error in gpu_pearson_corr: {e}")
        return np.nan, 0


def load_and_build_matrix(file_path, blocksize='256MB'):
    """
    Load specified features (e.g., 'PV1_T121') and build a correlation matrix.
    """
    
    print(f"Loading data from: {file_path}")
    print(f"Blocksize: {blocksize}")
    print("=" * 100)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"ERROR: File '{file_path}' does not exist!")
        return None
    
    try:
        # First, read a small sample to get column names
        print("Reading CSV header to identify columns...")
        sample_df = pd.read_csv(file_path, nrows=10)
        all_columns = sample_df.columns.tolist()
        
        # Filter features based on configuration
        if FEATURES_TO_ANALYZE:
            feature_cols = [f for f in FEATURES_TO_ANALYZE if f in all_columns]
            missing = [f for f in FEATURES_TO_ANALYZE if f not in all_columns]
            if missing:
                print(f"\n⚠ WARNING: These features were not found: {missing}")
        else:
            print("ERROR: No features specified. Please set FEATURES_TO_ANALYZE.")
            return None
        
        if not feature_cols:
            print("ERROR: No valid features found to analyze.")
            return None
            
        print(f"✓ Building correlation matrix for {len(feature_cols)} features:")
        for f in feature_cols:
            print(f"  - {f}")
        
        # Load only necessary columns using dask_cudf
        print("\nLoading data with dask_cudf (GPU accelerated, chunked processing)...")
        cols_to_load = feature_cols
        
        ddf = dask_cudf.read_csv(file_path, blocksize=blocksize, usecols=cols_to_load)
        
        print(f"✓ Data will be processed in chunks of {blocksize}")
        
        # Process data chunk by chunk
        print("Processing chunks and loading data...")
        
        # This will hold the raw (pre-calculated) data for each feature
        data_chunks_dict = {col: [] for col in feature_cols}
        
        total_rows = 0
        n_partitions = ddf.npartitions
        
        for part_idx in range(n_partitions):
            partition = ddf.get_partition(part_idx).compute()
            total_rows += len(partition)
            
            # --- SIMPLIFIED: Simply read the data for each feature ---
            for col in feature_cols:
                data_chunks_dict[col].append(partition[col].to_numpy())
            
            # Clear GPU memory for this partition
            del partition
            
            if (part_idx + 1) % 10 == 0:
                print(f"  Processed {part_idx + 1}/{n_partitions} partitions ({total_rows:,} rows)...")
        
        print(f"✓ Loaded {total_rows:,} rows total from {n_partitions} partitions")
        
        # Concatenate all chunks in CPU memory
        print("Concatenating data chunks...")
        
        # This dict now holds the full series for each 'PV1_T121' type feature
        data_arrays = {}
        for col in feature_cols:
            data_arrays[col] = np.concatenate(data_chunks_dict[col])
        
        # Clear the lists to free memory
        del data_chunks_dict
        
        # --- ANALYSIS LOOP: Build N x N Matrix ---
        print("\n" + "=" * 100)
        print(f"ANALYZING {len(feature_cols)}x{len(feature_cols)} CORRELATION MATRIX")
        print("=" * 100)
        
        # Create empty dataframe for results
        results_matrix = pd.DataFrame(index=feature_cols, columns=feature_cols, dtype=np.float16)
        
        for i, f1 in enumerate(feature_cols):
            print(f"  Processing row {i+1}/{len(feature_cols)}: {f1}")
            for j, f2 in enumerate(feature_cols):
                
                # Diagonal is always 1
                if i == j:
                    results_matrix.loc[f1, f2] = 1.0
                    continue
                
                # Matrix is symmetric, skip if already computed
                if not pd.isna(results_matrix.loc[f2, f1]):
                   results_matrix.loc[f1, f2] = results_matrix.loc[f2, f1]
                   continue
                
                # Get the data for both features
                data1 = data_arrays[f1]
                data2 = data_arrays[f2]
                
                # Use your existing GPU correlation function
                corr, n_valid = gpu_pearson_corr(data1, data2)
                
                if n_valid < 10:
                    print(f"    - WARNING: Only {n_valid} valid points for {f1} vs {f2}. Corr set to NaN.")
                    corr = np.nan
                    
                results_matrix.loc[f1, f2] = corr

        # --- SAVE AND DISPLAY RESULTS ---
        
        # Save results
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save matrix as CSV
        csv_file = os.path.join(OUTPUT_DIR, f'correlation_matrix_{timestamp}.csv')
        results_matrix.to_csv(csv_file)
        
        print("\n" + "=" * 100)
        print("CORRELATION MATRIX")
        print("=" * 100)
        
        # Print the formatted matrix to console
        with pd.option_context('display.max_rows', None, 
                               'display.max_columns', None, 
                               'display.width', 200,
                               'display.float_format', '{:,.4f}'.format):
            print(results_matrix)
        
        print("\n" + "=" * 100)
        print(f"✓ Matrix saved to:")
        print(f"  CSV: {csv_file}")
        print("=" * 100)
        
        return results_matrix
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("\n" + "=" * 100)
    print(f"GPU-ACCELERATED {len(FEATURES_TO_ANALYZE)}x{len(FEATURES_TO_ANALYZE)} CORRELATION MATRIX ANALYSIS")
    print("=" * 100)
    print(f"Data file: {DATA_FILE}")
    print(f"Chunk size: {BLOCKSIZE}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    print("\nMode: Analyzing specified features:")
    for f in FEATURES_TO_ANALYZE:
        print(f"  - {f}")
    
    print("=" * 100 + "\n")
    
    results = load_and_build_matrix(DATA_FILE, BLOCKSIZE)