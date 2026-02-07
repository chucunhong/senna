import pandas as pd
import numpy as np
from ctgan import CTGAN
import glob
import os
import argparse
import warnings

warnings.filterwarnings('ignore')

# Dataset Configuration
# Map simple names to filenames if needed, or just scan data folder
DATA_DIR = "data"
OUTPUT_DIR = "baseline_data/ctgan"
SAMPLES_TO_GENERATE = 1000

def get_discrete_columns(df):
    """
    Identify discrete/categorical columns for CTGAN.
    CTGAN requires specifying discrete columns explicitly.
    """
    discrete_columns = []
    for col in df.columns:
        # Check if object or categorical
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            discrete_columns.append(col)
        # Check if integer but low cardinality (potential categorical)
        elif pd.api.types.is_integer_dtype(df[col]):
            if df[col].nunique() < 20: # Heuristic: <20 unique values -> discrete
                discrete_columns.append(col)
    return discrete_columns

def generate_data(csv_path, output_path, epochs=300):
    print(f"Processing {csv_path}...")
    try:
        # Load Data
        df = pd.read_csv(csv_path, na_values=["?", "nan", "NaN", " ", ""])
        # Simple mode imputation for training stability
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna("MISSING")
            else:
                df[col] = df[col].fillna(df[col].mean())

        # Identify discrete columns
        discrete_cols = get_discrete_columns(df)
        print(f"  Discrete columns: {discrete_cols}")
        
        # Initialize CTGAN
        model = CTGAN(epochs=epochs, verbose=True)
        
        # Train
        print(f"  Training CTGAN on {len(df)} samples...")
        model.fit(df, discrete_columns=discrete_cols)
        
        # Generate
        print(f"  Generating {SAMPLES_TO_GENERATE} samples...")
        synthetic_data = model.sample(SAMPLES_TO_GENERATE)
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        synthetic_data.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")
        
    except Exception as e:
        print(f"  Failed to process {csv_path}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data using CTGAN.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    args = parser.parse_args()

    # Target Datasets
    # The user said "4 original datasets". 
    # We prioritize the ones we know are used in fairness evals
    target_files = [
        "adult.csv",
        "bank-marketing.csv" # 4th likely candidate
    ]
    
    # Also look for others if explicit ones are missing
    
    found_files = []
    for fname in target_files:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            found_files.append(path)
        else:
            # Try alternate names
            alt_path = os.path.join(DATA_DIR, fname.replace("-", "_"))
            if os.path.exists(alt_path):
                found_files.append(alt_path)

    if not found_files:
        print("No target datasets found in data/. creating generic list.")
        found_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

    print(f"Found datasets: {found_files}")

    for csv_path in found_files:
        filename = os.path.basename(csv_path)
        output_path = os.path.join(OUTPUT_DIR, filename)
        generate_data(csv_path, output_path, epochs=args.epochs)

if __name__ == "__main__":
    main()
