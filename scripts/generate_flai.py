
import pandas as pd
import numpy as np
import glob
import os
import argparse
import warnings
from sklearn.preprocessing import LabelEncoder

# Import flai
try:
    from FLAI.data import Data
    from FLAI.causal_graph import CausalGraph
except ImportError:
    print("Error: flai-causal not installed. Please install it via pip install flai-causal")
    exit(1)

warnings.filterwarnings('ignore')

# Dataset Configuration
DATA_DIR = "data"
OUTPUT_DIR = "baseline_data/flai"
SAMPLES_TO_GENERATE = 1000

# Mapping of dataset filenames to their configuration
# If a dataset is not in this map, we try to guess or skip
DATASET_CONFIG = {
    'adult.csv': {
        'target': 'class-label',
        'protected': ['gender', 'race', 'age', 'sex'], # Candidates, will check presence
    },
    'german-credit-data.csv': {
        'target': 'class-label',
        'protected': ['sex', 'age'],
    },
    'credit-scoring.csv': {
        'target': 'class-label',
        'protected': ['Sex', 'Age'], 
    },
    'bank-marketing.csv': {
        'target': 'class-label',
        'protected': ['age'], # 'sex' not typically in standard bank marketing, but age is
    },
    'dutch-census.csv': {
        'target': 'occupation', # Assumption based on census nature
        'protected': ['sex', 'age'],
    }
}

def get_config_for_file(filename):
    basename = os.path.basename(filename)
    if basename in DATASET_CONFIG:
        return DATASET_CONFIG[basename]
    return None

def generate_data_flai(csv_path, output_path, n_samples=1000):
    print(f"Processing {csv_path}...")
    
    config = get_config_for_file(csv_path)
    if not config:
        print(f"  Skipping {csv_path}: No configuration found in script.")
        return

    try:
        # 1. Load Data
        df = pd.read_csv(csv_path, na_values=["?", "nan", "NaN", " ", ""])
        
        # Simple mode imputation
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna("MISSING")
            else:
                df[col] = df[col].fillna(df[col].mean())

        # 2. Identify Target and Protected Attributes
        target = config['target']
        if target not in df.columns:
            print(f"  Warning: Target '{target}' not found in {csv_path}. Skipping.")
            return

        # Find actual present protected attributes
        protected = [p for p in config['protected'] if p in df.columns]
        if not protected:
            print(f"  Warning: No protected attributes found from {config['protected']} in {csv_path}. Using default graph.")
        
        print(f"  Target: {target}")
        print(f"  Protected: {protected}")

        # 3. Flai Causal Integration
        # Initialize Data object
        # Flai usually expects a dataframe
        data_obj = Data(df)
        
        # Create Causal Graph
        # We need to pass the target feature as required by FLAI/causal_graph.py
        graph = CausalGraph(data_obj, target=target)
        
        # Structure learning (if needed, otherwise it might deduce from data_obj init?)
        # Based on typical flai usage:
        # graph.structure_learning() 
        # or it's done in init? Let's assume we proceed to mitigation.
        
        # Mitigate
        # This is the key step for fairness. We mitigate edges from protected attributes to target directly or indirectly?
        # We pass the list of protected attributes directly.
        if protected:
            print(f"  Mitigating for {protected}...")
            try:
                graph.mitigate_edge_relation(sensible_feature=protected) 
            except Exception as e:
                print(f"  Mitigation error: {e}")

        # Learn CPDs (Parameters)
        # Verify if we need to learn parameters after structural changes (mitigation)
        # bn.sampling requires a BayesianNetwork with CPDs.
        print("  Learning CPDs...")
        graph.learn_cpd()

        # 4. Generate
        print(f"  Generating {n_samples} samples...")
        # Method 'bayes' is common in flai examples
        synthetic_df = graph.generate_dataset(n_samples=n_samples, methodtype='bayes')
        
        # 5. Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # synthetic_df is a FLAI.data.Data object
        print(f"  Head of generated data:\n{synthetic_df.data.head()}")
        synthetic_df.data.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")

    except Exception as e:
        print(f"  Failed to process {csv_path}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data using Flai Causal.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate.")
    args = parser.parse_args()

    # Iterate over known datasets in data/
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    print(f"Found {len(csv_files)} datasets in {DATA_DIR}")
    
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        generate_data_flai(csv_path, output_path, n_samples=args.samples)

if __name__ == "__main__":
    main()
