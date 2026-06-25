
import pandas as pd
import numpy as np
import glob
import os
import argparse
import warnings
import torch

from tabfairgan import TFG

warnings.filterwarnings('ignore')

# Dataset Configuration
DATA_DIR = "data"
OUTPUT_DIR = "baseline_data/tabfairgan"

# Configuration for fairness
# S: Sensitive Attribute, Y: Target Attribute
# S_under: Underprivileged group value (must be string)
# Y_desire: Desired outcome value (must be string)
DATASET_CONFIG = {
    'adult.csv': {
        'fairness_config': {
            'fair_epochs': 10, # Reduced from 50 for reasonable runtime in demo, increase for production
            'lamda': 0.5,
            'S': 'gender',
            'Y': 'class-label',
            'S_under': 'Female',
            'Y_desire': '1'
        }
    },
    'german-credit-data.csv': {
        'fairness_config': {
            'fair_epochs': 10,
            'lamda': 0.5,
            'S': 'sex',
            'Y': 'class-label',
            'S_under': 'female',
            'Y_desire': '1'
        }
    },
    'credit-scoring.csv': {
        'fairness_config': {
             'fair_epochs': 10,
             'lamda': 0.5,
             'S': 'Sex',
             'Y': 'class-label',
             'S_under': '2',
             'Y_desire': '1'
        }
    }
}

def generate_data_tabfairgan(csv_path, output_path, n_samples=1000, epochs=10, batch_size=256, device_arg=None):
    print(f"Processing {csv_path}...")
    filename = os.path.basename(csv_path)
    config = DATASET_CONFIG.get(filename, {'fairness_config': None})
    fair_config = config['fairness_config']

    try:
        # 1. Load Data
        df = pd.read_csv(csv_path, na_values=["?", "nan", "NaN", " ", ""])
        
        # Simple mode imputation
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna("MISSING")
            else:
                df[col] = df[col].fillna(df[col].mean())

        # 2. Pre-process for Fairness (Ensure Strings)
        if fair_config:
            S_col = fair_config['S']
            Y_col = fair_config['Y']
            
            # Ensure columns exist
            if S_col not in df.columns:
                print(f"  Warning: Sensitive column {S_col} not found. Skipping fairness.")
                fair_config = None
            elif Y_col not in df.columns:
                print(f"  Warning: Target column {Y_col} not found. Skipping fairness.")
                fair_config = None
            else:
                # TabFairGAN requires S and Y to be strings for OHE and fairness logic matches
                df[S_col] = df[S_col].astype(str)
                df[Y_col] = df[Y_col].astype(str)
                
                # Dynamic adjustment of fair_epochs
                original_fair_epochs = fair_config.get('fair_epochs', 0)
                if original_fair_epochs >= epochs:
                    print(f"  Adjusting fair_epochs ({original_fair_epochs}) to be less than total epochs ({epochs})...")
                    fair_config['fair_epochs'] = max(1, int(epochs * 0.5))
                
                print(f"  Fairness Enabled: {fair_config}")

        # 3. Initialize TabFairGAN
        # Check device
        if device_arg:
             device = device_arg
        elif torch.cuda.is_available():
            device = 'cuda:0'
        else:
            # Default to CPU because MPS (Metal) often causes NaNs/instability with GAN gradient penalties
            device = 'cpu'
        
        print(f"  Using device: {device}")
        
        tfg = TFG(
            df=df, 
            epochs=epochs, 
            batch_size=batch_size, 
            device=device,
            fairness_config=fair_config
        )
        
        # 4. Train
        print(f"  Training for {epochs} epochs...")
        tfg.train()
        
        # 5. Generate
        print(f"  Generating {n_samples} samples...")
        synthetic_data = tfg.generate_fake_df(num_rows=n_samples)
        
        # 6. Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        synthetic_data.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")

    except Exception as e:
        print(f"  Failed to process {csv_path}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data using TabFairGAN.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.") 
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--dataset", type=str, default=None, help="Specific dataset filename to process (e.g. adult.csv)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda, mps). If None, auto-detects.")
    
    args = parser.parse_args()

    # Iterate over known datasets in data/
    if args.dataset:
        csv_files = [os.path.join(DATA_DIR, args.dataset)]
    else:
        csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    print(f"Found {len(csv_files)} datasets in {DATA_DIR}")
    
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        generate_data_tabfairgan(
            csv_path, 
            output_path, 
            n_samples=args.samples, 
            epochs=args.epochs, 
            batch_size=args.batch_size,
            device_arg=args.device
        )

if __name__ == "__main__":
    main()
