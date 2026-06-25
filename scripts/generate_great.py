
import pandas as pd
import numpy as np
import glob
import os
import argparse
import warnings
import torch

# Import be_great
try:
    from be_great import GReaT
except ImportError:
    print("Error: be-great not installed. Please install it via pip install be-great")
    # We might continue if it's just a check, but here we exit
    # exit(1) # Commented out to allow script inspection if install fails

warnings.filterwarnings('ignore')

# Dataset Configuration
DATA_DIR = "data"
OUTPUT_DIR = "baseline_data/great"
SAMPLES_TO_GENERATE = 1000

# Mapping of dataset filenames to their configuration if needed
# GReaT is generic, so we might just process all CSVs in data/
DATASET_CONFIG = {
    'adult.csv': {'target': 'class-label'},
    'german-credit-data.csv': {'target': 'class-label'},
    'credit-scoring.csv': {'target': 'class-label'},
}

def generate_data_great(csv_path, output_path, n_samples=1000, epochs=5, batch_size=32, llm='gpt2'):
    print(f"Processing {csv_path}...")
    
    try:
        # 1. Load Data
        df = pd.read_csv(csv_path, na_values=["?", "nan", "NaN", " ", ""])
        
        # Simple mode imputation to avoid NaNs which might confuse GReaT string conversion
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna("MISSING")
            else:
                df[col] = df[col].fillna(df[col].mean())

        # 2. Initialize GReaT
        # Check for MPS (Mac) or CUDA
        device = 'cpu'
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        
        print(f"  Using device: {device}")
        print(f"  LLM: {llm}")
        
        
        from be_great import GReaT # Import here to ensure it's available
        
        # GReaT constructor does NOT take device. It takes it in sample (or implicitly for training via HF Accelerator).
        great = GReaT(llm=llm, batch_size=batch_size, epochs=epochs)
        
        # 3. Fit
        print(f"  Fitting GReaT on {len(df)} samples for {epochs} epochs...")
        great.fit(df)
        
        # 4. Generate
        print(f"  Generating {n_samples} samples using device {device}...")
        synthetic_data = great.sample(n_samples=n_samples, device=device)
        
        # 5. Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        synthetic_data.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")

    except Exception as e:
        print(f"  Failed to process {csv_path}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data using GReaT (GPT-2).")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.") # Reduced default for speed
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--llm", type=str, default='gpt2', help="HuggingFace model name (e.g. gpt2, distilgpt2).")
    
    args = parser.parse_args()

    # Iterate over known datasets in data/
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    print(f"Found {len(csv_files)} datasets in {DATA_DIR}")
    
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        generate_data_great(
            csv_path, 
            output_path, 
            n_samples=args.samples, 
            epochs=args.epochs, 
            batch_size=args.batch_size,
            llm=args.llm
        )

if __name__ == "__main__":
    main()
