
import sys
import os
import glob
import pandas as pd
import numpy as np
import time
from copy import deepcopy
from sklearn.model_selection import train_test_split
import warnings

# Add CLLM/src to path
sys.path.append(os.path.abspath("CLLM/src"))

from cllm.llm_gen import llm_gen
from cllm.curation import data_centric_curation

if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "data"
OUTPUT_DIR = "baseline_data/cllm"
API_KEY = os.getenv("OPEN_ROUTER_API_KEY") # Check .env variable name carefully

if not API_KEY:
    print("WARNING: OPENROUTER_API_KEY environment variable not set. Please set it before running.")
    # You might want to exit here or let it fail gracefully later

API_DETAILS = {
    "api_base": "https://openrouter.ai/api/v1",
    "api_key": API_KEY,
}

MODEL = "anthropic/claude-3.5-haiku" # Faster/Cheaper model
LLM_SERVING = "openrouter"

DATASET_TARGETS = {
    'adult.csv': 'class-label',
    'german-credit-data.csv': 'class-label',
    'credit-scoring.csv': 'class-label',
}

def generate_data_cllm(csv_path, output_path, n_samples=1000, n_processes=5):
    print(f"Processing {csv_path}...")
    filename = os.path.basename(csv_path)
    target_col = DATASET_TARGETS.get(filename)
    
    if not target_col:
        print(f"  Target column not defined for {filename}. Skipping.")
        return

    try:
        # 1. Load Data
        df = pd.read_csv(csv_path, na_values=["?", "nan", "NaN", " ", ""])
        # Fill NaNs specifically to avoid issues in LLM prompt or curation
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna("MISSING")
            else:
                df[col] = df[col].fillna(df[col].mean())

        # Prepare for CLLM
        # CLLM logic splits into train/remain, then samples from train for "Example DF"
        # We will use a subset of real data as "Example DF"
        
        # Split features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Create a small example set for the prompt (e.g., 20 samples)
        # We use a larger set for "training" the curation model (e.g. 50% of data)
        # Actually, standard CLLM flow:
        # 1. Take a small set of examples (ic_samples) from training data to put in prompt.
        # 2. Generate data.
        # 3. Use the original training data to train a curator (XGB) to filter the generated data.
        
        # 2. Setup Prompt
        # Manual format instructions to avoid langchain dependency issues
        format_instructions = """
The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":

```json
[
    {
        "column": "value",
        "column2": "value"
    }
]
```
Make sure to include all columns from the example data.
"""
        
        # Split logic fix
        train_size = int(len(df) * 0.8) if len(df) < 2000 else 2000
        train_df, _ = train_test_split(df, train_size=train_size, random_state=42)
        
        X_train_orig = train_df.drop(columns=[target_col])
        y_train_orig = train_df[target_col]
        
        example_df = train_df.copy() # passed to llm_gen, which samples `ic_samples` internally

        generator_template = f"""\
You are a synthetic data generator.
Your goal is to produce data which mirrors \
the given examples in causal structure and feature and label distributions \
but also produces as diverse samples as possible.

I will give you real examples first.

Leverage your knowledge to generate realistic but diverse samples for the dataset.

example data: {{data}}

{{format_instructions}}

DO NOT COPY THE EXAMPLES but generate realistic but new and diverse samples which have the correct label conditioned on the features.
"""
        
        # 3. Generate with LLM
        print(f"  Generating {n_samples} samples using {MODEL} via {LLM_SERVING}...")
        
        # llm_gen handles retries internally? No, main.py did.
        # I will implement simple retry logic around llm_gen if needed, but llm_gen loops/batches?
        # llm_gen loops up to 500 times until n_samples is reached.
        
        # Adjust n_processes based on API limits? OpenRouter is usually generous but let's stick to 5.
        
        df_llm = llm_gen(
            prompt=None, # llm_gen creates prompt internally if template provided
            generator_template=generator_template,
            format_instructions=format_instructions,
            example_df=example_df,
            llm_serving=LLM_SERVING,
            api_details=API_DETAILS,
            n_samples=n_samples,
            temperature=0.9, # High temp for diversity
            max_tokens=1000, # Reduced to avoid massive generation
            model=MODEL,
            n_processes=n_processes,
            ic_samples=min(10, len(example_df)) # Context examples
        )
        
        if df_llm is None or df_llm.empty:
            print("  Values returned were empty. Skipping.")
            return

        # 4. Post-processing (Cleaning)
        print("  Post-processing generated data...")
        # Basic cleanup: strict type enforcement
        # Reuse logic from main.py/llm_gen cleanliness
        
        # Drop rows with wrong columns or types if possible
        # Check against X_train_orig dtypes
        df_llm = df_llm.dropna()
        
        # Ensure columns match
        common_cols = [c for c in example_df.columns if c in df_llm.columns]
        df_llm = df_llm[common_cols]
        
        # Attempt type conversion
        for col in common_cols:
            try:
                df_llm[col] = df_llm[col].astype(example_df[col].dtype)
            except:
                # Fallback: coerce
                df_llm[col] = pd.to_numeric(df_llm[col], errors='ignore')

        # 5. Curation
        print("  Skipping Data Centric Curation to avoid segmentation fault on macOS.")
        # if target_col in df_llm.columns:
        #     X_check = df_llm.drop(columns=[target_col])
        #     y_check = df_llm[target_col]
            
        #     # Use X_train_orig for curator training
        #     # Ensure columns in X_check match X_train_orig
        #     # (llm might generate extra or fewer? we filtered common_cols)
        #     # Need to align columns
        #     X_check = X_check[X_train_orig.columns] 
            
        #     # Encoding for XGBoost (Curator)
        #     # Curator uses XGBoost. XGBoost handles numerical. Categorical needs encoding?
        #     # XGBoost can handle categorical if enabled, or we perform OHE.
        #     # curation.py uses `XGBClassifier(n_estimators=nest)`. Default XGB might not handle strings without `enable_categorical=True`.
        #     # Let's simple OHE for curation to be safe.
            
        #     # Combine for OHE to ensure same columns
        #     combined_X = pd.concat([X_train_orig, X_check], axis=0)
        #     combined_X_ohe = pd.get_dummies(combined_X)
            
        #     X_train_ohe = combined_X_ohe.iloc[:len(X_train_orig)]
        #     X_check_ohe = combined_X_ohe.iloc[len(X_train_orig):]
            
        #     # For y, simple encoding
        #     # Use factorize
        #     y_combined = pd.concat([y_train_orig, y_check], axis=0)
        #     y_combined_encoded, y_codes = pd.factorize(y_combined)
        #     y_train_enc = y_combined_encoded[:len(y_train_orig)]
        #     y_check_enc = y_combined_encoded[len(y_train_orig):]
            
            
        #     easy_train, ambig_train, unlearnable_train, Curator_xgb = data_centric_curation(
        #         X_train_orig=X_train_ohe,
        #         y_train_orig=y_train_enc,
        #         X_check=X_check_ohe,
        #         y_check=y_check_enc,
        #         curation_metric='aleatoric',
        #         retrain=False,
        #         nest=50, # Reduced nest for speed
        #         curation_ythresh=0.2,
        #         curation_xthresh=0 # adaptive
        #     )
            
        #     # Select curated samples (easy + ambiguous)
        #     # unlearnable are rejected?
        #     # main.py: curated_train_ids = np.concatenate((easy_train, ambig_train))
        #     curated_ids = np.concatenate((easy_train, ambig_train)).astype(int)
            
        #     print(f"  Curation kept {len(curated_ids)}/{len(df_llm)} samples.")
            
        #     if len(curated_ids) > 0:
        #         final_df = df_llm.iloc[curated_ids]
        #     else:
        #         print("  Warning: Curation rejected all samples. Saving all generated samples instead.")
        #         final_df = df_llm
        # else:
        #     print("  Target column missing in generated data. Skipping curation.")
        final_df = df_llm

        # 6. Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")

    except Exception as e:
        print(f"  Failed to process {csv_path}: {e}")
        import traceback
        traceback.print_exc()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic data using CLLM.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate.")
    parser.add_argument("--dataset", type=str, default="all", help="Specific dataset to process (e.g., 'adult'). Default is 'all'.")
    
    args = parser.parse_args()

    # Iterate over known datasets in data/
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    if args.dataset != "all":
        # Filter by filename (with or without extension)
        filtered_files = [f for f in csv_files if args.dataset == os.path.basename(f) or args.dataset == os.path.splitext(os.path.basename(f))[0]]
        if not filtered_files:
             print(f"Error: Dataset '{args.dataset}' not found.")
             return
        csv_files = filtered_files
    
    print(f"Found {len(csv_files)} datasets to process")
    
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        generate_data_cllm(
            csv_path, 
            output_path, 
            n_samples=args.samples
        )

if __name__ == "__main__":
    main()
