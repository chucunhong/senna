
import sys
import os
import glob
import pandas as pd
import numpy as np
import time
import json
import argparse
from copy import deepcopy
import warnings

# Add TabGEN-ICL to path
sys.path.append(os.path.abspath("TabGEN-ICL"))
# Add TabGEN-ICL/src_llm to path as well to ensure imports work if they rely on relative paths inside the module
sys.path.append(os.path.abspath("TabGEN-ICL/src_llm"))

from src_llm.llm_gen import llm_gen
from src_llm.dataset_llm_templates import markdown_templates_RES_RAG, json_templates_RES_RAG, create_json_model
from src_llm.utils import sample_and_split

if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "data"
OUTPUT_DIR = "baseline_data/tabgen"
API_KEY = os.getenv("OPEN_ROUTER_API_KEY") 
API_BASE = "https://openrouter.ai/api/v1"
MODEL = "meta-llama/llama-3.1-8b-instruct" 
# MODEL = "mistralai/mistral-7b-instruct" # Alternative

DATASET_MAPPING = {
    'adult.csv': 'adult', 
    'german-credit-data.csv': 'german-credit-data',
    'credit-scoring.csv': 'credit-scoring',
    'bank-marketing.csv': 'bank-marketing',
    'dutch-census.csv': 'dutch-census'
}

YLABEL_MAP = {
    "adult": "class-label",
    "german-credit-data": "class-label",
    "credit-scoring": "class-label",
    "bank-marketing": "class-label",
    "dutch-census": "occupation"
}

def generate_data_tabgen(csv_path, output_path, n_samples=1000, n_processes=8, chunk_size=100):
    filename = os.path.basename(csv_path)
    dataset_name = DATASET_MAPPING.get(filename, filename.replace('.csv', ''))
    
    print(f"Processing {dataset_name} from {csv_path}...")
    
    # Setup directories as expected by TabGEN-ICL (synthetic/{name}/real.csv)
    synth_dir = f"TabGEN-ICL/synthetic/{dataset_name}"
    os.makedirs(synth_dir, exist_ok=True)
    real_csv_path = f"{synth_dir}/real.csv"
    
    # Load and save real data
    df = pd.read_csv(csv_path, na_values=["?", "nan", "NaN", " ", ""])
    
    # Basic cleaning before saving
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("MISSING")
        else:
            df[col] = df[col].fillna(df[col].mean())
            
    df.to_csv(real_csv_path, index=False)
    
    ylabel = YLABEL_MAP.get(dataset_name)
    if not ylabel:
        print(f"  Target column not defined for {dataset_name}. Skipping.")
        return

    try:
        # Load data using TabGEN logic (partially)
        df_train = pd.read_csv(real_csv_path)
        
        df_feat_train = df_train.drop(columns=[ylabel])
        df_label_train = df_train[ylabel]

        # Use all data for setup? TabGEN samples internally.
        # But sample_and_split utility handles splitting.
        # We'll use 100% data as "train" effectively for generation context source.
        X_train_orig = df_feat_train
        y_train_orig = df_label_train
        
        # Prepare templates
        # We will use JSON format as it is structured
        output_format = 'json'
        # generator_template = json_templates_RES_RAG(role='generator') # Doesn't take args in current implementation?
        # Check src_llm/dataset_llm_templates.py again. json_templates_RES_RAG takes no args.
        # But wait, create_json_model needs df.
        
        # Custom prompt to control sample count
        generator_template = """
    You are a synthetic data generator tasked with creating new tabular data samples that closely mirror the distribution and characteristics of the original dataset.

    Instructions:
    1. Analyze the provided real samples carefully.
    2. Generate 10 new synthetic data samples that maintain the statistical properties of the real data.
    3. Ensure all attributes cover their full expected ranges.
    4. Output MUST be a valid JSON object with a key "JSON".

    Real samples:
    {data}

    {format_instructions}
    """
        
        # format_instructions = '{"JSON":[{col1:value1,col2:value2, ...}, {col1:value1,col2:value2, ...}, ...]}'
        # Better instruction for JSON mode
        format_instructions = """
The output must be a valid JSON object with a single key "JSON" containing a list of 10 objects.
Example:
{
  "JSON": [
    {"col1": "val1", "col2": 1},
    ...
  ]
}
"""

        df_orig = pd.concat([X_train_orig, y_train_orig], axis=1)
        json_response_schema = create_json_model(df_orig, dataname=dataset_name)
        
        example_df = df_orig.sample(frac=1).reset_index(drop=True)
        # Reduce context samples to save tokens/cost
        ic_samples = min(len(X_train_orig), 20)
        
        print(f"  Generating {n_samples} samples using {MODEL} (Context: {ic_samples} samples)...")
        
        df_llm = llm_gen(
            dataname=dataset_name,
            generator_template=generator_template,
            openai_key=API_KEY,
            n_samples=n_samples,
            example_df=example_df,
            output_format=output_format,
            response_schema=json_response_schema,
            format_instructions=format_instructions,
            model=MODEL,
            temperature=1.0, # Diversity
            max_tokens=4000, # Increased limit
            n_processes=n_processes,
            ic_samples=ic_samples,
            api_base=API_BASE
        )
        
        if df_llm is None or df_llm.empty:
             print("  Generation returned empty DataFrame.")
             return

        # Basic cleanup matches original types
        try:
             df_llm = df_llm.astype(df_orig.dtypes)
        except Exception as e:
             print(f"  Type casting failed: {e}. Attempting standard cleanup.")
             # Fallback cleanup
             for col in df_orig.columns:
                 if col in df_llm.columns:
                     if df_orig[col].dtype == 'object':
                         df_llm[col] = df_llm[col].astype(str)
                     else:
                         df_llm[col] = pd.to_numeric(df_llm[col], errors='coerce')

        # Save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_llm.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")

    except Exception as e:
        print(f"  Failed to process {dataset_name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data using TabGEN-ICL.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate.")
    parser.add_argument("--processes", type=int, default=8, help="Number of parallel processes.")
    args = parser.parse_args()

    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"Found {len(csv_files)} datasets in {DATA_DIR}")
    
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        generate_data_tabgen(
            csv_path, 
            output_path, 
            n_samples=args.samples,
            n_processes=args.processes
        )

if __name__ == "__main__":
    # Check key
    if not API_KEY:
        print("Error: OPEN_ROUTER_API_KEY not found in .env")
        sys.exit(1)
    main()
