
import argparse
import json
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.validate import visualize_synthetic_data_pca

def main():
    parser = argparse.ArgumentParser(description="Visualize existing synthetic data (PCA)")
    parser.add_argument("--file", type=str, required=True, help="Path to synthetic JSON file")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    print(f"Loading synthetic data from {args.file}...")
    with open(args.file, 'r') as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} samples.")
    
    # Extract sentences and labels
    sentences = [entry['sentence'] for entry in data]
    labels = []
    
    for i, entry in enumerate(data):
        # different formats depending on how it was saved
        label = entry.get('class-label') or entry.get('label')
        if label is None:
            print(f"Warning: Sample {i} has no label. Skipping.")
            continue
        labels.append(int(label))
        
    if len(labels) < len(sentences):
        print("Error: Could not extract labels for all samples.")
        return

    # Create visualization directory
    output_dir = os.path.dirname(args.file)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"Generating visualization in {vis_dir}...")
    
    visualize_synthetic_data_pca(
        synthetic_sentences=sentences,
        synthetic_labels=labels,
        dataset_name=args.dataset,
        model_path=args.model_path,
        save_dir=vis_dir,
        compare_with_original=True
    )
    
    print("Done!")

if __name__ == "__main__":
    main()
