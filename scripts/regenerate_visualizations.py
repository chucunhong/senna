#!/usr/bin/env python3
"""
Regenerate all visualizations using the best trained model.

This script loads the best model from models/{dataset}/ and regenerates
all visualizations without re-training.

Usage:
    python regenerate_visualizations.py --dataset german-credit-data
    python regenerate_visualizations.py --dataset adult --samples 100
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import sys
import glob
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from components.models import CD_Model
from components.dataset import TextDataset
from components.tokenization import compute_and_cache_tokenized_data
from components.utils import get_device
from src.data_loader import load_dataset
from src.visualization import (
    visualize_reconstruction_and_attention,
    visualize_batch_analysis,
    visualize_attention_heatmap_batch,
    analyze_attention_difference,
    visualize_latent_space_pca,
    visualize_attention_on_text,
    visualize_attention_comparison
)

device = get_device()

print(f"Using device: {device}")


def load_best_model(dataset_name: str, model_path: str = None):
    """
    Load the best trained model for a dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'german-credit-data')
        model_path: Optional specific model path. If None, loads the latest best model.
    
    Returns:
        Tuple of (model, checkpoint)
    """
    if model_path is None:
        model_dir = f"models/{dataset_name}"
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        model_files = glob.glob(os.path.join(model_dir, "best_model_*.pth"))
        if not model_files:
            raise FileNotFoundError(f"No best_model_*.pth files found in {model_dir}")
        
        model_path = max(model_files, key=os.path.getmtime)
    
    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract model parameters from checkpoint
    num_classes = checkpoint.get('num_classes', 2)
    latent_dim = checkpoint.get('latent_dim', 32)
    d_model = checkpoint.get('d_model', 768)
    
    # Create model and load state
    model = CD_Model(num_classes=num_classes, d_model=d_model, latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.4f}" if checkpoint.get('loss') else "  Loss: N/A")
    print(f"  Acc(z_c→y): {checkpoint.get('acc_z_c', 'N/A'):.3f}" if checkpoint.get('acc_z_c') else "  Acc(z_c→y): N/A")
    print(f"  Acc(z_d→y): {checkpoint.get('acc_z_d', 'N/A'):.3f}" if checkpoint.get('acc_z_d') else "  Acc(z_d→y): N/A")
    
    return model, checkpoint


def regenerate_all_visualizations(
    dataset_name: str,
    model_path: str = None,
    n_samples: int = 50,
    output_dir: str = None
):
    """
    Regenerate all visualizations using the best model.
    
    Args:
        dataset_name: Name of the dataset
        model_path: Optional specific model path
        n_samples: Number of samples for detailed visualizations
        output_dir: Output directory (default: visualizations/{dataset_name}/)
    """
    print(f"\n{'='*70}")
    print(f"REGENERATING VISUALIZATIONS")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}\n")
    
    # Load model
    model, checkpoint = load_best_model(dataset_name, model_path)
    
    # Load data from processed_data/{dataset_name}.csv
    print(f"\nLoading data from processed_data/{dataset_name}.csv...")
    sentences, labels, df, label_mapping = load_dataset(dataset_name)
    print(f"✓ Loaded {len(sentences)} samples")
    print(f"  Label distribution: {sum(labels)} positive, {len(labels) - sum(labels)} negative")
    
    # Setup tokenizer and cache
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    cached_data = compute_and_cache_tokenized_data(sentences, dataset_name, max_len=256)
    
    # Create datasets
    full_dataset = TextDataset(sentences, labels, tokenizer, max_len=256, cached_data=cached_data)
    vis_dataset = TextDataset(sentences[:n_samples], labels[:n_samples], tokenizer, max_len=256, cached_data=cached_data)
    
    # Setup output directory
    if output_dir is None:
        output_dir = f"visualizations/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # ====== GENERATE ALL VISUALIZATIONS ======
    
    # 1. Single sample visualizations (reconstruction + attention)
    print(f"\n{'='*50}")
    print(f"1. Generating single sample visualizations...")
    print(f"{'='*50}")
    single_loader = DataLoader(vis_dataset, batch_size=1, shuffle=False)
    for i, batch in enumerate(single_loader):
        if i >= min(5, n_samples):  # Limit to 5 samples
            break
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sentence = batch['text'][0]
        visualize_reconstruction_and_attention(model, input_ids, attention_mask, sentence, save_dir=output_dir)
    
    # 2. Batch analysis
    print(f"\n{'='*50}")
    print(f"2. Generating batch analysis visualization...")
    print(f"{'='*50}")
    vis_loader = DataLoader(vis_dataset, batch_size=8, shuffle=False)
    visualize_batch_analysis(model, vis_loader, num_samples=n_samples, save_dir=output_dir)
    
    # 3. Attention heatmaps
    print(f"\n{'='*50}")
    print(f"3. Generating unified attention heatmaps...")
    print(f"{'='*50}")
    visualize_attention_heatmap_batch(model, vis_loader, num_samples=n_samples, save_dir=output_dir)
    
    # 4. Attention separation analysis
    print(f"\n{'='*50}")
    print(f"4. Analyzing attention separation quality...")
    print(f"{'='*50}")
    analysis_loader = DataLoader(vis_dataset, batch_size=16, shuffle=False)
    stats = analyze_attention_difference(model, analysis_loader, num_samples=n_samples, save_dir=output_dir)
    
    # 5. Latent space PCA (MAIN VISUALIZATION)
    print(f"\n{'='*50}")
    print(f"5. Visualizing latent space with PCA (FULL DATASET)...")
    print(f"{'='*50}")
    full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)
    latent_stats = visualize_latent_space_pca(model, full_loader, num_samples=None, save_dir=output_dir)
    
    # 6. Attention on text (HTML)
    print(f"\n{'='*50}")
    print(f"6. Generating attention visualization on text (HTML)...")
    print(f"{'='*50}")
    visualize_attention_on_text(model, vis_loader, num_samples=n_samples, save_dir=output_dir)
    
    # 7. Attention comparison bar charts
    print(f"\n{'='*50}")
    print(f"7. Generating attention comparison visualizations...")
    print(f"{'='*50}")
    visualize_attention_comparison(model, vis_loader, num_samples=min(10, n_samples), save_dir=output_dir)
    
    print(f"\n{'='*70}")
    print(f"✓ ALL VISUALIZATIONS REGENERATED SUCCESSFULLY!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate all visualizations using best model")
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="german-credit-data",
        help="Dataset name (default: german-credit-data)"
    )
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        default=None,
        help="Specific model path (default: latest best_model_*.pth)"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=50,
        help="Number of samples for detailed visualizations (default: 50)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory (default: visualizations/{dataset}/)"
    )
    
    args = parser.parse_args()
    
    regenerate_all_visualizations(
        dataset_name=args.dataset,
        model_path=args.model_path,
        n_samples=args.samples,
        output_dir=args.output_dir
    )
