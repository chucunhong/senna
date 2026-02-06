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
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from components.models import CD_Model
from components.dataset import TextDataset
from components.tokenization import compute_and_cache_tokenized_data
from components.utils import get_device
from src.data_loader import load_dataset, load_sentence_template
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
    
    return model, checkpoint


def get_normalized_field_values(row, template_config):
    """
    Reconstruct the exact string values used in the sentence for each field.
    Mirrors logic from src/data_loader.py format_sentence_from_template.
    """
    def get_val(field_name):
        mapping = template_config['field_mapping'].get(field_name)
        if mapping is None:
            val = row.get(field_name)
            return str(val).strip() if pd.notna(val) else None
        
        if isinstance(mapping, list):
            for variant in mapping:
                val = row.get(variant)
                if pd.notna(val): return str(val).strip()
        else:
            val = row.get(mapping)
            if pd.notna(val): return str(val).strip()
        return None

    def normalize_value(value, field_name):
        if not value: return None
        value_str = str(value).strip().lower()
        
        # Logic from data_loader.py
        if 'people' in field_name.lower() or 'maintenance' in field_name.lower():
            try:
                num = int(float(value))
                return '1 dependent' if num == 1 else f'{num} dependents'
            except: pass
            
        if 'credit' in field_name.lower() and 'existing' in field_name.lower():
            try:
                num = int(float(value))
                return '1 existing credit' if num == 1 else f'{num} existing credits'
            except: pass
            
        try:
            float_val = float(value)
            if float_val == int(float_val): value = str(int(float_val))
            else: value = str(value).strip()
        except: value = str(value).strip()
        
        value_str = str(value).strip().lower()
        
        if value_str in ['yes', 'y', '1', 'true', '1.0']:
            if 'telephone' in field_name.lower(): return 'has'
            elif 'foreign' in field_name.lower(): return 'is a'
            elif 'changed-phone' in field_name.lower(): return 'yes' # credit-scoring
            return str(value).strip()
            
        if value_str in ['no', 'n', '0', 'false', 'none', '0.0']:
            if 'telephone' in field_name.lower(): return 'has no'
            elif 'foreign' in field_name.lower(): return 'is not a'
            elif 'debtor' in field_name.lower() and value_str == 'none': return 'no'
            elif 'installment' in field_name.lower() and value_str == 'none': return 'no'
            elif 'changed-phone' in field_name.lower(): return 'no' # credit-scoring
            return str(value).strip()
            
        return str(value).strip()

    value_transforms = template_config.get('value_transforms', {})
    field_values = {}
    
    # Process inputs
    for field_name in template_config['field_mapping'].keys():
        raw_value = get_val(field_name)
        if raw_value:
            if field_name in value_transforms:
                transforms = value_transforms[field_name]
                raw_lower = str(raw_value).strip().lower()
                if raw_lower in transforms:
                    field_values[field_name] = transforms[raw_lower]
                else:
                    field_values[field_name] = normalize_value(raw_value, field_name)
            else:
                field_values[field_name] = normalize_value(raw_value, field_name)
    
    return field_values


def build_template_regex(template_config):
    """
    Convert the sentence template into a regex pattern to extract exact field spans.
    """
    sentences = template_config['sentences'][:-1] # Skip label sentence
    full_template = " ".join(sentences)
    
    # Escape the template to treat static text as literals
    # We want to escape everything EXCEPT the {placeholders}
    
    # 1. Find all placeholders
    placeholders = re.findall(r'\{([a-zA-Z0-9-_]+)\}', full_template)
    
    # 2. Split by placeholders to get static parts
    # split regex: \{[a-zA-Z0-9-_]+\}
    static_parts = re.split(r'\{[a-zA-Z0-9-_]+\}', full_template)
    
    # 3. Build regex
    # We need to permit flexible whitespace where the template has spaces
    # And handle the placeholders
    
    regex_pattern = ""
    field_group_map = {} # map normalized group name to original field name or "Template: ..."
    
    for i, part in enumerate(static_parts):
        # Escape static text
        escaped_part = re.escape(part)
        # Allow flexible whitespace
        escaped_part = escaped_part.replace(r'\ ', r'\s+')
        
        # Add capture group for static text (Template part)
        static_group_name = f"static_{i}"
        regex_pattern += f"(?P<{static_group_name}>{escaped_part})"
        # We won't add to field_group_map here, we'll handle statics dynamically
        
        # If there is a placeholder following this part
        if i < len(placeholders):
            field_name = placeholders[i]
            # Create a valid group name
            group_name = f"field_{i}" 
            field_group_map[group_name] = field_name
            
            # Add capture group for the field
            regex_pattern += f"(?P<{group_name}>.*?)"
            
    # Add start/end anchors
    regex_pattern = r"^" + regex_pattern + r"$"
    
    return regex_pattern, field_group_map, placeholders


def analyze_field_attention(model, dataset_name, df, sentences, tokenizer, n_samples, output_dir):
    """
    Analyze attention weights on specific fields for z_c and z_d.
    Uses robust template-based regex extraction.
    """
    print(f"Analyzing attention on original data fields ({n_samples} samples)...")
    
    try:
        template_config = load_sentence_template(dataset_name)
    except Exception as e:
        print(f"Could not load template for {dataset_name}: {e}. Skipping field analysis.")
        return

    # Build regex extractor
    try:
        regex_pattern, field_group_map, placeholders = build_template_regex(template_config)
        # Relax the pattern slightly for robustness against punctuation variants
        # e.g. strictly ending with . vs . + space
        regex_pattern = regex_pattern.rstrip('$')
        print(f"Built template regex for extraction.")
        regex = re.compile(regex_pattern, re.IGNORECASE | re.DOTALL)
    except Exception as e:
        print(f"Failed to build regex: {e}. Falling back to approximation?")
        return

    # Store aggregated scores - Use defaultdict for dynamic keys
    from collections import defaultdict
    field_scores_c = defaultdict(list)
    field_scores_d = defaultdict(list)
    
    model.eval()
    
    count = 0
    match_failures = 0
    
    for i in range(min(len(df), n_samples)):
        sentence = sentences[i]
        
        # 2. Find spans using Regex
        match = regex.match(sentence)
        
        spans = {}
        if match:
            # Handle Fields
            for group_name, field_name in field_group_map.items():
                start, end = match.span(group_name)
                if end > start:
                    spans[field_name] = (start, end)
            
            # Handle Static Parts (Template words)
            # Iterate over all groups, check if static
            for name, value in match.groupdict().items():
                if name.startswith('static_') and value:
                    start_group, end_group = match.span(name)
                    # Split into words (tokens)
                    # We use regex to find words (alphanumeric) AND punctuation separately
                    # Pattern matches: Alphanumeric sequence OR non-whitespace non-alphanumeric char
                    for m_word in re.finditer(r'[a-zA-Z0-9]+|[^\s\w]', value):
                        word = m_word.group(0)
                        # Filter out punctuation only tokens if desired? 
                        # Keep everything for now as "tokens"
                        
                        word_start_abs = start_group + m_word.start()
                        word_end_abs = start_group + m_word.end()
                        
                        # Create a unique key for this word in this template position
                        # name is static_0, static_1...
                        # We explicitly want "T{i}: word"
                        idx = name.split('_')[1]
                        
                        # Clean word for key (remove some punctuation for display?)
                        # But keep full for uniqueness
                        
                        key = f"T{idx}: '{word}'"
                        spans[key] = (word_start_abs, word_end_abs)

        else:
            match_failures += 1
            if match_failures <= 5: 
                print(f"Regex match failed for sample {i}.")
            continue

        # 3. Tokenize and get attention
        inputs = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True, 
                          max_length=256, truncation=True, padding="max_length")
        
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        offset_mapping = inputs['offset_mapping'][0] # [seq_len, 2]
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            attn_c = outputs['attn_c'][0] # [seq_len]
            attn_d = outputs['attn_d'][0] # [seq_len]
            
        # 4. Map spans to tokens
        for field, (start_char, end_char) in spans.items():
            token_indices = []
            for t_idx, (t_start, t_end) in enumerate(offset_mapping):
                if t_start == t_end == 0: continue 
                
                # Check overlap
                overlap_start = max(t_start, start_char)
                overlap_end = min(t_end, end_char)
                
                if overlap_end > overlap_start:
                    token_indices.append(t_idx)
            
            if token_indices:
                score_c = attn_c[token_indices].sum().item()
                score_d = attn_d[token_indices].sum().item()
                
                field_scores_c[field].append(score_c)
                field_scores_d[field].append(score_d)
        
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} samples...")
            
    print(f"Successfully matched {count} samples. Failed matches: {match_failures}")

    # 5. Aggregate and Visualize
    print("\n aggregating results...")
    avg_scores_c = {}
    avg_scores_d = {}
    
    # Check all keys in field_group_map
    all_keys = list(field_scores_c.keys())
    
    for k in all_keys:
        if field_scores_c[k]:
            avg_scores_c[k] = np.mean(field_scores_c[k])
            avg_scores_d[k] = np.mean(field_scores_d[k])
        else:
            avg_scores_c[k] = 0.0
            avg_scores_d[k] = 0.0

    # --- Table 1: Top 5 Fields Only (Original Fields) ---
    fields_only = [k for k in all_keys if not k.startswith("T")] # Template keys start with T{i}
    
    # Sort for z_c
    top_c_fields = sorted(fields_only, key=lambda x: avg_scores_c[x], reverse=True)[:5]
    # Sort for z_d
    top_d_fields = sorted(fields_only, key=lambda x: avg_scores_d[x], reverse=True)[:5]
    
    print("\n" + "="*50)
    print("TABLE 1: Top 5 Original Data Fields")
    print("="*50)
    print(f"{'Rank':<5} {'z_c (Task Relevant)':<30} {'Score':<10} | {'z_d (Task Independent)':<30} {'Score':<10}")
    print("-" * 90)
    for i in range(5):
        c_field = top_c_fields[i] if i < len(top_c_fields) else "-"
        c_score = avg_scores_c.get(c_field, 0)
        d_field = top_d_fields[i] if i < len(top_d_fields) else "-"
        d_score = avg_scores_d.get(d_field, 0)
        print(f"{i+1:<5} {c_field:<30} {c_score:.4f}     | {d_field:<30} {d_score:.4f}")
    
    # Save Table 1 to CSV
    table1_data = []
    for i in range(5):
        c_field = top_c_fields[i] if i < len(top_c_fields) else None
        d_field = top_d_fields[i] if i < len(top_d_fields) else None
        if c_field or d_field:
            table1_data.append({
                'Rank': i+1,
                'z_c_Field': c_field,
                'z_c_Score': avg_scores_c.get(c_field, 0),
                'z_d_Field': d_field,
                'z_d_Score': avg_scores_d.get(d_field, 0)
            })
    pd.DataFrame(table1_data).to_csv(os.path.join(output_dir, "top5_fields.csv"), index=False)

    # --- Table 2: Top 5 All Tokens (Fields + Template) ---
    all_tokens = [k for k in all_keys if avg_scores_c[k] > 0 or avg_scores_d[k] > 0]
    
    # Sort for z_c
    top_c_all = sorted(all_tokens, key=lambda x: avg_scores_c[x], reverse=True)[:5]
    # Sort for z_d
    top_d_all = sorted(all_tokens, key=lambda x: avg_scores_d[x], reverse=True)[:5]
    
    print("\n" + "="*50)
    print("TABLE 2: Top 5 All Tokens (Fields + Template)")
    print("="*50)
    print(f"{'Rank':<5} {'z_c (Task Relevant)':<30} {'Score':<10} | {'z_d (Task Independent)':<30} {'Score':<10}")
    print("-" * 90)
    for i in range(5):
        c_item = top_c_all[i] if i < len(top_c_all) else "-"
        c_score = avg_scores_c.get(c_item, 0)
        d_item = top_d_all[i] if i < len(top_d_all) else "-"
        d_score = avg_scores_d.get(d_item, 0)
        
        # Truncate for display
        c_disp = (c_item[:27] + '..') if len(c_item) > 29 else c_item
        d_disp = (d_item[:27] + '..') if len(d_item) > 29 else d_item
        
        print(f"{i+1:<5} {c_disp:<30} {c_score:.4f}     | {d_disp:<30} {d_score:.4f}")

    # Save Table 2 to CSV
    table2_data = []
    for i in range(5):
        c_item = top_c_all[i] if i < len(top_c_all) else None
        d_item = top_d_all[i] if i < len(top_d_all) else None
        if c_item or d_item:
            table2_data.append({
                'Rank': i+1,
                'z_c_Item': c_item,
                'z_c_Score': avg_scores_c.get(c_item, 0),
                'z_d_Item': d_item,
                'z_d_Score': avg_scores_d.get(d_item, 0)
            })
    pd.DataFrame(table2_data).to_csv(os.path.join(output_dir, "top5_all_tokens.csv"), index=False)



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
    # print(f"\n{'='*50}")
    # print(f"1. Generating single sample visualizations...")
    # print(f"{'='*50}")
    # single_loader = DataLoader(vis_dataset, batch_size=1, shuffle=False)
    # for i, batch in enumerate(single_loader):
    #     if i >= min(5, n_samples):  # Limit to 5 samples
    #         break
    #     input_ids = batch['input_ids'].to(device)
    #     attention_mask = batch['attention_mask'].to(device)
    #     sentence = batch['text'][0]
    #     visualize_reconstruction_and_attention(model, input_ids, attention_mask, sentence, save_dir=output_dir)
    
    # 2. Batch analysis
    # print(f"\n{'='*50}")
    # print(f"2. Generating batch analysis visualization...")
    # print(f"{'='*50}")
    # vis_loader = DataLoader(vis_dataset, batch_size=8, shuffle=False)
    # visualize_batch_analysis(model, vis_loader, num_samples=n_samples, save_dir=output_dir)
    
    # 3. Attention heatmaps
    # print(f"\n{'='*50}")
    # print(f"3. Generating unified attention heatmaps...")
    # print(f"{'='*50}")
    # visualize_attention_heatmap_batch(model, vis_loader, num_samples=n_samples, save_dir=output_dir)
    
    # 4. Attention separation analysis
    # print(f"\n{'='*50}")
    # print(f"4. Analyzing attention separation quality...")
    # print(f"{'='*50}")
    # analysis_loader = DataLoader(vis_dataset, batch_size=16, shuffle=False)
    # stats = analyze_attention_difference(model, analysis_loader, num_samples=n_samples, save_dir=output_dir)
    
    # 5. Latent space PCA (MAIN VISUALIZATION)
    # print(f"\n{'='*50}")
    # print(f"5. Visualizing latent space with PCA (FULL DATASET)...")
    # print(f"{'='*50}")
    # full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)
    # latent_stats = visualize_latent_space_pca(model, full_loader, num_samples=None, save_dir=output_dir)
    
    # 6. Attention on text (HTML)
    # print(f"\n{'='*50}")
    # print(f"6. Generating attention visualization on text (HTML)...")
    # print(f"{'='*50}")
    # visualize_attention_on_text(model, vis_loader, num_samples=n_samples, save_dir=output_dir)
    
    # 7. Attention comparison bar charts
    # print(f"\n{'='*50}")
    # print(f"7. Generating attention comparison visualizations...")
    # print(f"{'='*50}")
    # visualize_attention_comparison(model, vis_loader, num_samples=min(10, n_samples), save_dir=output_dir)


    # 8. Feature Attention Analysis (NEW)
    print(f"\n{'='*50}")
    print(f"8. Analyzing attention on specific feature fields...")
    print(f"{'='*50}")
    # Use full dataset for stats if possible, or large subset
    analyze_field_attention(model, dataset_name, df, sentences, tokenizer, n_samples=len(df), output_dir=output_dir)
    
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
