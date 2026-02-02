"""
Validation module for synthetic data generation.
For each generated sample:
1. Extract z_c and z_d using the trained model
2. Find k nearest neighbors based on z_c from original dataset
3. Check if the label matches the majority label of neighbors
4. Reject if label doesn't match majority
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer
import os
import glob
import hashlib
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import re
import pandas as pd

from components.models import CD_Model, device
from components.dataset import TextDataset
from data_loader import load_german_credit_data_balanced, load_dataset_generic
from components.tokenization import compute_and_cache_tokenized_data


def get_cache_path(dataset_name: str, model_path: str) -> str:
    """
    Generate cache file path for z_c and z_d encodings of original data.
    
    Args:
        dataset_name: Name of dataset
        model_path: Path to model checkpoint
    
    Returns:
        Path to cache file
    """
    # Create hash from model path to ensure cache is model-specific
    model_hash = hashlib.md5(model_path.encode()).hexdigest()[:8]
    cache_dir = f"cache/embeddings/{dataset_name}"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"z_c_z_d_{model_hash}.pkl")
    return cache_file


def load_cached_encodings(cache_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, List[str], List[int]]]:
    """
    Load cached z_c, z_d encodings and original data.
    
    Args:
        cache_path: Path to cache file
    
    Returns:
        Tuple of (z_c, z_d, sentences, labels) or None if cache doesn't exist
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            print(f"✓ Loaded cached encodings from: {cache_path}")
            return (
                cache_data['z_c'],
                cache_data['z_d'],
                cache_data['sentences'],
                cache_data['labels']
            )
        except Exception as e:
            print(f"⚠ Failed to load cache: {e}")
            return None
    return None


def save_cached_encodings(
    cache_path: str,
    z_c: np.ndarray,
    z_d: np.ndarray,
    sentences: List[str],
    labels: List[int]
):
    """
    Save z_c, z_d encodings and original data to cache.
    
    Args:
        cache_path: Path to cache file
        z_c: z_c encodings [n_samples, latent_dim]
        z_d: z_d encodings [n_samples, latent_dim]
        sentences: Original sentences
        labels: Original labels
    """
    try:
        cache_data = {
            'z_c': z_c,
            'z_d': z_d,
            'sentences': sentences,
            'labels': labels
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"✓ Saved encodings cache to: {cache_path}")
    except Exception as e:
        print(f"⚠ Failed to save cache: {e}")


def load_trained_model(dataset_name: str, model_path: Optional[str] = None) -> Tuple[CD_Model, Dict]:
    """
    Load trained model from checkpoint.
    
    Args:
        dataset_name: Name of dataset (e.g., 'german-credit-data', 'adult')
        model_path: Optional path to model checkpoint. If None, will find latest model in models/{dataset_name}/
    
    Returns:
        Tuple of (model, checkpoint_info)
    """
    if model_path is None:
        # Find latest model in models/{dataset_name}/
        model_dir = f"models/{dataset_name}"
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Find all .pth files
        model_files = glob.glob(os.path.join(model_dir, "best_model_*.pth"))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")
        
        # Get latest model (by modification time)
        model_path = max(model_files, key=os.path.getmtime)
        print(f"Found latest model: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model parameters from checkpoint
    latent_dim = checkpoint.get('latent_dim', 64)
    num_classes = checkpoint.get('num_classes', 2)
    d_model = checkpoint.get('d_model', 768)
    
    # Create model
    model = CD_Model(
        num_classes=num_classes,
        d_model=d_model,
        latent_dim=latent_dim,
        freeze_bert=False
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from {model_path}")
    print(f"  - Latent dim: {latent_dim}")
    print(f"  - Num classes: {num_classes}")
    print(f"  - D model: {d_model}")
    
    return model, checkpoint


def split_sentence_into_main_and_label(sentence: str, dataset_name: str) -> Tuple[str, str]:
    """
    Split sentence into main part (without label sentence) and label sentence.
    
    Args:
        sentence: Full sentence that may contain label description
        dataset_name: Name of dataset
    
    Returns:
        Tuple of (main_sentence, label_sentence)
        - main_sentence: Sentence without label description
        - label_sentence: Label sentence (empty string if not found)
    """
    if dataset_name == 'german-credit-data':
        # Pattern: "The credit application has been evaluated and the decision is ..."
        pattern = r'(.*?)(\s*The credit application has been evaluated and the decision is.*$)'
        match = re.search(pattern, sentence, re.IGNORECASE | re.DOTALL)
        if match:
            main_sentence = match.group(1).strip()
            label_sentence = match.group(2).strip()
            return main_sentence, label_sentence
    elif dataset_name == 'adult':
        # Pattern: "The income classification for this person is ..."
        pattern = r'(.*?)(\s*The income classification for this person is.*$)'
        match = re.search(pattern, sentence, re.IGNORECASE | re.DOTALL)
        if match:
            main_sentence = match.group(1).strip()
            label_sentence = match.group(2).strip()
            return main_sentence, label_sentence
    
    # If no label sentence found, return full sentence as main
    return sentence.strip(), ""


def extract_label_from_label_sentence(label_sentence: str, dataset_name: str) -> Optional[int]:
    """
    Extract label value from label sentence.
    
    Args:
        label_sentence: Label sentence (e.g., "The credit application has been evaluated and the decision is good credit risk.")
        dataset_name: Name of dataset
    
    Returns:
        Label value (0 or 1), or None if cannot extract
    """
    if not label_sentence:
        return None
    
    label_sentence_lower = label_sentence.lower()
    
    if dataset_name == 'german-credit-data':
        # Check for "good credit risk" or "bad credit risk"
        # Note: 0 = good credit risk, 1 = bad credit risk
        if 'good credit risk' in label_sentence_lower:
            return 0  # 0 = good credit risk
        elif 'bad credit risk' in label_sentence_lower:
            return 1  # 1 = bad credit risk
        # Also check for numeric patterns like "decision is 1" or "decision is 0"
        match = re.search(r'decision\s+is\s+(\d+)', label_sentence_lower)
        if match:
            return int(match.group(1))
    elif dataset_name == 'adult':
        # Check for "above 50K" or "below 50K"
        if 'above 50k' in label_sentence_lower or 'above 50 k' in label_sentence_lower:
            return 1
        elif 'below 50k' in label_sentence_lower or 'below 50 k' in label_sentence_lower:
            return 0
        # Also check for numeric patterns
        match = re.search(r'classification.*?is\s+(\d+)', label_sentence_lower)
        if match:
            return int(match.group(1))
    
    return None


def load_original_data_from_processed(dataset_name: str, processed_data_dir: str = "processed_data") -> Tuple[List[str], List[int]]:
    """
    Load original data from processed_data/{dataset_name}.csv.
    
    Args:
        dataset_name: Name of dataset
        processed_data_dir: Directory containing processed data
    
    Returns:
        Tuple of (sentences, labels)
    """
    import pandas as pd
    
    csv_path = os.path.join(processed_data_dir, f"{dataset_name}.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Processed data file not found: {csv_path}\n"
            f"Please ensure the file exists in {processed_data_dir}/ directory."
        )
    
    print(f"Loading original data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'sentence' not in df.columns:
        raise ValueError(f"CSV file must have 'sentence' column. Found columns: {df.columns.tolist()}")
    if 'label' not in df.columns:
        raise ValueError(f"CSV file must have 'label' column. Found columns: {df.columns.tolist()}")
    
    sentences = df['sentence'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()
    
    print(f"Loaded {len(sentences)} samples from processed data")
    print(f"Label distribution: {sum(labels)} (1), {len(labels) - sum(labels)} (0)")
    
    return sentences, labels


def remove_label_description_from_sentence(sentence: str, dataset_name: str) -> str:
    """
    Remove label description from sentence before encoding z_c and z_d.
    Label description should NOT be included when extracting embeddings.
    
    Args:
        sentence: Sentence that may contain label description
        dataset_name: Name of dataset
    
    Returns:
        Sentence without label description
    """
    if dataset_name == 'german-credit-data':
        # Remove patterns like "The credit application has been evaluated and the decision is ..."
        # Remove everything from "The credit application" onwards
        pattern = r'\s*The credit application has been evaluated and the decision is.*$'
        sentence = re.sub(pattern, '', sentence, flags=re.IGNORECASE)
        # Also remove if it's just "good credit risk" or "bad credit risk" at the end
        sentence = re.sub(r'\s*(good|bad)\s+credit\s+risk\.?\s*$', '', sentence, flags=re.IGNORECASE)
    elif dataset_name == 'adult':
        # Remove patterns like "The income classification for this person is ..."
        pattern = r'\s*The income classification for this person is.*$'
        sentence = re.sub(pattern, '', sentence, flags=re.IGNORECASE)
        # Also remove "above 50K" or "below 50K" at the end
        sentence = re.sub(r'\s*(above|below)\s+50K\.?\s*$', '', sentence, flags=re.IGNORECASE)
    
    # Clean up trailing spaces and punctuation
    sentence = sentence.strip()
    sentence = re.sub(r'\s+$', '', sentence)
    # Remove trailing period if it's the last character (but keep if sentence ends naturally)
    if sentence.endswith('.'):
        # Only remove if it's likely from label removal
        if not any(word in sentence.lower() for word in ['status', 'months', 'years', 'credits', 'dependents']):
            sentence = sentence[:-1].strip()
    
    return sentence


def add_label_description_to_sentence(sentence: str, label: int, dataset_name: str) -> str:
    """
    Add label description to sentence, matching the format used during training.
    
    Args:
        sentence: Original sentence without label
        label: Label value (0 or 1)
        dataset_name: Name of dataset
    
    Returns:
        Sentence with label description appended
    """
    if dataset_name == 'german-credit-data':
        if label == 1:
            label_desc = "good credit risk"
        else:
            label_desc = "bad credit risk"
        return f"{sentence} The credit application has been evaluated and the decision is {label_desc}."
    elif dataset_name == 'adult':
        if label == 1:
            label_desc = "above 50K"
        else:
            label_desc = "below 50K"
        return f"{sentence} The income classification for this person is {label_desc}."
    else:
        # Default format
        return f"{sentence} Label: {label}."


def check_duplicates(
    synthetic_sentences: List[str],
    original_sentences: List[str],
    existing_sentences: Optional[List[str]] = None
) -> Tuple[List[bool], Dict]:
    """
    Check for duplicate sentences:
    1. Check if any synthetic sentence matches an original sentence
    2. Check if any synthetic sentences are duplicates of each other
    
    Args:
        synthetic_sentences: List of generated sentences to check
        original_sentences: List of original dataset sentences
        existing_sentences: Optional list of previously generated sentences (from previous batches)
    
    Returns:
        Tuple of (is_unique_list, details_dict)
        - is_unique_list: List of booleans, True if sentence is unique
        - details_dict: Dictionary with duplicate information
    """
    is_unique = [True] * len(synthetic_sentences)
    details = {
        'duplicates_with_original': [],
        'duplicates_within_batch': [],
        'duplicates_with_existing': []
    }
    
    # Normalize sentences for comparison (lowercase, strip whitespace)
    normalized_original = {s.lower().strip(): i for i, s in enumerate(original_sentences)}
    normalized_existing = {}
    if existing_sentences:
        normalized_existing = {s.lower().strip(): i for i, s in enumerate(existing_sentences)}
    
    # Check each synthetic sentence
    seen_in_batch = {}
    
    for i, sentence in enumerate(synthetic_sentences):
        normalized = sentence.lower().strip()
        
        # Check against original dataset
        if normalized in normalized_original:
            is_unique[i] = False
            details['duplicates_with_original'].append({
                'index': i,
                'sentence': sentence[:100] + '...' if len(sentence) > 100 else sentence,
                'matches_original_index': normalized_original[normalized]
            })
        
        # Check against existing generated sentences (from previous batches)
        if normalized_existing and normalized in normalized_existing:
            is_unique[i] = False
            details['duplicates_with_existing'].append({
                'index': i,
                'sentence': sentence[:100] + '...' if len(sentence) > 100 else sentence,
                'matches_existing_index': normalized_existing[normalized]
            })
        
        # Check for duplicates within current batch
        if normalized in seen_in_batch:
            is_unique[i] = False
            is_unique[seen_in_batch[normalized]] = False  # Mark the first occurrence as duplicate too
            details['duplicates_within_batch'].append({
                'index': i,
                'sentence': sentence[:100] + '...' if len(sentence) > 100 else sentence,
                'duplicate_of_index': seen_in_batch[normalized]
            })
        else:
            seen_in_batch[normalized] = i
    
    return is_unique, details


def extract_z_c_z_d(
    model: CD_Model, 
    sentences: List[str], 
    tokenizer, 
    batch_size: int = 32,
    dataset_name: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract z_c and z_d encodings from sentences using trained model.
    
    IMPORTANT: Label description is REMOVED from sentences before encoding.
    This ensures z_c and z_d are learned without label information leakage.
    
    Args:
        model: Trained CD_Model
        sentences: List of sentences to encode (may contain label description, will be removed)
        tokenizer: BERT tokenizer
        batch_size: Batch size for processing
        dataset_name: Optional dataset name for removing label description
    
    Returns:
        Tuple of (z_c_array, z_d_array) where each is [n_samples, latent_dim]
    """
    # Remove label description from sentences before encoding
    # Label description should NOT be included when extracting z_c and z_d
    if dataset_name is not None:
        sentences_clean = []
        for sentence in sentences:
            sentence_clean = remove_label_description_from_sentence(sentence, dataset_name)
            sentences_clean.append(sentence_clean)
        sentences = sentences_clean
    
    model.eval()
    
    # Tokenize sentences
    # IMPORTANT: Use same tokenization as TextDataset (padding="max_length" not padding=True)
    # This ensures consistency with training dataloader
    encoded = tokenizer(
        sentences,
        padding="max_length",  # Same as TextDataset
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    z_c_list = []
    z_d_list = []
    
    with torch.no_grad():
        # Process in batches
        for i in range(0, len(sentences), batch_size):
            batch_input_ids = input_ids[i:i+batch_size]
            batch_attention_mask = attention_mask[i:i+batch_size]
            
            # Forward pass
            output = model(batch_input_ids, batch_attention_mask)
            
            z_c_batch = output['z_c'].cpu().numpy()  # [batch_size, latent_dim]
            z_d_batch = output['z_d'].cpu().numpy()  # [batch_size, latent_dim]
            
            z_c_list.append(z_c_batch)
            z_d_list.append(z_d_batch)
    
    z_c_array = np.vstack(z_c_list)  # [n_samples, latent_dim]
    z_d_array = np.vstack(z_d_list)  # [n_samples, latent_dim]
    
    return z_c_array, z_d_array


def find_k_nearest_neighbors(
    z_c_sample: np.ndarray,
    z_c_original: np.ndarray,
    k: int = 9,
    metric: str = 'cosine'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find k nearest neighbors by comparing z_c distances.
    
    Args:
        z_c_sample: z_c encoding of sample [latent_dim]
        z_c_original: z_c encodings from original dataset [n_original, latent_dim]
        k: Number of neighbors to find
        metric: Distance metric ('cosine' or 'euclidean')
    
    Returns:
        Tuple of (distances, indices) where:
        - distances: [k] distances to k nearest neighbors
        - indices: [k] indices of k nearest neighbors in original dataset
    """
    # Normalize for cosine distance
    if metric == 'cosine':
        # Normalize vectors
        z_c_sample_norm = z_c_sample / (np.linalg.norm(z_c_sample) + 1e-8)
        z_c_original_norm = z_c_original / (np.linalg.norm(z_c_original, axis=1, keepdims=True) + 1e-8)
        # Cosine distance = 1 - cosine similarity
        distances = 1 - np.dot(z_c_original_norm, z_c_sample_norm)
    else:  # euclidean
        distances = np.linalg.norm(z_c_original - z_c_sample, axis=1)
    
    # Find k nearest neighbors
    k_nearest_indices = np.argsort(distances)[:k]
    k_nearest_distances = distances[k_nearest_indices]
    
    return k_nearest_distances, k_nearest_indices


def check_duplicates(
    synthetic_sentences: List[str],
    original_sentences: List[str],
    existing_sentences: Optional[List[str]] = None
) -> Tuple[List[bool], Dict]:
    """
    Check for duplicate sentences:
    1. Check if any synthetic sentence matches an original sentence
    2. Check if any synthetic sentences are duplicates of each other
    
    Args:
        synthetic_sentences: List of generated sentences to check
        original_sentences: List of original dataset sentences
        existing_sentences: Optional list of previously generated sentences (from previous batches)
    
    Returns:
        Tuple of (is_unique_list, details_dict)
        - is_unique_list: List of booleans, True if sentence is unique
        - details_dict: Dictionary with duplicate information
    """
    is_unique = [True] * len(synthetic_sentences)
    details = {
        'duplicates_with_original': [],
        'duplicates_within_batch': [],
        'duplicates_with_existing': []
    }
    
    # Normalize sentences for comparison (lowercase, strip whitespace)
    normalized_original = {s.lower().strip(): i for i, s in enumerate(original_sentences)}
    normalized_existing = {}
    if existing_sentences:
        normalized_existing = {s.lower().strip(): i for i, s in enumerate(existing_sentences)}
    
    # Check each synthetic sentence
    seen_in_batch = {}
    
    for i, sentence in enumerate(synthetic_sentences):
        normalized = sentence.lower().strip()
        
        # Check against original dataset
        if normalized in normalized_original:
            is_unique[i] = False
            details['duplicates_with_original'].append({
                'index': i,
                'sentence': sentence[:100] + '...' if len(sentence) > 100 else sentence,
                'matches_original_index': normalized_original[normalized]
            })
        
        # Check against existing generated sentences (from previous batches)
        if normalized_existing and normalized in normalized_existing:
            is_unique[i] = False
            details['duplicates_with_existing'].append({
                'index': i,
                'sentence': sentence[:100] + '...' if len(sentence) > 100 else sentence,
                'matches_existing_index': normalized_existing[normalized]
            })
        
        # Check for duplicates within current batch
        if normalized in seen_in_batch:
            is_unique[i] = False
            is_unique[seen_in_batch[normalized]] = False  # Mark the first occurrence as duplicate too
            details['duplicates_within_batch'].append({
                'index': i,
                'sentence': sentence[:100] + '...' if len(sentence) > 100 else sentence,
                'duplicate_of_index': seen_in_batch[normalized]
            })
        else:
            seen_in_batch[normalized] = i
    
    return is_unique, details


def add_label_description_to_sentence(sentence: str, label: int, dataset_name: str) -> str:
    """
    Add label description to sentence, matching the format used during training.
    
    Args:
        sentence: Original sentence without label
        label: Label value (0 or 1)
        dataset_name: Name of dataset
    
    Returns:
        Sentence with label description appended
    """
    if dataset_name == 'german-credit-data':
        if label == 1:
            label_desc = "good credit risk"
        else:
            label_desc = "bad credit risk"
        return f"{sentence} The credit application has been evaluated and the decision is {label_desc}."
    elif dataset_name == 'adult':
        if label == 1:
            label_desc = "above 50K"
        else:
            label_desc = "below 50K"
        return f"{sentence} The income classification for this person is {label_desc}."
    else:
        # Default format
        return f"{sentence} Label: {label}."


def validate_sample(
    z_c_sample: np.ndarray,
    z_c_original: np.ndarray,
    original_labels: np.ndarray,
    k: int = 9,
    majority_threshold: float = 0.5,
    metric: str = 'cosine'
) -> Tuple[bool, int, float]:
    """
    Validate a single sample by checking if its label matches majority of k nearest neighbors.
    
    Args:
        z_c_sample: z_c encoding of sample [latent_dim]
        z_c_original: z_c encodings from original dataset [n_original, latent_dim]
        original_labels: Labels of original dataset [n_original]
        k: Number of neighbors to check
        majority_threshold: Minimum fraction of neighbors that must have same label (default: 0.5 = majority)
        metric: Distance metric ('cosine' or 'euclidean')
    
    Returns:
        Tuple of (is_valid, majority_label, majority_ratio)
        - is_valid: True if sample label matches majority label of neighbors
        - majority_label: Label of majority of neighbors (0 or 1)
        - majority_ratio: Fraction of neighbors with majority label
    """
    # Find k nearest neighbors by comparing distances
    distances, indices = find_k_nearest_neighbors(z_c_sample, z_c_original, k=k, metric=metric)
    
    # Get labels of neighbors
    neighbor_labels = original_labels[indices]  # [k]
    
    # Count labels
    unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
    majority_count = counts.max()
    majority_label = unique_labels[counts.argmax()]
    majority_ratio = majority_count / k
    
    # Check if majority ratio meets threshold
    # For tie cases (e.g., 4-5), we need at least > 50% to have a clear majority
    is_valid = majority_ratio > majority_threshold  # Use > instead of >= to handle ties better
    
    return is_valid, int(majority_label), majority_ratio


def validate_synthetic_data(
    synthetic_sentences: List[str],
    dataset_name: str,
    k: int = 9,
    majority_threshold: float = 0.5,
    model_path: Optional[str] = None,
    batch_size: int = 32,
    existing_sentences: Optional[List[str]] = None
) -> Dict:
    """
    Validate synthetic data by checking consistency with original dataset.
    
    New validation flow:
    1. Load original data from processed_data/{dataset_name}.csv (sentences without label sentence)
    2. Encode all original sentences (main parts only, no label sentence) and cache z_c, z_d
    3. Split synthetic sentences into main part (for encoding) and label sentence (for label extraction)
    4. Find k nearest neighbors based on z_c, compare label majority
    5. If valid, extract fields like original data
    6. Check for duplicates
    
    Args:
        synthetic_sentences: List of generated sentences (may include label sentence)
        dataset_name: Name of dataset (e.g., 'german-credit-data', 'adult')
        k: Number of nearest neighbors to check (default: 9)
        majority_threshold: Minimum fraction of neighbors that must have same label (default: 0.5)
        model_path: Optional path to model checkpoint
        batch_size: Batch size for encoding
        existing_sentences: Optional list of previously generated sentences (for duplicate checking)
    
    Returns:
        Dictionary with validation results:
        {
            'valid_indices': List of indices of valid samples,
            'invalid_indices': List of indices of invalid samples,
            'validation_details': List of dicts with details for each sample,
            'valid_data': List of valid samples with extracted fields (only if valid)
        }
    """
    print(f"\n{'='*70}")
    print(f"VALIDATING SYNTHETIC DATA")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"Number of synthetic samples: {len(synthetic_sentences)}")
    print(f"k (neighbors): {k}")
    print(f"Majority threshold: {majority_threshold}")
    print(f"{'='*70}\n")
    
    # Step 1: Load trained model
    print("Step 1: Loading trained model...")
    model, checkpoint = load_trained_model(dataset_name, model_path)
    
    # Get model path for cache
    if model_path is None:
        model_dir = f"models/{dataset_name}"
        model_files = glob.glob(os.path.join(model_dir, "best_model_*.pth"))
        if model_files:
            model_path = max(model_files, key=os.path.getmtime)
        else:
            model_path = f"models/{dataset_name}/default.pth"
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Step 2: Load original data from processed_data/{dataset_name}.csv
    print(f"\nStep 2: Loading original data from processed_data/{dataset_name}.csv...")
    try:
        original_sentences_full, original_labels = load_original_data_from_processed(dataset_name)
    except FileNotFoundError:
        print(f"  ⚠ Processed data not found, falling back to data_loader...")
        if dataset_name == "german-credit-data":
            original_sentences_full, original_labels, _, _ = load_german_credit_data_balanced(n_samples=None)
        else:
            original_sentences_full, original_labels, _, _ = load_dataset_generic(dataset_name)
    
    # Split original sentences into main parts (without label sentence)
    print(f"  Splitting original sentences into main parts (removing label sentence)...")
    original_main_sentences = []
    for sentence in original_sentences_full:
        main_part, _ = split_sentence_into_main_and_label(sentence, dataset_name)
        original_main_sentences.append(main_part)
    
    # Step 3: Encode original data (main parts only) and cache
    cache_path = get_cache_path(dataset_name, model_path)
    cached_data = load_cached_encodings(cache_path)
    
    if cached_data is not None:
        z_c_original, z_d_original, cached_sentences, cached_labels = cached_data
        # Verify cache matches current data
        if len(cached_sentences) == len(original_main_sentences):
            print(f"  ✓ Using cached encodings")
            print(f"  Original dataset size: {len(original_main_sentences)}")
            print(f"  Original label distribution: {np.sum(original_labels)} (1), {len(original_labels) - np.sum(original_labels)} (0)")
        else:
            print(f"  ⚠ Cache size mismatch, re-encoding...")
            cached_data = None
    
    if cached_data is None:
        print(f"\nStep 3: Encoding original data (main parts only, no label sentence)...")
        z_c_original, z_d_original = extract_z_c_z_d(
            model, 
            original_main_sentences, 
            tokenizer, 
            batch_size=batch_size,
            dataset_name=dataset_name
        )
        print(f"  Original z_c shape: {z_c_original.shape}")
        print(f"  Original z_d shape: {z_d_original.shape}")
        
        # Save to cache (using main sentences, not full sentences)
        save_cached_encodings(cache_path, z_c_original, z_d_original, original_main_sentences, original_labels)
    
    # Step 4: Split synthetic sentences into main part and label sentence
    print(f"\nStep 4: Splitting synthetic sentences into main part and label sentence...")
    synthetic_main_sentences = []
    synthetic_label_sentences = []
    synthetic_labels_extracted = []
    
    # Debug: Check first few sentences
    if len(synthetic_sentences) > 0:
        print(f"  Debug: First sentence (first 200 chars): {synthetic_sentences[0][:200]}...")
    
    for i, sentence in enumerate(synthetic_sentences):
        main_part, label_sentence = split_sentence_into_main_and_label(sentence, dataset_name)
        synthetic_main_sentences.append(main_part)
        synthetic_label_sentences.append(label_sentence)
        
        # Extract label from label sentence
        extracted_label = extract_label_from_label_sentence(label_sentence, dataset_name)
        if extracted_label is None:
            # Fallback: try to extract from full sentence if label sentence is empty
            extracted_label = extract_label_from_label_sentence(sentence, dataset_name)
        
        # Debug first few cases
        if i < 3:
            print(f"  Debug sample {i+1}:")
            print(f"    Full sentence (first 150 chars): {sentence[:150]}...")
            print(f"    Label sentence: '{label_sentence[:100] if label_sentence else 'EMPTY'}...'")
            print(f"    Extracted label: {extracted_label}")
        
        synthetic_labels_extracted.append(extracted_label)
    
    n_extracted = sum(1 for l in synthetic_labels_extracted if l is not None)
    print(f"  Extracted {n_extracted} labels from label sentences")
    
    if n_extracted == 0:
        print(f"  ⚠ WARNING: No labels extracted! This will cause all samples to be invalid.")
        print(f"  Checking if sentences have label sentence pattern...")
        # Check a few sentences for label patterns
        for i in range(min(3, len(synthetic_sentences))):
            sentence = synthetic_sentences[i]
            has_pattern = (
                'credit application has been evaluated' in sentence.lower() or
                'income classification' in sentence.lower() or
                'good credit risk' in sentence.lower() or
                'bad credit risk' in sentence.lower()
            )
            print(f"    Sample {i+1}: Has label pattern = {has_pattern}")
    
    # Step 5: Encode synthetic data (main parts only)
    print(f"\nStep 5: Encoding synthetic data (main parts only, no label sentence)...")
    z_c_synthetic, z_d_synthetic = extract_z_c_z_d(
        model, 
        synthetic_main_sentences, 
        tokenizer, 
        batch_size=batch_size,
        dataset_name=dataset_name
    )
    print(f"  Synthetic z_c shape: {z_c_synthetic.shape}")
    print(f"  Synthetic z_d shape: {z_d_synthetic.shape}")
    
    # Step 6: Validate each synthetic sample
    print(f"\nStep 6: Validating {len(synthetic_sentences)} synthetic samples...")
    print(f"  - Comparing z_c with {len(original_main_sentences)} original samples")
    print(f"  - Finding {k} nearest neighbors for each sample")
    print(f"  - Checking if extracted label matches majority (threshold: {majority_threshold})")
    
    valid_indices = []
    invalid_indices = []
    validation_details = []
    
    original_labels_array = np.array(original_labels)
    
    for i in range(len(synthetic_sentences)):
        sentence = synthetic_sentences[i]
        main_sentence = synthetic_main_sentences[i]
        extracted_label = synthetic_labels_extracted[i]
        
        # Skip if label cannot be extracted
        if extracted_label is None:
            invalid_indices.append(i)
            validation_details.append({
                'index': i,
                'sentence': sentence[:100] + '...' if len(sentence) > 100 else sentence,
                'extracted_label': None,
                'is_valid': False,
                'reason': 'label_extraction_failed',
                'majority_label': None,
                'majority_ratio': None,
                'label_matches': False
            })
            continue
        
        z_c_sample = z_c_synthetic[i]
        
        # Find k nearest neighbors by comparing z_c distances and check majority label
        is_valid, majority_label, majority_ratio = validate_sample(
            z_c_sample,
            z_c_original,
            original_labels_array,
            k=k,
            majority_threshold=majority_threshold,
            metric='cosine'
        )
        
        # Check if extracted label matches majority label
        label_matches = (extracted_label == majority_label)
        has_clear_majority = (majority_ratio > 0.5)  # Must have >50% majority
        final_valid = label_matches and has_clear_majority
        
        validation_details.append({
            'index': i,
            'sentence': sentence[:100] + '...' if len(sentence) > 100 else sentence,
            'extracted_label': int(extracted_label),
            'majority_label': int(majority_label),
            'majority_ratio': float(majority_ratio),
            'label_matches': bool(label_matches),
            'is_valid': bool(final_valid)
        })
        
        if final_valid:
            valid_indices.append(i)
        else:
            invalid_indices.append(i)
        
        if (i + 1) % 100 == 0:
            print(f"  Validated {i + 1}/{len(synthetic_sentences)} samples...")
    
    # Step 7: Extract fields for valid samples (only if validation passed)
    print(f"\nStep 7: Extracting fields for {len(valid_indices)} valid samples...")
    # Import extract_fields_from_sentences from generation script
    # For now, we'll create a simplified version or import it
    valid_data = []
    for idx in valid_indices:
        entry = {
            'sentence': synthetic_sentences[idx],
            'label': synthetic_labels_extracted[idx]
        }
        valid_data.append(entry)
    
    # Step 8: Check for duplicates (on valid samples only)
    print(f"\nStep 8: Checking for duplicates...")
    valid_main_sentences = [synthetic_main_sentences[i] for i in valid_indices]
    is_unique, duplicate_details = check_duplicates(
        synthetic_sentences=valid_main_sentences,
        original_sentences=original_main_sentences,
        existing_sentences=existing_sentences
    )
    
    # Remove duplicates from valid_indices
    final_valid_indices = [valid_indices[i] for i in range(len(valid_indices)) if is_unique[i]]
    n_duplicates = len(valid_indices) - len(final_valid_indices)
    
    if n_duplicates > 0:
        print(f"  ⚠ Found {n_duplicates} duplicate(s) among valid samples")
        invalid_indices.extend([valid_indices[i] for i in range(len(valid_indices)) if not is_unique[i]])
    else:
        print(f"  ✓ No duplicates found")
    
    print(f"\n{'='*70}")
    print("VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"Valid samples: {len(final_valid_indices)} ({100*len(final_valid_indices)/len(synthetic_sentences):.1f}%)")
    print(f"Invalid samples: {len(invalid_indices)} ({100*len(invalid_indices)/len(synthetic_sentences):.1f}%)")
    print(f"{'='*70}\n")
    
    return {
        'valid_indices': final_valid_indices,
        'invalid_indices': invalid_indices,
        'validation_details': validation_details,
        'valid_data': [valid_data[i] for i in range(len(valid_data)) if valid_indices[i] in final_valid_indices]
    }


def filter_valid_samples(
    synthetic_data: List[Dict],
    validation_results: Dict
) -> List[Dict]:
    """
    Filter synthetic data to keep only valid samples.
    
    Args:
        synthetic_data: List of synthetic data entries (each with 'sentence', 'label', etc.)
        validation_results: Results from validate_synthetic_data()
    
    Returns:
        Filtered list containing only valid samples
    """
    valid_indices = set(validation_results['valid_indices'])
    filtered_data = [entry for i, entry in enumerate(synthetic_data) if i in valid_indices]
    return filtered_data


def visualize_synthetic_data_pca(
    synthetic_sentences: List[str],
    synthetic_labels: List[int],
    dataset_name: str,
    model_path: Optional[str] = None,
    save_dir: str = "visualizations",
    compare_with_original: bool = True
):
    """
    Visualize z_c and z_d of synthetic data using PCA 2D projection.
    Optionally compare with original data.
    
    Args:
        synthetic_sentences: List of synthetic sentences
        synthetic_labels: List of labels for synthetic sentences
        dataset_name: Name of dataset
        model_path: Optional path to model checkpoint
        save_dir: Directory to save visualizations
        compare_with_original: If True, also show original data for comparison
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("VISUALIZING SYNTHETIC DATA (PCA 2D)")
    print(f"{'='*70}")
    
    # Load model
    model, checkpoint = load_trained_model(dataset_name, model_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Extract z_c and z_d from synthetic data
    # First, split sentences into main parts (without label sentence) for encoding
    print(f"\nSplitting synthetic sentences into main parts (removing label sentence)...")
    synthetic_main_sentences = []
    for sentence in synthetic_sentences:
        main_part, _ = split_sentence_into_main_and_label(sentence, dataset_name)
        synthetic_main_sentences.append(main_part)
    
    print(f"Extracting z_c and z_d from {len(synthetic_main_sentences)} synthetic samples...")
    z_c_synthetic, z_d_synthetic = extract_z_c_z_d(
        model, 
        synthetic_main_sentences, 
        tokenizer, 
        batch_size=32,
        dataset_name=dataset_name
    )
    synthetic_labels_array = np.array(synthetic_labels)
    
    print(f"✓ Synthetic z_c shape: {z_c_synthetic.shape}")
    print(f"✓ Synthetic z_d shape: {z_d_synthetic.shape}")
    
    # Load original data if comparison is enabled
    z_c_original = None
    z_d_original = None
    original_labels_array = None
    
    if compare_with_original:
        print(f"\nLoading original data for comparison...")
        if model_path is None:
            model_dir = f"models/{dataset_name}"
            model_files = glob.glob(os.path.join(model_dir, "best_model_*.pth"))
            if model_files:
                model_path = max(model_files, key=os.path.getmtime)
            else:
                model_path = f"models/{dataset_name}/default.pth"
        
        cache_path = get_cache_path(dataset_name, model_path)
        cached_data = load_cached_encodings(cache_path)
        
        if cached_data is not None:
            z_c_original, z_d_original, _, original_labels = cached_data
            original_labels_array = np.array(original_labels)
            print(f"✓ Loaded original data from cache: {len(original_labels)} samples")
        else:
            # Load and extract if not cached
            # IMPORTANT: Use the SAME method as during training to ensure consistency
            # Original sentences from load_german_credit_data_balanced() already do NOT have label sentence
            # (format_sentence_from_template skips the last sentence which contains label)
            if dataset_name == "german-credit-data":
                original_sentences, original_labels, _, _ = load_german_credit_data_balanced(n_samples=None)
            else:
                original_sentences, original_labels, _, _ = load_dataset_generic(dataset_name)
            
            # IMPORTANT: Original sentences already do NOT have label sentence
            # Use the SAME extraction method as during training (via dataloader)
            # This ensures z_c and z_d are extracted identically
            print(f"  Extracting z_c and z_d using same method as training...")
            print(f"  (Original sentences already do not contain label sentence)")
            
            # Use extract_z_c_z_d but WITHOUT removing label sentence again
            # Pass dataset_name=None to skip label removal (sentences are already clean)
            z_c_original, z_d_original = extract_z_c_z_d(
                model, 
                original_sentences,  # Already clean (no label sentence)
                tokenizer, 
                batch_size=32,
                dataset_name=None  # Don't remove label sentence again
            )
            original_labels_array = np.array(original_labels)
            print(f"✓ Extracted original data: {len(original_labels)} samples")
            
            # Save to cache for future use
            save_cached_encodings(cache_path, z_c_original, z_d_original, original_sentences, original_labels)
    
    # Verify label consistency with neighbors (for visualization)
    # This should be done AFTER loading original data
    majority_labels_from_neighbors = None
    if compare_with_original and z_c_original is not None:
        print(f"\nVerifying label consistency with neighbors...")
        
        majority_labels_list = []
        majority_ratios_list = []
        for i in range(len(synthetic_sentences)):
            z_c_sample = z_c_synthetic[i]
            distances, indices = find_k_nearest_neighbors(z_c_sample, z_c_original, k=9, metric='cosine')
            neighbor_labels = original_labels_array[indices]
            
            # Count labels with proper handling of ties
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            majority_count = counts.max()
            majority_label = unique_labels[counts.argmax()]
            majority_ratio = majority_count / len(neighbor_labels)
            
            majority_labels_list.append(int(majority_label))
            majority_ratios_list.append(majority_ratio)
        
        majority_labels_from_neighbors = np.array(majority_labels_list)
        majority_ratios = np.array(majority_ratios_list)
        
        # Check consistency - label must match AND majority ratio must be > threshold
        label_matches = (synthetic_labels_array == majority_labels_from_neighbors)
        ratio_valid = (majority_ratios > 0.5)  # Must have clear majority (>50%)
        valid_mask = label_matches & ratio_valid
        
        n_matches = np.sum(valid_mask)
        n_mismatches = len(synthetic_sentences) - n_matches
        
        print(f"  ✓ {n_matches}/{len(synthetic_sentences)} synthetic samples have labels matching majority of neighbors ({100*n_matches/len(synthetic_sentences):.1f}%)")
        
        if n_mismatches > 0:
            label_only_mismatches = np.sum(label_matches & ~ratio_valid)
            ratio_only_mismatches = np.sum(~label_matches & ratio_valid)
            both_mismatches = np.sum(~label_matches & ~ratio_valid)
            
            print(f"  ⚠ Filtering out {n_mismatches} samples:")
            if label_only_mismatches > 0:
                print(f"     - {label_only_mismatches} samples: label mismatch (ratio OK)")
            if ratio_only_mismatches > 0:
                print(f"     - {ratio_only_mismatches} samples: weak majority ratio (label OK)")
            if both_mismatches > 0:
                print(f"     - {both_mismatches} samples: both label and ratio mismatch")
            
            # Filter out mismatches
            z_c_synthetic = z_c_synthetic[valid_mask]
            z_d_synthetic = z_d_synthetic[valid_mask]
            synthetic_labels_array = synthetic_labels_array[valid_mask]
            majority_labels_from_neighbors = majority_labels_from_neighbors[valid_mask]
            
            print(f"  ✓ After filtering: {len(z_c_synthetic)} samples remain for visualization")
    
    # Apply PCA
    # IMPORTANT: Fit PCA on original data only (same as during training)
    # Then transform both original and synthetic data using the same PCA
    # This ensures consistent projection for comparison
    print(f"\nApplying PCA(2) to project to 2D...")
    print("  Fitting PCA on original data (same as during training)...")
    
    if compare_with_original and z_c_original is not None:
        # Fit PCA on original data only (same as visualize_latent_space_pca during training)
        pca_c = PCA(n_components=2)
        pca_d = PCA(n_components=2)
        
        # Fit on original data
        pca_c.fit(z_c_original)
        pca_d.fit(z_d_original)
        
        # Transform both original and synthetic using the same PCA
        z_c_original_2d = pca_c.transform(z_c_original)
        z_c_synthetic_2d = pca_c.transform(z_c_synthetic)
        z_d_original_2d = pca_d.transform(z_d_original)
        z_d_synthetic_2d = pca_d.transform(z_d_synthetic)
        
        print("  ✓ Fitted PCA on original data, transformed both original and synthetic")
    else:
        # Fit PCA only on synthetic data (fallback if no original data)
        pca_c = PCA(n_components=2)
        pca_d = PCA(n_components=2)
        
        z_c_synthetic_2d = pca_c.fit_transform(z_c_synthetic)
        z_d_synthetic_2d = pca_d.fit_transform(z_d_synthetic)
        z_c_original_2d = None
        z_d_original_2d = None
    
    print(f"✓ z_c explained variance: {pca_c.explained_variance_ratio_.sum():.4f}")
    print(f"✓ z_d explained variance: {pca_d.explained_variance_ratio_.sum():.4f}")
    
    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # z_c scatter plot
    ax = axes[0]
    if compare_with_original and z_c_original_2d is not None:
        # Plot original data (lighter, smaller)
        ax.scatter(z_c_original_2d[original_labels_array == 0, 0], 
                  z_c_original_2d[original_labels_array == 0, 1],
                  c='lightblue', alpha=0.3, s=20, label='Original (Label 0)', marker='o')
        ax.scatter(z_c_original_2d[original_labels_array == 1, 0], 
                  z_c_original_2d[original_labels_array == 1, 1],
                  c='lightcoral', alpha=0.3, s=20, label='Original (Label 1)', marker='o')
    
    # Plot synthetic data (darker, larger)
    # Use majority labels from neighbors if available (for consistency check)
    # Note: Mismatches have already been filtered out above
    if majority_labels_from_neighbors is not None:
        # Plot by majority label (what neighbors suggest)
        ax.scatter(z_c_synthetic_2d[majority_labels_from_neighbors == 0, 0], 
                  z_c_synthetic_2d[majority_labels_from_neighbors == 0, 1],
                  c='blue', alpha=0.7, s=80, label='Synthetic (Label 0)', marker='^', edgecolors='darkblue', linewidths=1)
        ax.scatter(z_c_synthetic_2d[majority_labels_from_neighbors == 1, 0], 
                  z_c_synthetic_2d[majority_labels_from_neighbors == 1, 1],
                  c='red', alpha=0.7, s=80, label='Synthetic (Label 1)', marker='^', edgecolors='darkred', linewidths=1)
    else:
        # Plot by synthetic labels
        ax.scatter(z_c_synthetic_2d[synthetic_labels_array == 0, 0], 
                  z_c_synthetic_2d[synthetic_labels_array == 0, 1],
                  c='blue', alpha=0.7, s=80, label='Synthetic (Label 0)', marker='^', edgecolors='darkblue', linewidths=1)
        ax.scatter(z_c_synthetic_2d[synthetic_labels_array == 1, 0], 
                  z_c_synthetic_2d[synthetic_labels_array == 1, 1],
                  c='red', alpha=0.7, s=80, label='Synthetic (Label 1)', marker='^', edgecolors='darkred', linewidths=1)
    
    ax.set_xlabel(f'PC1 ({pca_c.explained_variance_ratio_[0]:.2%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca_c.explained_variance_ratio_[1]:.2%})', fontsize=12)
    title = f'z_c (Content) - PCA 2D Projection'
    if compare_with_original:
        title += f' (Synthetic vs Original)\n{len(z_c_synthetic)} synthetic samples shown'
    else:
        title += f'\n{len(z_c_synthetic)} synthetic samples'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)
    
    # z_d scatter plot
    ax = axes[1]
    if compare_with_original and z_d_original_2d is not None:
        # Plot original data (lighter, smaller)
        ax.scatter(z_d_original_2d[original_labels_array == 0, 0], 
                  z_d_original_2d[original_labels_array == 0, 1],
                  c='lightblue', alpha=0.3, s=20, label='Original (Label 0)', marker='o')
        ax.scatter(z_d_original_2d[original_labels_array == 1, 0], 
                  z_d_original_2d[original_labels_array == 1, 1],
                  c='lightcoral', alpha=0.3, s=20, label='Original (Label 1)', marker='o')
    
    # Plot synthetic data (darker, larger)
    # Use majority labels from neighbors if available (for consistency check)
    # Note: Mismatches have already been filtered out above
    if majority_labels_from_neighbors is not None:
        # Plot by majority label (what neighbors suggest)
        ax.scatter(z_d_synthetic_2d[majority_labels_from_neighbors == 0, 0], 
                  z_d_synthetic_2d[majority_labels_from_neighbors == 0, 1],
                  c='blue', alpha=0.7, s=80, label='Synthetic (Label 0)', marker='^', edgecolors='darkblue', linewidths=1)
        ax.scatter(z_d_synthetic_2d[majority_labels_from_neighbors == 1, 0], 
                  z_d_synthetic_2d[majority_labels_from_neighbors == 1, 1],
                  c='red', alpha=0.7, s=80, label='Synthetic (Label 1)', marker='^', edgecolors='darkred', linewidths=1)
    else:
        # Plot by synthetic labels
        ax.scatter(z_d_synthetic_2d[synthetic_labels_array == 0, 0], 
                  z_d_synthetic_2d[synthetic_labels_array == 0, 1],
                  c='blue', alpha=0.7, s=80, label='Synthetic (Label 0)', marker='^', edgecolors='darkblue', linewidths=1)
        ax.scatter(z_d_synthetic_2d[synthetic_labels_array == 1, 0], 
                  z_d_synthetic_2d[synthetic_labels_array == 1, 1],
                  c='red', alpha=0.7, s=80, label='Synthetic (Label 1)', marker='^', edgecolors='darkred', linewidths=1)
    
    ax.set_xlabel(f'PC1 ({pca_d.explained_variance_ratio_[0]:.2%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca_d.explained_variance_ratio_[1]:.2%})', fontsize=12)
    title = f'z_d (Demographic) - PCA 2D Projection'
    if compare_with_original:
        title += f' (Synthetic vs Original)\n{len(z_d_synthetic)} synthetic samples shown'
    else:
        title += f'\n{len(z_d_synthetic)} synthetic samples'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(save_dir, "synthetic_data_pca_2d.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved PCA visualization to {filename}")
    plt.close()
    
    print(f"{'='*70}\n")
