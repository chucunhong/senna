#!/usr/bin/env python3
"""
Script to generate synthetic data using various LLM models from Hugging Face.
Uses prompts from prompts.py and in-context examples from sampled data CSV.
Supports multiple models and includes label generation.

Usage:
    python generate_synthetic_data.py \
        --dataset german-credit-data \
        --csv-file sampled_data/german-credit-data/sampled_sentences_combined.csv \
        --output-file synthetic_data.json \
        --n-examples 10 \
        --model mistralai/Mistral-7B-Instruct-v0.2
"""

import argparse
import pandas as pd
import json
import os
import numpy as np
from typing import List, Dict
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from prompts import prompts


def load_icl_examples(csv_path: str, shuffle: bool = True, random_state: int = 42, dataset_name: str = None) -> str:
    """
    Load ALL in-context examples from CSV file with optional shuffling.
    Formats examples according to the template in prompts.py, including label description.
    
    Args:
        csv_path: Path to CSV file with sampled sentences
        shuffle: Whether to shuffle the data before sampling (default: True)
        random_state: Random seed for shuffling (default: 42)
        dataset_name: Name of dataset (e.g., 'german-credit-data', 'adult') for label formatting
    
    Returns:
        Formatted string of examples with labels (all examples from file)
    """
    print(f"Loading in-context examples from: {csv_path}")
    
    # Check if file is JSON and find corresponding CSV
    if csv_path.endswith('.json'):
        print(f"Warning: File is JSON, not CSV. Looking for corresponding CSV file...")
        # Try to find corresponding CSV file
        base_path = csv_path.replace('_metadata.json', '').replace('.json', '')
        csv_candidate = base_path + '.csv'
        if csv_candidate.endswith('_combined_metadata.csv'):
            csv_candidate = csv_candidate.replace('_combined_metadata.csv', '_combined.csv')
        elif csv_candidate.endswith('_metadata.csv'):
            csv_candidate = csv_candidate.replace('_metadata.csv', '.csv')
        
        # Also try in the same directory
        if not os.path.exists(csv_candidate):
            dir_path = os.path.dirname(csv_path)
            csv_filename = os.path.basename(csv_path).replace('_metadata.json', '.csv').replace('.json', '.csv')
            csv_candidate = os.path.join(dir_path, csv_filename)
        
        if os.path.exists(csv_candidate):
            print(f"Found corresponding CSV file: {csv_candidate}")
            csv_path = csv_candidate
        else:
            raise FileNotFoundError(
                f"Could not find corresponding CSV file for JSON file: {csv_path}\n"
                f"Please provide a CSV file with 'sentence' and 'label' columns."
            )
    
    df = pd.read_csv(csv_path)
    
    # Shuffle if requested
    if shuffle:
        print(f"Shuffling data with random_state={random_state}...")
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        print(f"✓ Data shuffled")
    
    # Use all examples from file
    df_sample = df
    
    # Format examples with labels according to template
    examples = []
    for idx, row in df_sample.iterrows():
        sentence = row['sentence']
        label = row.get('label', None)
        
        # Check if sentence already ends with label description
        sentence_clean = sentence.strip()
        already_has_label_desc = (
            sentence_clean.endswith('good credit risk.') or 
            sentence_clean.endswith('bad credit risk.') or
            sentence_clean.endswith('approved.') or
            sentence_clean.endswith('rejected.') or
            'decision is' in sentence_clean.lower() or
            'income classification' in sentence_clean.lower()
        )
        
        # Add label description if not already present and dataset_name is provided
        if not already_has_label_desc and dataset_name and label is not None:
            if dataset_name == 'german-credit-data':
                # Format: "The credit application has been evaluated and the decision is {label_description}."
                if label == 0 or str(label).strip() == '0':
                    label_desc = "good credit risk"
                elif label == 1 or str(label).strip() == '1':
                    label_desc = "bad credit risk"
                else:
                    label_desc = f"label {label}"
                
                # Ensure sentence ends with period before adding label description
                if not sentence_clean.endswith('.'):
                    sentence_clean += '.'
                sentence_clean += f" The credit application has been evaluated and the decision is {label_desc}."
            
            elif dataset_name == 'adult':
                # Format: "The income classification for this person is {label_description}."
                if label == 0 or str(label).strip() == '0':
                    label_desc = "income <=50K"
                elif label == 1 or str(label).strip() == '1':
                    label_desc = "income >50K"
                else:
                    label_desc = f"label {label}"
                
                # Ensure sentence ends with period before adding label description
                if not sentence_clean.endswith('.'):
                    sentence_clean += '.'
                sentence_clean += f" The income classification for this person is {label_desc}."
        
        # Format as example
        examples.append(f"Example {idx + 1}: {sentence_clean}")
    
    icl_text = "\n".join(examples)
    print(f"✓ Loaded {len(df_sample)} in-context examples (all examples from file, with labels formatted according to template)")
    
    return icl_text


def save_label_data(csv_path: str, dataset_name: str, output_dir: str = None):
    """
    Save label_0 and label_1 data separately to simple CSV files (only sentences).
    This allows reusing the data without needing to cluster again.
    
    Args:
        csv_path: Path to combined CSV file with all data
        dataset_name: Name of dataset
        output_dir: Output directory (default: sampled_data/{dataset_name}/)
    
    Returns:
        Tuple of (label_0_file, label_1_file) paths
    """
    if output_dir is None:
        output_dir = os.path.join("sampled_data", dataset_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving label data separately from: {csv_path}")
    
    # Check if file is JSON or CSV
    if csv_path.endswith('.json'):
        print(f"Warning: File is JSON, not CSV. Looking for corresponding CSV file...")
        # Try to find corresponding CSV file
        base_path = csv_path.replace('_metadata.json', '').replace('.json', '')
        csv_candidate = base_path + '.csv'
        if csv_candidate.endswith('_combined_metadata.csv'):
            csv_candidate = csv_candidate.replace('_combined_metadata.csv', '_combined.csv')
        elif csv_candidate.endswith('_metadata.csv'):
            csv_candidate = csv_candidate.replace('_metadata.csv', '.csv')
        
        # Also try in the same directory
        if not os.path.exists(csv_candidate):
            dir_path = os.path.dirname(csv_path)
            csv_filename = os.path.basename(csv_path).replace('_metadata.json', '.csv').replace('.json', '.csv')
            csv_candidate = os.path.join(dir_path, csv_filename)
        
        if os.path.exists(csv_candidate):
            print(f"Found corresponding CSV file: {csv_candidate}")
            csv_path = csv_candidate
        else:
            print(f"Could not find corresponding CSV file. Skipping label data saving.")
            print("Note: This function expects a CSV file with 'sentence' and 'label' columns.")
            return None, None
    
    # Try to read as CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        print("Skipping label data saving.")
        return None, None
    
    # Check available columns
    print(f"Available columns: {df.columns.tolist()}")
    
    # Find label column (try common names)
    label_col = None
    for col_name in ['label', 'Label', 'target', 'Target', 'class', 'Class']:
        if col_name in df.columns:
            label_col = col_name
            break
    
    if label_col is None:
        print(f"Warning: No label column found in CSV. Available columns: {df.columns.tolist()}")
        print("Skipping label-based splitting. Saving all sentences to a single file.")
        
        # Check if sentence column exists
        sentence_col = None
        for col_name in ['sentence', 'Sentence', 'text', 'Text']:
            if col_name in df.columns:
                sentence_col = col_name
                break
        
        if sentence_col is None:
            raise ValueError(f"Could not find sentence column. Available columns: {df.columns.tolist()}")
        
        # Save all sentences to a single file
        all_sentences_file = os.path.join(output_dir, "all_sentences.csv")
        df_simple = pd.DataFrame({'sentence': df[sentence_col]})
        df_simple.to_csv(all_sentences_file, index=False)
        print(f"✓ Saved {len(df_simple)} sentences to: {all_sentences_file}")
        return all_sentences_file, None
    
    # Split by label
    df_label_0 = df[df[label_col] == 0].copy()
    df_label_1 = df[df[label_col] == 1].copy()
    
    # Check if sentence column exists
    sentence_col = None
    for col_name in ['sentence', 'Sentence', 'text', 'Text']:
        if col_name in df.columns:
            sentence_col = col_name
            break
    
    if sentence_col is None:
        raise ValueError(f"Could not find sentence column. Available columns: {df.columns.tolist()}")
    
    # Save only sentences (simple format for easy reuse)
    label_0_file = os.path.join(output_dir, "label_0_data.csv")
    label_1_file = os.path.join(output_dir, "label_1_data.csv")
    
    # Save label_0 (only sentence column)
    if len(df_label_0) > 0:
        df_label_0_simple = pd.DataFrame({'sentence': df_label_0[sentence_col]})
        df_label_0_simple.to_csv(label_0_file, index=False)
        print(f"✓ Saved {len(df_label_0_simple)} label_0 sentences to: {label_0_file}")
    else:
        print(f"⚠ No label_0 data found. Skipping {label_0_file}")
    
    # Save label_1 (only sentence column)
    if len(df_label_1) > 0:
        df_label_1_simple = pd.DataFrame({'sentence': df_label_1[sentence_col]})
        df_label_1_simple.to_csv(label_1_file, index=False)
        print(f"✓ Saved {len(df_label_1_simple)} label_1 sentences to: {label_1_file}")
    else:
        print(f"⚠ No label_1 data found. Skipping {label_1_file}")
    
    return label_0_file, label_1_file


def format_prompt(dataset_name: str, icl_examples: str, n_samples: int = 2) -> tuple:
    """
    Format prompt with in-context examples for chat format.
    
    Args:
        dataset_name: Name of dataset (e.g., 'german-credit-data')
        icl_examples: In-context examples as formatted string
        n_samples: Number of samples to generate (for template formatting)
    
    Returns:
        Tuple of (system_message, user_message) for chat format
    """
    if dataset_name not in prompts:
        raise ValueError(f"Dataset '{dataset_name}' not found in prompts. Available: {list(prompts.keys())}")
    
    prompt_template = prompts[dataset_name]
    
    # Split by "System role:" and "User instructions:"
    parts = prompt_template.split('System role:')
    if len(parts) < 2:
        raise ValueError("Prompt template must contain 'System role:' section")
    
    rest = parts[1]
    user_parts = rest.split('User instructions:')
    
    if len(user_parts) < 2:
        raise ValueError("Prompt template must contain 'User instructions:' section")
    
    # System message is everything after "System role:" until "User instructions:"
    system_message = user_parts[0].strip()
    
    # User message is everything after "User instructions:"
    user_message_template = user_parts[1].strip()
    
    # Format user message with icl_examples and n_samples
    try:
        user_message = user_message_template.format(icl_examples=icl_examples, n_samples=n_samples)
    except KeyError:
        # If n_samples is not in template, just use icl_examples
        user_message = user_message_template.format(icl_examples=icl_examples)
    
    return system_message, user_message


def generate_with_model(
    system_message: str,
    user_message: str,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    max_new_tokens: int = 7000,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = None,
    use_pipeline: bool = True
) -> str:
    """
    Generate text using a language model with chat format.
    Supports various models from Hugging Face.
    
    Args:
        system_message: System message/role
        user_message: User message with instructions and examples
        model_name: Hugging Face model name (default: mistralai/Mistral-7B-Instruct-v0.2)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        device: Device to use ('cuda', 'cpu', or None for auto)
        use_pipeline: Whether to use pipeline API (simpler) or direct model loading
    
    Returns:
        Generated text
    """
    print(f"\nLoading model: {model_name}")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Format as chat messages
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    # Print messages before sending to model
    print("\n" + "="*70)
    print("MESSAGES TO SEND TO MODEL:")
    print("="*70)
    print("\n[SYSTEM MESSAGE]")
    print("-" * 70)
    print(system_message)
    print("\n[USER MESSAGE]")
    print("-" * 70)
    print(user_message)
    print("="*70 + "\n")
    
    if use_pipeline:
        # Use pipeline API with chat format
        print("Using pipeline API with chat format...")
        generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        
        # Apply chat template
        tokenizer = generator.tokenizer
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Print formatted prompt
        print("\n[FORMATTED PROMPT (after chat template)]")
        print("-" * 70)
        print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
        print("-" * 70 + "\n")
        
        print("Generating text...")
        outputs = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            return_full_text=False,
        )
        
        generated_text = outputs[0]['generated_text']
    else:
        # Direct model loading with chat template
        print("Loading model and tokenizer directly...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Print formatted prompt
        print("\n[FORMATTED PROMPT (after chat template)]")
        print("-" * 70)
        print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
        print("-" * 70 + "\n")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model = model.to(device)
        
        # Generate
        print("Generating text...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode output (only the generated part, not the prompt)
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    print("✓ Generation complete")
    return generated_text


def parse_generated_output(generated_text: str, dataset_name: str) -> List[Dict]:
    """
    Parse generated output to extract sentences and labels.
    Handles various output formats including JSON and natural language.
    
    Args:
        generated_text: Generated text that should contain sentences and labels
        dataset_name: Name of dataset for context
    
    Returns:
        List of dictionaries with 'sentence' and 'label' keys
    """
    import re
    
    results = []
    
    # Try to parse as JSON first
    try:
        # Look for JSON object
        json_match = re.search(r'\{[^{}]*\}', generated_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed_json = json.loads(json_str)
            
            # Extract sentences and labels from JSON
            for key, value in parsed_json.items():
                if isinstance(value, str):
                    # Try to extract sentence and label from the value
                    sentence, label = extract_sentence_and_label(value)
                    if sentence:
                        results.append({
                            'sentence': sentence,
                            'label': label,
                            'raw': value
                        })
            
            if results:
                return results
    except (json.JSONDecodeError, Exception) as e:
        print(f"Could not parse as JSON: {e}")
    
    # Try to extract sentences with labels from natural language
    # Pattern: sentence text (Label: X) or sentence text. Label: X
    patterns = [
        r'([^\(]+)\s*\(Label:\s*(\d+)\)',  # sentence (Label: 0)
        r'([^\.]+)\.\s*Label:\s*(\d+)',     # sentence. Label: 0
        r'Example\s+\d+:\s*([^\(]+)\s*\(Label:\s*(\d+)\)',  # Example 1: sentence (Label: 0)
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, generated_text, re.IGNORECASE)
        if matches:
            for match in matches:
                sentence = match[0].strip()
                try:
                    label = int(match[1].strip())
                    results.append({
                        'sentence': sentence,
                        'label': label,
                        'raw': match[0]
                    })
                except (ValueError, IndexError):
                    continue
            
            if results:
                return results
    
    # Fallback: try to extract just sentences and infer labels
    # Split by newlines and look for sentence patterns
    lines = generated_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('Example'):
            continue
        
        # Try to extract sentence and label
        sentence, label = extract_sentence_and_label(line)
        if sentence:
            results.append({
                'sentence': sentence,
                'label': label,
                'raw': line
            })
    
    # If still no results, return raw text as single sentence
    if not results:
        results.append({
            'sentence': generated_text.strip(),
            'label': None,
            'raw': generated_text
        })
    
    return results


def extract_sentence_and_label(text: str) -> tuple:
    """
    Extract sentence and label from text.
    Handles various formats like "sentence (Label: 0)" or "sentence. Label: 0"
    
    Args:
        text: Text to extract from
    
    Returns:
        Tuple of (sentence, label) where label is int or None
    """
    import re
    
    # Try patterns for label extraction
    label_patterns = [
        r'\(Label:\s*(\d+)\)',
        r'Label:\s*(\d+)',
        r'label:\s*(\d+)',
    ]
    
    label = None
    sentence = text
    
    for pattern in label_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                label = int(match.group(1))
                # Remove label from sentence
                sentence = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
                # Clean up trailing punctuation
                sentence = re.sub(r'[\.\,\;]+$', '', sentence).strip()
                break
            except (ValueError, IndexError):
                continue
    
    return sentence, label


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data using various LLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with Mistral
  python generate_synthetic_data.py \\
      --dataset german-credit-data \\
      --csv-file sampled_data/german-credit-data/sampled_sentences_combined.csv \\
      --output-file synthetic_data.json

  # With custom model and more examples
  python generate_synthetic_data.py \\
      --dataset german-credit-data \\
      --csv-file sampled_data/german-credit-data/sampled_sentences_combined.csv \\
      --output-file synthetic_data.json \\
      --n-examples 100 \\
      --model mistralai/Mistral-7B-Instruct-v0.2 \\
      --no-shuffle

  # With Llama model
  python generate_synthetic_data.py \\
      --dataset adult \\
      --csv-file sampled_data/adult/sampled_sentences_combined.csv \\
      --model meta-llama/Llama-2-7b-chat-hf
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'german-credit-data', 'adult')"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        required=True,
        help="Path to CSV file with sampled sentences for in-context examples"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file path for generated data (if not specified, will auto-generate with timestamp and n_examples)"
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=1,
        help="Number of data samples to generate (default: 1)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Hugging Face model name (default: mistralai/Mistral-7B-Instruct-v0.2). "
             "Examples: mistralai/Mistral-7B-Instruct-v0.2, meta-llama/Llama-2-7b-chat-hf, "
             "microsoft/Phi-3-mini-4k-instruct"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter (default: 0.9)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=['cuda', 'cpu', 'auto'],
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--no-pipeline",
        action="store_true",
        help="Use direct model loading instead of pipeline API"
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Do not shuffle the in-context examples before loading"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Generate output file name if not specified
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("sampled_data", args.dataset)
        os.makedirs(output_dir, exist_ok=True)
        args.output_file = os.path.join(
            output_dir,
            f"synthetic_data_{timestamp}_n{args.n_examples}.json"
        )
    
    print("="*70)
    print("SYNTHETIC DATA GENERATION")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Number of samples to generate: {args.n_examples}")
    print(f"Shuffle ICL examples: {not args.no_shuffle}")
    print(f"Output file: {args.output_file}")
    print("="*70)
    
    # Load in-context examples
    if not os.path.exists(args.csv_file):
        raise FileNotFoundError(f"CSV file not found: {args.csv_file}")
    
    # Save label data separately for future reuse
    output_dir = os.path.join("sampled_data", args.dataset)
    save_label_data(args.csv_file, args.dataset, output_dir)
    
    icl_examples = load_icl_examples(
        args.csv_file, 
        shuffle=not args.no_shuffle,
        random_state=args.random_state,
        dataset_name=args.dataset
    )
    
    # Format prompt for chat
    print(f"\nFormatting prompt for dataset: {args.dataset}")
    system_message, user_message = format_prompt(args.dataset, icl_examples, n_samples=args.n_examples)
    
    # Generate samples
    all_results = []
    for i in range(args.n_examples):
        print(f"\n{'='*70}")
        print(f"Generating sample {i+1}/{args.n_examples}")
        print(f"{'='*70}")
        
        generated_text = generate_with_model(
            system_message=system_message,
            user_message=user_message,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
            use_pipeline=not args.no_pipeline
        )
        
        # Parse output to extract sentences and labels
        parsed_results = parse_generated_output(generated_text, args.dataset)
        
        for result in parsed_results:
            result['sample_id'] = i + 1
            result['raw_generated_text'] = generated_text
            all_results.append(result)
        
        print(f"\nGenerated text (sample {i+1}):")
        print("-" * 70)
        print(generated_text[:500] + "..." if len(generated_text) > 500 else generated_text)
        print("-" * 70)
        print(f"\nParsed {len(parsed_results)} sentence(s) with labels")
        for idx, result in enumerate(parsed_results, 1):
            print(f"  {idx}. Sentence: {result['sentence'][:80]}...")
            print(f"     Label: {result['label']}")
    
    # Save results
    output_data = {
        'dataset': args.dataset,
        'model': args.model,
        'n_generated_samples': args.n_examples,
        'shuffled_icl': not args.no_shuffle,
        'random_state': args.random_state,
        'generated_samples': all_results
    }
    
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("GENERATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {args.output_file}")
    print(f"Generated {len(all_results)} sentence(s) from {args.n_examples} sample(s)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
