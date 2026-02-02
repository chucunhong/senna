#!/usr/bin/env python3
"""
Script to generate synthetic data using Google Gemini API.
Uses prompts from prompts.py and in-context examples from sampled data CSV.
Reads API keys from .env file.

Usage:
    python generate_synthetic_data_gemini.py \
        --dataset german-credit-data \
        --csv-file sampled_data/german-credit-data/sampled_sentences_combined.csv \
        --output-file synthetic_data.json \
        --n-examples 10 \
        --model gemini-2.0-flash-exp
"""

import argparse
import pandas as pd
import json
import os
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from prompts import prompts


def convert_to_original_csv_format(combined_data: List[Dict], dataset_name: str) -> pd.DataFrame:
    """
    Convert combined data (with sentence + extracted fields) to original CSV format.
    Only keeps columns that exist in the original CSV file.
    
    Args:
        combined_data: List of entries with sentence + all extracted fields
        dataset_name: Name of dataset (e.g., 'german-credit-data', 'adult')
    
    Returns:
        DataFrame with columns matching original CSV format
    """
    # Define column order for each dataset (matching original CSV)
    if dataset_name == 'german-credit-data':
        columns = [
            'checking-account', 'duration', 'credit-history', 'purpose', 'credit-amount',
            'savings-account', 'employment-since', 'installment-rate', 'other-debtors',
            'residence-since', 'property', 'age', 'other-installment', 'housing',
            'existing-credits', 'job', 'number-people-provide-maintenance-for',
            'telephone', 'foreign-worker', 'sex', 'marital-status', 'class-label'
        ]
    elif dataset_name == 'adult':
        columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
            'marital-status', 'occupation', 'relationship', 'race', 'gender',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class-label'
        ]
    else:
        # For unknown datasets, use all available columns except 'sentence'
        columns = [col for col in combined_data[0].keys() if col != 'sentence']
    
    # Create DataFrame with only the columns that exist in original CSV
    rows = []
    for entry in combined_data:
        row = {}
        for col in columns:
            # Get value from entry (may be None if not extracted)
            row[col] = entry.get(col, None)
        rows.append(row)
    
    df = pd.DataFrame(rows, columns=columns)
    
    return df

# Try to import Google Generative AI SDK
try:
    from google import genai
    from dotenv import load_dotenv
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠ Warning: google-genai or python-dotenv package not installed.")
    print("Install with: pip install google-genai python-dotenv")

# Load environment variables from .env file
if GEMINI_AVAILABLE:
    load_dotenv()


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


def generate_with_gemini(
    system_message: str,
    user_message: str,
    model: str = "gemini-3-pro-preview",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_output_tokens: int = 2000,
) -> str:
    """
    Generate text using Google Gemini API.
    
    Args:
        system_message: System message/role
        user_message: User message with instructions and examples
        model: Model name (e.g., 'gemini-2.0-flash-exp', 'gemini-1.5-pro')
        api_key: API key (if None, will try to get from GOOGLE_API_KEY or GEMINI_API_KEY environment variable)
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_output_tokens: Maximum number of tokens to generate
    
    Returns:
        Generated text
    """
    if not GEMINI_AVAILABLE:
        raise ImportError(
            "google-genai and python-dotenv packages are required. "
            "Install with: pip install google-genai python-dotenv"
        )
    
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if api_key is None:
            raise ValueError(
                "API key not provided and GOOGLE_API_KEY/GEMINI_API_KEY environment variable not set. "
                "Please provide --api-key or set the environment variable in .env file."
            )
    
    # Set API key in environment if provided
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    # Initialize client (will use API key from environment)
    print(f"\nUsing Google Gemini API with model: {model}")
    client = genai.Client()
    
    # Combine system message and user message for Gemini
    # Gemini uses a single prompt, so we combine system and user messages
    full_prompt = f"{system_message}\n\n{user_message}"
    
    # Print messages before sending to API
    print("\n" + "="*70)
    print("MESSAGES TO SEND TO API:")
    print("="*70)
    print("\n[SYSTEM MESSAGE]")
    print("-" * 70)
    print(system_message)
    print("\n[USER MESSAGE]")
    print("-" * 70)
    print(user_message[:1000] + "..." if len(user_message) > 1000 else user_message)
    print("="*70 + "\n")
    
    # Call API
    print("Calling API...")
    try:
        # Call generate_content directly (as shown in Gemini 3 Developer Guide)
        # Simple call with just model and contents
        response = client.models.generate_content(
            model=model,
            contents=full_prompt
        )
        
        generated_text = response.text
        
        # Print usage info if available
        print(f"\n✓ Generation complete")
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            if hasattr(usage, 'prompt_token_count'):
                print(f"  Prompt tokens: {usage.prompt_token_count}")
            if hasattr(usage, 'candidates_token_count'):
                print(f"  Completion tokens: {usage.candidates_token_count}")
            if hasattr(usage, 'total_token_count'):
                print(f"  Total tokens: {usage.total_token_count}")
        
        return generated_text
        
    except Exception as e:
        print(f"\n❌ Error calling API: {e}")
        raise


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
    
    # Try to parse as JSON first (most common format)
    try:
        # Try to find JSON object (handle nested braces and text before/after)
        json_str = generated_text.strip()
        
        # Remove markdown code blocks if present
        if json_str.startswith('```'):
            lines = json_str.split('\n')
            json_str = '\n'.join(lines[1:-1]) if len(lines) > 2 else json_str
            json_str = json_str.strip()
        
        # Try to extract JSON object from text (in case there's text before/after)
        # Find the first { and last } to extract JSON
        first_brace = json_str.find('{')
        last_brace = json_str.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = json_str[first_brace:last_brace+1]
        
        # Try to parse as JSON
        parsed_json = json.loads(json_str)
        
        # Extract sentences and labels from JSON
        # Sort keys numerically to maintain order (handle both string and int keys)
        def sort_key(x):
            if isinstance(x, str) and x.isdigit():
                return int(x)
            elif isinstance(x, int):
                return x
            else:
                return float('inf')
        
        sorted_keys = sorted(parsed_json.keys(), key=sort_key)
        
        for key in sorted_keys:
            value = parsed_json[key]
            if isinstance(value, str):
                # Extract sentence and label from the value
                sentence, label = extract_sentence_and_label(value, dataset_name)
                if sentence:
                    results.append({
                        'sentence': sentence,
                        'label': label,
                        'raw': value,
                        'json_key': str(key)
                    })
        
        if results:
            print(f"✓ Parsed {len(results)} samples from JSON (expected all {len(sorted_keys)} entries)")
            return results
    except (json.JSONDecodeError, Exception) as e:
        print(f"Could not parse as JSON: {e}")
        print("Trying alternative parsing methods...")
    
    # Try to extract sentences with labels from natural language
    # Pattern: sentence ending with "good credit risk" or "bad credit risk"
    if dataset_name == 'german-credit-data':
        # Pattern: Full sentence ending with credit risk decision
        credit_risk_pattern = r'([^\.]+(?:\.\s+[^\.]+)*\.\s+The\s+credit\s+application\s+has\s+been\s+evaluated\s+and\s+the\s+decision\s+is\s+(good|bad)\s+credit\s+risk\.)'
        matches = re.findall(credit_risk_pattern, generated_text, re.IGNORECASE | re.DOTALL)
        if matches:
            for match in matches:
                full_sentence = match[0].strip()
                risk_type = match[1].lower()
                label = 0 if risk_type == 'good' else 1
                # Remove the label description to get clean sentence
                sentence = re.sub(r'\.?\s*The\s+credit\s+application\s+has\s+been\s+evaluated\s+and\s+the\s+decision\s+is\s+(good|bad)\s+credit\s+risk\.?', '', full_sentence, flags=re.IGNORECASE).strip()
                if sentence:
                    results.append({
                        'sentence': sentence,
                        'label': label,
                        'raw': full_sentence
                    })
            if results:
                print(f"✓ Parsed {len(results)} samples from natural language (credit risk pattern)")
                return results
    
    # Try numeric label patterns
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
                print(f"✓ Parsed {len(results)} samples from natural language (numeric label pattern)")
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


def extract_sentence_and_label(text: str, dataset_name: str = None) -> tuple:
    """
    Extract sentence and label from text.
    Handles various formats including "good credit risk" / "bad credit risk" and "income <=50K" / "income >50K"
    
    Args:
        text: Text to extract from
        dataset_name: Name of dataset to determine label format
    
    Returns:
        Tuple of (sentence, label) where label is int or None
    """
    import re
    
    label = None
    sentence = text.strip()
    
    # Try to extract label from natural language format first
    if dataset_name == 'german-credit-data':
        # Look for "good credit risk" or "bad credit risk"
        if re.search(r'good\s+credit\s+risk', sentence, re.IGNORECASE):
            label = 0  # 0 = good credit risk
            # Remove the label description
            sentence = re.sub(r'\.?\s*The\s+credit\s+application\s+has\s+been\s+evaluated\s+and\s+the\s+decision\s+is\s+good\s+credit\s+risk\.?', '', sentence, flags=re.IGNORECASE).strip()
        elif re.search(r'bad\s+credit\s+risk', sentence, re.IGNORECASE):
            label = 1  # 1 = bad credit risk
            # Remove the label description
            sentence = re.sub(r'\.?\s*The\s+credit\s+application\s+has\s+been\s+evaluated\s+and\s+the\s+decision\s+is\s+bad\s+credit\s+risk\.?', '', sentence, flags=re.IGNORECASE).strip()
    
    elif dataset_name == 'adult':
        # Look for "income <=50K" or "income >50K"
        if re.search(r'income\s*<=\s*50K', sentence, re.IGNORECASE):
            label = 0  # 0 = income <=50K
            sentence = re.sub(r'\.?\s*The\s+income\s+classification\s+for\s+this\s+person\s+is\s+income\s*<=\s*50K\.?', '', sentence, flags=re.IGNORECASE).strip()
        elif re.search(r'income\s*>\s*50K', sentence, re.IGNORECASE):
            label = 1  # 1 = income >50K
            sentence = re.sub(r'\.?\s*The\s+income\s+classification\s+for\s+this\s+person\s+is\s+income\s*>\s*50K\.?', '', sentence, flags=re.IGNORECASE).strip()
    
    # If not found, try numeric label patterns
    if label is None:
        label_patterns = [
            r'\(Label:\s*(\d+)\)',
            r'Label:\s*(\d+)',
            r'label:\s*(\d+)',
        ]
        
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
    
    # Clean up sentence
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    sentence = re.sub(r'\.+$', '.', sentence)  # Ensure single period at end
    
    return sentence, label


def extract_fields_from_sentences(results: List[Dict], dataset_name: str) -> List[Dict]:
    """
    Extract structured fields from parsed sentences using template-based reverse extraction.
    Matches the logic from data_loader.py to ensure consistency.
    
    Args:
        results: List of parsed results with 'sentence' and 'label'
        dataset_name: Name of dataset to determine extraction patterns
    
    Returns:
        List of dictionaries with extracted fields matching original data structure
    """
    import re
    import json
    
    # Load template to understand field structure
    try:
        with open('sentence_templates.json', 'r', encoding='utf-8') as f:
            templates = json.load(f)
        template_config = templates.get(dataset_name, {})
        field_mapping = template_config.get('field_mapping', {})
    except (FileNotFoundError, KeyError):
        field_mapping = {}
        print("Warning: Could not load sentence_templates.json, using basic extraction")
    
    extracted = []
    
    for result in results:
        sentence = result.get('sentence', '')
        label = result.get('label')
        
        # Don't include 'sentence' in extracted_fields - we'll use raw sentence from result
        extracted_fields = {'label': label}
        
        if dataset_name == 'german-credit-data':
            # Extract age - pattern: "A {age}-year-old"
            age_match = re.search(r'A\s+(\d+)-year-old', sentence, re.IGNORECASE)
            if age_match:
                extracted_fields['age'] = int(age_match.group(1))
            
            # Extract sex/gender - pattern: "{age}-year-old {sex} applicant"
            sex_match = re.search(r'\d+-year-old\s+(male|female)', sentence, re.IGNORECASE)
            if sex_match:
                extracted_fields['sex'] = sex_match.group(1).lower()
            
            # Extract marital status - pattern: "with {marital_status} status"
            marital_match = re.search(r'with\s+([^,]+?)\s+status', sentence, re.IGNORECASE)
            if marital_match:
                marital_val = marital_match.group(1).strip()
                # Normalize common variations
                if '/' in marital_val:
                    extracted_fields['marital-status'] = marital_val
                else:
                    extracted_fields['marital-status'] = marital_val
            
            # Extract job - pattern: "working as {job}."
            job_match = re.search(r'working as\s+([^.,]+?)(?:\.|,|The)', sentence, re.IGNORECASE)
            if job_match:
                extracted_fields['job'] = job_match.group(1).strip()
            
            # Extract credit amount - pattern: "request is {credit_amount} for"
            amount_match = re.search(r'request is\s+(\d+)', sentence, re.IGNORECASE)
            if amount_match:
                extracted_fields['credit-amount'] = int(amount_match.group(1))
            
            # Extract duration - pattern: "over {duration} months"
            duration_match = re.search(r'over\s+(\d+)\s+months?', sentence, re.IGNORECASE)
            if duration_match:
                extracted_fields['duration'] = int(duration_match.group(1))
            
            # Extract purpose - pattern: "for {purpose} over"
            purpose_match = re.search(r'for\s+([^,]+?)\s+over\s+\d+\s+months?', sentence, re.IGNORECASE)
            if purpose_match:
                extracted_fields['purpose'] = purpose_match.group(1).strip()
            
            # Extract checking account - pattern: "checking account {checking_account} and"
            checking_match = re.search(r'checking account\s+([^,]+?)(?:\s+and\s+savings|\.|,|The)', sentence, re.IGNORECASE)
            if checking_match:
                extracted_fields['checking-account'] = checking_match.group(1).strip()
            
            # Extract savings account - pattern: "savings account {savings_account}."
            savings_match = re.search(r'savings account\s+([^,]+?)(?:\.|,|The)', sentence, re.IGNORECASE)
            if savings_match:
                extracted_fields['savings-account'] = savings_match.group(1).strip()
            
            # Extract credit history - pattern: "credit history {credit_history},"
            history_match = re.search(r'credit history\s+([^,]+?)(?:,|\.|employment)', sentence, re.IGNORECASE)
            if history_match:
                extracted_fields['credit-history'] = history_match.group(1).strip()
            
            # Extract employment since - pattern: "employment since {employment_since},"
            employment_match = re.search(r'employment since\s+([^,]+?)(?:,|\.|\d+\s+installment)', sentence, re.IGNORECASE)
            if employment_match:
                extracted_fields['employment-since'] = employment_match.group(1).strip()
            
            # Extract installment rate - pattern: "{installment_rate} installment rate"
            rate_match = re.search(r'(\d+)\s+installment rate', sentence, re.IGNORECASE)
            if rate_match:
                extracted_fields['installment-rate'] = int(rate_match.group(1))
            
            # Extract other debtors - pattern: "other debtors {other_debtors}."
            debtors_match = re.search(r'other debtors\s+([^,]+?)(?:\.|,|Current)', sentence, re.IGNORECASE)
            if debtors_match:
                debtors_val = debtors_match.group(1).strip().lower()
                # Normalize: "no" -> "none"
                if debtors_val == 'no':
                    extracted_fields['other-debtors'] = 'none'
                else:
                    extracted_fields['other-debtors'] = debtors_match.group(1).strip()
            
            # Extract residence since - pattern: "residence since {residence_since} years"
            residence_match = re.search(r'residence since\s+(\d+)\s+years?', sentence, re.IGNORECASE)
            if residence_match:
                extracted_fields['residence-since'] = int(residence_match.group(1))
            
            # Extract property - pattern: "property {property},"
            property_match = re.search(r'property\s+([^,]+?)(?:,|\.|other\s+installment)', sentence, re.IGNORECASE)
            if property_match:
                extracted_fields['property'] = property_match.group(1).strip()
            
            # Extract other installment - pattern: "other installment plans {other_installment},"
            other_installment_match = re.search(r'other installment plans\s+([^,]+?)(?:,|\.|housing)', sentence, re.IGNORECASE)
            if other_installment_match:
                other_inst_val = other_installment_match.group(1).strip().lower()
                # Normalize: "no" -> "none"
                if other_inst_val == 'no':
                    extracted_fields['other-installment'] = 'none'
                else:
                    extracted_fields['other-installment'] = other_installment_match.group(1).strip()
            
            # Extract housing - pattern: "housing {housing},"
            housing_match = re.search(r'housing\s+([^,]+?)(?:,|\.|and\s+\d+\s+existing)', sentence, re.IGNORECASE)
            if housing_match:
                extracted_fields['housing'] = housing_match.group(1).strip()
            
            # Extract existing credits - pattern: "{existing_credits} existing credits"
            credits_match = re.search(r'(\d+)\s+existing credits?', sentence, re.IGNORECASE)
            if credits_match:
                extracted_fields['existing-credits'] = int(credits_match.group(1))
            
            # Extract number-people-provide-maintenance-for - pattern: "supports {num_people} dependent"
            num_people_match = re.search(r'supports\s+(\d+)\s+dependent', sentence, re.IGNORECASE)
            if num_people_match:
                extracted_fields['number-people-provide-maintenance-for'] = int(num_people_match.group(1))
            
            # Extract telephone - pattern: "{telephone} telephone access"
            if re.search(r'have\s+no\s+telephone', sentence, re.IGNORECASE):
                extracted_fields['telephone'] = 'none'
            elif re.search(r'have\s+telephone\s+access', sentence, re.IGNORECASE):
                extracted_fields['telephone'] = 'yes'
            
            # Extract foreign worker - pattern: "has {foreign_worker} foreign-worker status"
            if re.search(r'has\s+a\s+foreign-worker\s+status', sentence, re.IGNORECASE):
                extracted_fields['foreign-worker'] = 'yes'
            elif re.search(r'has\s+not\s+a\s+foreign-worker', sentence, re.IGNORECASE):
                extracted_fields['foreign-worker'] = 'no'
            elif re.search(r'foreign-worker\s+status\s+a\b', sentence, re.IGNORECASE):
                extracted_fields['foreign-worker'] = 'yes'
            
            # Map label to class-label (as in original CSV)
            if 'label' in extracted_fields:
                extracted_fields['class-label'] = extracted_fields.pop('label')
        
        elif dataset_name == 'adult':
            # Extract age - pattern: "A {age}-year-old" or "{age} year old"
            age_match = re.search(r'(?:A\s+)?(\d+)(?:-year-old|\s+year\s+old)', sentence, re.IGNORECASE)
            if age_match:
                extracted_fields['age'] = int(age_match.group(1))
            
            # Extract gender - pattern: "{gender} with race" or "{gender} person"
            if re.search(r'\bmale\b', sentence, re.IGNORECASE):
                extracted_fields['gender'] = 'Male'
            elif re.search(r'\bfemale\b', sentence, re.IGNORECASE):
                extracted_fields['gender'] = 'Female'
            
            # Extract race - pattern: "race {race},"
            race_match = re.search(r'race\s+([^,]+?)(?:,|\.|and)', sentence, re.IGNORECASE)
            if race_match:
                extracted_fields['race'] = race_match.group(1).strip()
            
            # Extract marital status - pattern: "marital status {marital_status},"
            marital_match = re.search(r'marital status\s+([^,]+?)(?:,|\.|and)', sentence, re.IGNORECASE)
            if marital_match:
                extracted_fields['marital-status'] = marital_match.group(1).strip()
            
            # Extract relationship - pattern: "relationship {relationship}."
            relationship_match = re.search(r'relationship\s+([^,]+?)(?:,|\.|and)', sentence, re.IGNORECASE)
            if relationship_match:
                extracted_fields['relationship'] = relationship_match.group(1).strip()
            
            # Extract occupation - pattern: "works as {occupation} in"
            occupation_match = re.search(r'works as\s+([^,]+?)(?:\s+in|,|\.|and)', sentence, re.IGNORECASE)
            if occupation_match:
                extracted_fields['occupation'] = occupation_match.group(1).strip()
            
            # Extract workclass - pattern: "in {workclass},"
            workclass_match = re.search(r'in\s+([^,]+?)(?:\s+with|,|\.|and)', sentence, re.IGNORECASE)
            if workclass_match:
                extracted_fields['workclass'] = workclass_match.group(1).strip()
            
            # Extract education - pattern: "with {education} education"
            education_match = re.search(r'with\s+([^,]+?)\s+education', sentence, re.IGNORECASE)
            if education_match:
                extracted_fields['education'] = education_match.group(1).strip()
            
            # Extract educational-num - pattern: "level {educational_num}"
            edu_num_match = re.search(r'level\s+(\d+)', sentence, re.IGNORECASE)
            if edu_num_match:
                extracted_fields['educational-num'] = int(edu_num_match.group(1))
            
            # Extract hours-per-week - pattern: "working {hours_per_week} hours"
            hours_match = re.search(r'working\s+(\d+)\s+hours', sentence, re.IGNORECASE)
            if hours_match:
                extracted_fields['hours-per-week'] = int(hours_match.group(1))
            
            # Extract capital-gain - pattern: "capital gain is {capital_gain},"
            gain_match = re.search(r'capital gain is\s+(\d+)', sentence, re.IGNORECASE)
            if gain_match:
                extracted_fields['capital-gain'] = int(gain_match.group(1))
            
            # Extract capital-loss - pattern: "capital loss is {capital_loss},"
            loss_match = re.search(r'capital loss is\s+(\d+)', sentence, re.IGNORECASE)
            if loss_match:
                extracted_fields['capital-loss'] = int(loss_match.group(1))
            
            # Extract native-country - pattern: "native country is {native_country},"
            country_match = re.search(r'native country is\s+([^,]+?)(?:,|\.|and)', sentence, re.IGNORECASE)
            if country_match:
                extracted_fields['native-country'] = country_match.group(1).strip()
            
            # Extract fnlwgt - pattern: "fnlwgt is {fnlwgt}."
            fnlwgt_match = re.search(r'fnlwgt is\s+(\d+)', sentence, re.IGNORECASE)
            if fnlwgt_match:
                extracted_fields['fnlwgt'] = int(fnlwgt_match.group(1))
            
            # Map label to class-label (as in original CSV)
            if 'label' in extracted_fields:
                extracted_fields['class-label'] = extracted_fields.pop('label')
        
        extracted.append(extracted_fields)
    
    return extracted


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data using Google Gemini API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using Gemini API with API key from .env file
  python generate_synthetic_data_gemini.py \\
      --dataset german-credit-data \\
      --csv-file sampled_data/german-credit-data/sampled_sentences_combined.csv \\
      --output-file synthetic_data.json \\
      --n-examples 10 \\
      --model gemini-2.0-flash-exp

  # With explicit API key
  python generate_synthetic_data_gemini.py \\
      --dataset german-credit-data \\
      --csv-file sampled_data/german-credit-data/sampled_sentences_combined.csv \\
      --model gemini-2.0-flash-exp \\
      --api-key YOUR_GOOGLE_API_KEY

Note: Create a .env file in the project root with:
  GOOGLE_API_KEY=your_api_key_here
  or
  GEMINI_API_KEY=your_api_key_here
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
        help="Number of valid data samples to generate (default: 1). Will generate more if validation rejects samples."
    )
    parser.add_argument(
        "--max-samples-per-call",
        type=int,
        default=100,
        help="Maximum number of samples to request per API call (default: 100)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3-pro-preview",
        help="Model name (default: gemini-3-pro-preview). Examples: gemini-3-pro-preview, gemini-3-flash, gemini-1.5-pro, gemini-1.5-flash"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Google API key (if not provided, will try to get from GOOGLE_API_KEY or GEMINI_API_KEY environment variable or .env file)"
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=2000,
        help="Maximum number of tokens to generate (default: 2000)"
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
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Disable validation (default: validation is enabled)"
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=9,
        help="Number of neighbors for validation (default: 9)"
    )
    parser.add_argument(
        "--majority-threshold",
        type=float,
        default=0.5,
        help="Minimum fraction of neighbors that must have same label (default: 0.5)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model checkpoint (if not specified, will use latest in models/{dataset}/)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate PCA 2D visualization for z_c and z_d of synthetic data"
    )
    
    args = parser.parse_args()
    
    # Generate output directory and file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create folder structure: synthetic_data/{model_name}/
    model_name_clean = args.model.replace("/", "_").replace("-", "_")
    output_base_dir = os.path.join("synthetic_data", model_name_clean)
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Generate file name with timestamp
    json_filename = f"{timestamp}.json"
    
    if args.output_file is None:
        args.output_file = os.path.join(output_base_dir, json_filename)
    
    print("="*70)
    print("SYNTHETIC DATA GENERATION (Google Gemini API)")
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
    
    # Generate samples in batches until we have enough valid samples
    target_samples = args.n_examples
    max_samples_per_call = min(args.max_samples_per_call, 100)  # Cap at 100
    all_combined_data = []
    all_raw_texts = []
    call_count = 0
    
    print(f"\n{'='*70}")
    print(f"GENERATION STRATEGY")
    print(f"{'='*70}")
    print(f"Target: {target_samples} valid samples")
    print(f"Max samples per API call: {max_samples_per_call}")
    print(f"Will generate in batches until enough valid samples are collected")
    print(f"{'='*70}\n")
    
    while len(all_combined_data) < target_samples:
        call_count += 1
        # Always generate max_samples_per_call (or 100) per batch for better efficiency
        # Only reduce if we're very close to target (within 10 samples)
        remaining_needed = target_samples - len(all_combined_data)
        if remaining_needed <= 10:
            samples_to_generate = remaining_needed
        else:
            samples_to_generate = max_samples_per_call
        
        print(f"\n{'='*70}")
        print(f"API Call #{call_count}: Generating {samples_to_generate} samples")
        print(f"{'='*70}")
        print(f"Current valid samples: {len(all_combined_data)}/{target_samples}")
        print(f"Remaining needed: {remaining_needed}")
        
        # Format prompt for this batch
        system_message, user_message = format_prompt(args.dataset, icl_examples, n_samples=samples_to_generate)
        
        # Generate samples
        generated_text = generate_with_gemini(
            system_message=system_message,
            user_message=user_message,
            model=args.model,
            api_key=args.api_key,
            temperature=args.temperature,
            top_p=args.top_p,
            max_output_tokens=args.max_output_tokens,
        )
        all_raw_texts.append(generated_text)
        
        # Parse output to extract all sentences and labels
        print(f"\nParsing generated output...")
        parsed_results = parse_generated_output(generated_text, args.dataset)
        
        # Add metadata to each result
        batch_results = []
        for idx, result in enumerate(parsed_results, 1):
            result['sample_id'] = len(all_combined_data) + idx
            result['raw_generated_text'] = generated_text
            result['call_number'] = call_count
            batch_results.append(result)
        
        print(f"✓ Parsed {len(batch_results)} sentence(s) with labels")
        
        # Extract fields from sentences
        print(f"Extracting fields from sentences...")
        extracted_data = extract_fields_from_sentences(batch_results, args.dataset)
        
        # Combine sentence and extracted fields
        # IMPORTANT: Use 'raw' sentence (with label sentence) for validation, not cleaned sentence
        batch_combined_data = []
        for i, (result, extracted) in enumerate(zip(batch_results, extracted_data), 1):
            # Use raw sentence (with label sentence) if available, otherwise use cleaned sentence
            raw_sentence = result.get('raw', result.get('sentence', ''))
            entry = {
                'sentence': raw_sentence,  # Keep full sentence with label sentence for validation
            }
            # Add extracted fields, but don't overwrite 'sentence' if it's already set
            for key, value in extracted.items():
                if key != 'sentence':  # Don't overwrite sentence with cleaned version
                    entry[key] = value
            
            if 'label' not in entry or entry['label'] is None:
                entry['label'] = result.get('label')
            if 'class-label' not in entry and 'label' in entry:
                entry['class-label'] = entry['label']
            batch_combined_data.append(entry)
        
        # Validate this batch if validation is enabled
        if not args.no_validate:
            print(f"\nValidating batch #{call_count}...")
            try:
                from validate_synthetic_data import validate_synthetic_data, filter_valid_samples
                
                synthetic_sentences = [entry['sentence'] for entry in batch_combined_data]
                
                # Get existing sentences from previous batches (for duplicate checking)
                existing_sentences = [entry['sentence'] for entry in all_combined_data] if all_combined_data else None
                
                validation_results = validate_synthetic_data(
                    synthetic_sentences=synthetic_sentences,
                    dataset_name=args.dataset,
                    k=args.k_neighbors,
                    majority_threshold=args.majority_threshold,
                    model_path=args.model_path,
                    batch_size=32,
                    existing_sentences=existing_sentences
                )
                
                # Filter to keep only valid samples
                batch_combined_data = filter_valid_samples(batch_combined_data, validation_results)
                print(f"  - Batch #{call_count}: {len(batch_combined_data)} valid samples")
                
            except Exception as e:
                print(f"  ⚠ Validation failed for batch #{call_count}: {e}")
                # Keep all samples if validation fails
                import traceback
                traceback.print_exc()
        
        # Add valid samples to collection
        all_combined_data.extend(batch_combined_data)
        
        print(f"\nProgress: {len(all_combined_data)}/{target_samples} valid samples collected")
        
        # If we have enough, stop
        if len(all_combined_data) >= target_samples:
            break
    
    # Trim to exact target if we have more than needed
    if len(all_combined_data) > target_samples:
        print(f"\nTrimming to exact target: {len(all_combined_data)} -> {target_samples}")
        all_combined_data = all_combined_data[:target_samples]
    
    combined_data = all_combined_data
    print(f"\n✓ Total: {len(combined_data)} valid samples collected from {call_count} API call(s)")
    
    # Validation is already done during generation loop
    n_after_validation = len(combined_data)
    n_rejected = 0  # Will be calculated if needed
    
    # Save single JSON file with combined data (only valid samples after validation)
    output_data = {
        'dataset': args.dataset,
        'model': args.model,
        'timestamp': timestamp,
        'n_generated_samples': args.n_examples,
        'n_parsed_samples': len(combined_data),  # Total parsed samples (before trimming)
        'n_validated_samples': n_after_validation if not args.no_validate else None,
        'n_rejected_samples': n_rejected if not args.no_validate else None,
        'validation_enabled': not args.no_validate,
        'k_neighbors': args.k_neighbors if not args.no_validate else None,
        'majority_threshold': args.majority_threshold if not args.no_validate else None,
        'shuffled_icl': not args.no_shuffle,
        'random_state': args.random_state,
        'data': combined_data,  # Array of entries with sentence + all fields (only valid ones if validation enabled)
        'metadata': {
            'raw_generated_texts': all_raw_texts,
            'n_api_calls': call_count
        }
    }
    
    # Save JSON file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved JSON to: {args.output_file}")
    print(f"  - {len(combined_data)} entries")
    if combined_data:
        sample_entry = combined_data[0]
        print(f"  - Fields per entry: {len(sample_entry)} fields")
        print(f"  - Sample fields: {', '.join(list(sample_entry.keys())[:10])}...")
    
    # Save CSV file in original format (only columns from original CSV)
    if combined_data:
        print(f"\nConverting to original CSV format...")
        csv_df = convert_to_original_csv_format(combined_data, args.dataset)
        
        # Generate CSV filename (same as JSON but with .csv extension)
        csv_file = args.output_file.replace('.json', '.csv')
        csv_df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"✓ Saved CSV to: {csv_file}")
        print(f"  - {len(csv_df)} rows")
        print(f"  - {len(csv_df.columns)} columns (matching original CSV format)")
        print(f"  - Columns: {', '.join(list(csv_df.columns)[:10])}...")
    
    # Visualize if requested
    if args.visualize and combined_data:
        print(f"\n{'='*70}")
        print("GENERATING VISUALIZATION")
        print(f"{'='*70}")
        try:
            from validate_synthetic_data import visualize_synthetic_data_pca
            
            synthetic_sentences = [entry['sentence'] for entry in combined_data]
            synthetic_labels = []
            for entry in combined_data:
                label = entry.get('class-label') or entry.get('label')
                if label is None:
                    print(f"⚠ Warning: Entry has no label, skipping visualization")
                    break
                synthetic_labels.append(int(label))
            
            if len(synthetic_labels) == len(combined_data):
                # Create visualization directory in output folder
                vis_dir = os.path.join(os.path.dirname(args.output_file), "visualizations")
                os.makedirs(vis_dir, exist_ok=True)
                
                visualize_synthetic_data_pca(
                    synthetic_sentences=synthetic_sentences,
                    synthetic_labels=synthetic_labels,
                    dataset_name=args.dataset,
                    model_path=args.model_path,
                    save_dir=vis_dir,
                    compare_with_original=True
                )
        except Exception as e:
            print(f"⚠ Visualization failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("GENERATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nResults saved to:")
    print(f"  - JSON: {args.output_file}")
    if combined_data:
        print(f"  - CSV: {csv_file}")
    print(f"Generated {len(combined_data)} valid sample(s) from {call_count} API call(s)")
    print(f"After validation: {len(combined_data)} valid entries")
    print(f"Each entry contains: sentence (natural language) + all extracted fields")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
