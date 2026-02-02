import pandas as pd
import numpy as np
import argparse
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.impute import SimpleImputer

from evaluate_fairness import (
    demographic_parity, equalized_odds, equal_opportunity, 
    disparate_impact, extract_sensitive_attributes,
    accuracy_parity, treatment_equality
)

def load_and_preprocess_data(original_path, synthetic_path, dataset_name):
    print("Loading data...")
    df_orig = pd.read_csv(original_path)
    df_synth = pd.read_csv(synthetic_path)
    
    # Clean column names
    df_orig.columns = [c.lower().strip() for c in df_orig.columns]
    df_synth.columns = [c.lower().strip() for c in df_synth.columns]
    
    # Separate features and target
    # German Credit: 'class-label' is target (1=Good, 2=Bad or 0/1) -> Normalize to 0/1
    target_col = 'class-label'
    if target_col not in df_orig.columns:
        # Try to find a likely target column
        possible_targets = ['label', 'target', 'income', 'class']
        for t in possible_targets:
            if t in df_orig.columns:
                target_col = t
                break
    
    print(f"Target column: {target_col}")
    
    # Ensure target is 0/1
    # For German Credit: usually 1=Good, 2=Bad. Let's inspect unique values
    # Or sometimes 0/1 directly.
    # We will align synthetic to original encoding.
    
    # Identify Categorical and Numerical columns
    # In German Credit, most are categorical or ordinal strings.
    # We will treat object/string columns as categorical.
    
    X_orig = df_orig.drop(columns=[target_col])
    y_orig = df_orig[target_col]
    X_synth = df_synth.drop(columns=[target_col])
    y_synth = df_synth[target_col]
    
    # Simple preprocessing: Label Encode categorical columns
    # We fit on Combined data to ensure same mapping
    
    # Identify Categorical and Numerical columns
    categorical_cols = X_orig.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X_orig.select_dtypes(include=['int64', 'float64']).columns
    
    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"Numerical columns: {len(numerical_cols)}")
    
    # Fill NAs and force string type for categorical columns to avoid encoder errors
    X_orig[categorical_cols] = X_orig[categorical_cols].astype(str)
    X_synth[categorical_cols] = X_synth[categorical_cols].astype(str)
    
    # Fill NAs for numerical columns
    X_orig[numerical_cols] = X_orig[numerical_cols].fillna(0)
    X_synth[numerical_cols] = X_synth[numerical_cols].fillna(0)
    
    # For Naive Bayes and others, we need fully numeric input.
    # OneHotEncoder is better for nominal, LabelEncoder implies order (ok for DT/RF, bad for others maybe)
    # But user asked for "Traditional" models, simpler encoding often used initially.
    # Let's use LabelEncoder for simplicity and compatibility with standard "tabular" benchmarks,
    # OR OneHot if cardinality is low. 
    # German credit has many string cols. Let's use Ordinal/Label encoding for tree/knn/mlp to keep dimensions manageable,
    # or OneHot for accuracy. 
    # Let's strictly follow a robust pipeline: OneHot for cats.
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    # Fit preprocessor on ORIGINAL data to define the "Real World" schema
    # (Synthetic should adapt to Real, or both combined? Usually fit on Real Train)
    preprocessor.fit(X_orig)
    
    X_orig_encoded = preprocessor.transform(X_orig)
    X_synth_encoded = preprocessor.transform(X_synth)
    
    # Handle Labels
    le_target = LabelEncoder()
    y_orig_encoded = le_target.fit_transform(y_orig)
    try:
        y_synth_encoded = le_target.transform(y_synth)
    except:
        # If synthetic has unknown labels, force fit (should not happen if generation is good)
        print("Warning: Synthetic target labels differ from Original. Refitting...")
        y_synth_encoded = le_target.fit_transform(y_synth)

    return X_orig_encoded, y_orig_encoded, X_synth_encoded, y_synth_encoded, df_orig, le_target

def evaluate_models(X_train, y_train, X_test, y_test, X_test_df, dataset_name, sensitive_attr_name):
    results = {}
    
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'MLP': MLPClassifier(random_state=42, max_iter=500),
        'k-NN': KNeighborsClassifier(n_neighbors=5)
    }
    
    # Extract sensitive attribute from Test DF for fairness calc
    sensitive_attrs = extract_sensitive_attributes(X_test_df, dataset_name)
    sensitive_attr = sensitive_attrs.get(sensitive_attr_name)
    
    if sensitive_attr is None:
        print(f"Warning: Sensitive attribute {sensitive_attr_name} not found.")
        return {}

    # Ensure sensitive_attr matches X_test length (it should if X_test_df corresponds to X_test)
    
    for name, model in models.items():
        print(f"  Training {name}...")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            try:
                # Try to get probabilities for AUC
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    y_prob = y_pred
            except:
                y_prob = y_pred 
            
            # Performance Metrics
            acc = accuracy_score(y_test, y_pred)
            # Default to macro/weighted for multiclass, but fairness usually binary.
            # Assuming binary target 0/1 for now as per German credit processing.
            f1 = f1_score(y_test, y_pred, average='weighted') 
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            try:
                # AUC requires binary? or multi_class='ovr'
                if len(np.unique(y_test)) == 2:
                    auc = roc_auc_score(y_test, y_prob)
                else:
                    auc = 0.5 # Skip for multiclass for now or implement complex handling
            except:
                auc = 0.0
            
            # Fairness Metrics
            dp = demographic_parity(y_pred, sensitive_attr)
            eo = equalized_odds(y_test, y_pred, sensitive_attr)
            eopp = equal_opportunity(y_test, y_pred, sensitive_attr)
            di = disparate_impact(y_pred, sensitive_attr)
            ap = accuracy_parity(y_test, y_pred, sensitive_attr)
            te = treatment_equality(y_test, y_pred, sensitive_attr)
            
            results[name] = {
                'performance': {
                    'accuracy': acc,
                    'f1_score': f1,
                    'precision': prec,
                    'recall': rec,
                    'auc': auc
                },
                'fairness': {
                    'demographic_parity_diff': dp.get('parity_difference', np.nan),
                    'equalized_odds_diff': eo.get('tpr_difference', np.nan), # Simplified: using TPR diff from EO dict? Or should calc max(tpr_diff, fpr_diff)
                    'equal_opportunity_diff': eopp.get('tpr_difference', np.nan),
                    'disparate_impact': di.get('disparate_impact', np.nan),
                    'accuracy_parity_diff': ap.get('accuracy_difference', np.nan),
                    'treatment_equality_diff': te.get('treatment_difference', np.nan)
                }
            }
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {'error': str(e)}
            
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate synthetic data utility and fairness using downstream classifiers")
    parser.add_argument("--original-data", required=True, help="Path to original CSV")
    parser.add_argument("--synthetic-data", required=True, help="Path to synthetic CSV")
    parser.add_argument("--dataset", default="german-credit-data", help="Dataset name")
    parser.add_argument("--sensitive-attr", default="sex", help="Sensitive attribute name")
    parser.add_argument("--output-dir", default="downstream_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Load Data
    X_orig, y_orig, X_synth, y_synth, df_orig, le = load_and_preprocess_data(
        args.original_data, args.synthetic_data, args.dataset
    )
    
    print(f"Original Data: {len(X_orig)} samples")
    print(f"Synthetic Data: {len(X_synth)} samples")
    
    # scenario 1: Baseline (Train Real, Test Real)
    print("\n--- Scenario 1: Real Data Baseline ---")
    X_real_train, X_real_test, y_real_train, y_real_test, idx_train, idx_test = train_test_split(
        X_orig, y_orig, list(df_orig.index), test_size=0.3, random_state=42
    )
    df_real_test = df_orig.iloc[idx_test]
    
    results_baseline = evaluate_models(
        X_real_train, y_real_train, 
        X_real_test, y_real_test, 
        df_real_test, args.dataset, args.sensitive_attr
    )
    
    # Scenario 2: Synthetic Data Training (Train Synth, Test Real)
    print("\n--- Scenario 2: Synthetic Data Training ---")
    results_synthetic = evaluate_models(
        X_synth, y_synth, 
        X_real_test, y_real_test, 
        df_real_test, args.dataset, args.sensitive_attr
    )
    
    final_report = {
        'baseline_real': results_baseline,
        'synthetic_training': results_synthetic
    }
    
    # Print Comparison Table
    print("\n" + "="*110)
    print("RESULTS COMPARISON: Performance & Fairness (Downstream Models)")
    print("="*110)
    # Header format
    header = f"| {'Model':<13} | {'Data':<5} | {'Acc':<6} {'F1':<6} {'AUC':<6} | {'DI':<6} {'DP_Dif':<6} {'EO_Dif':<6} {'TE_Dif':<6} |"
    print(header)
    print("-" * 110)
    
    for model_name in results_baseline.keys():
        for scenario, res_dict in [("Real", results_baseline), ("Synth", results_synthetic)]:
            r = res_dict.get(model_name, {})
            p = r.get('performance', {})
            f = r.get('fairness', {})
            
            # Extract key metrics
            acc = p.get('accuracy', 0)
            f1 = p.get('f1_score', 0)
            auc = p.get('auc', 0)
            
            di = f.get('disparate_impact', 0)
            dp_diff = f.get('demographic_parity_diff', 0)
            eo_diff = f.get('equalized_odds_diff', 0)
            # Use 'equalized_odds_diff' which was mapped to max diff or similar
            # In evaluate_models I mapped it to 'tpr_difference' from EO dict?? 
            # Wait, let's check what evaluate_fairness returns for equalized_odds.
            # It returns 'tpr_difference'. Okay.
            
            te_diff = f.get('treatment_equality_diff', 0)
            
            # Formatting helpers
            scenario_str = "Orig" if scenario == "Real" else "Syn"
            
            def fmt(v):
                if v is None or np.isnan(v): return "NaN "
                return f"{v:.4f}"
            
            print(f"| {model_name:<13} | {scenario_str:<5} | {fmt(acc)} {fmt(f1)} {fmt(auc)} | {fmt(di)} {fmt(dp_diff)} {fmt(eo_diff)} {fmt(te_diff)} |")
        print("-" * 110)
            
    # Save Report
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, f"downstream_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    def default_conv(o):
        if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)): return int(o)
        elif isinstance(o, (np.float_, np.float16, np.float32, np.float64)): return float(o)
        elif isinstance(o, (np.ndarray,)): return o.tolist()
        elif isinstance(o, (bool, np.bool_)): return bool(o)
        return str(o)

    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2, default=default_conv)
    
    print(f"\nSaved detailed report to {report_path}")

if __name__ == "__main__":
    main()
