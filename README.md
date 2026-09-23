# SENNA: Semantic Encodings for Fair Tabular Data Generation via LLMs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **SENNA**: A framework for generating fair synthetic tabular data using disentangled representations and LLM-based generation.

🤗 [Quick Start](#quick-start) | 📊 [Datasets](#supported-datasets) | 🔬 [Full Pipeline](#full-pipeline)

---
## Journal Accepted 

The paper titled **"Fair Synthetic Tabular Data Generation via Disentangled Semantic Encoding and Large Language Models"** has been accepted for **Discover Artificial Intelligent (Q1)**. Congratulation my team.  

## 📰 Abstract

SENNA tackles the challenge of generating **fair synthetic data** for sensitive tabular datasets. Traditional synthetic data generators often fail to preserve fairness properties or produce representative samples across demographic groups. Our approach combines:

1.  **Disentangled Representation Learning** - Separating task-relevant (z_c) from task-independent (z_d) information using BERT and adversarial training
2.  **LLM-based Generation with In-Context Learning** - Using Google Gemini to generate diverse, high-quality samples
3.  **Multi-level Fairness Validation** - Ensuring both individual fairness (k-NN validation) and group fairness (demographic balance)
4.  **Natural Language Interface** - Converting tabular data to text for LLM processing, then deconverting with validation

![Full Pipeline](/assets/full-pipeline.png)

---

## 🎯 Key Features

-   ✨ **Disentangled Representations**: Learn z_c (task-relevant) and z_d (task-independent) spaces using BERT encoder
-   🎲 **Diverse ICL Sampling**: K-means clustering on z_d ensures diverse in-context learning examples
-   ⚖️ **Fairness Governor**: Automatic demographic balancing across sensitive attributes
-   🔍 **Label Validation**: k-NN on z_c embeddings validates generated samples before acceptance
-   📊 **Comprehensive Metrics**: Statistical Parity, Equalized Odds, Predictive Parity, Theil Index, ABROCA, and quality metrics
-   🚀 **Multiple Baselines**: Includes CTGAN, GREAT, TabFairGAN, CLLM, and TabGEN-ICL for comparison

---

## 🎯 Overview

SENNA tackles the challenge of generating fair synthetic data for tabular datasets. The core idea is to:

1.  **Convert tabular data to natural language sentences** - Making it suitable for LLM processing
2.  **Learn disentangled representations** - Separating task-relevant information from non-task relevant information
3.  **Generate synthetic data using LLMs** - With in-context learning from diverse examples
4.  **Data validation for multiple fairness** - Including individual fairness and group fairness
5.  **Deconvert back to tabular format** - Validate and convert generated text back to structured data

## 🏗️ Architecture

### Disentangled Representation Learning

The framework uses a shared BERT encoder with two separate attention heads:

- **z_c (Task-Relevant Space)**: Captures information needed to predict the label (high accuracy ~85-95%)
- **z_d (Task-Independent Space)**: Captures information independent of the label (accuracy ~50%, random guess)

```
Input Text (sentence)
    ↓
BERT Encoder (bert-base-uncased)
    ↓
Shared BERT Output [batch_size, seq_len, 768]
    ↓
    ├─→ ContentHead (Attention + VAE) → z_c [batch_size, latent_dim]
    │       ↓
    │   Task Classifier → y_pred (high accuracy)
    │
    └─→ DemographicHead (Attention + VAE) → z_d [batch_size, latent_dim]
            ↓
        Adversarial Classifier → y_pred (~50% accuracy)
    
    z_c, z_d
        ↓
    D_c(z_c) + D_d(z_d) → h_reconstructed (additive reconstruction)
```

### Components

1. **Shared BERT Encoder**: Encodes input text into embeddings
2. **ContentHead**: Attention pooling + VAE → `z_c` (task-relevant information)
3. **DemographicHead**: Attention pooling + VAE → `z_d` (task-independent information)
4. **Task Classifier**: Predicts label from `z_c` (target: high accuracy)
5. **Adversarial Classifier**: Predicts label from `z_d` (target: ~50% accuracy)
6. **Decoders**: `D_c(z_c)` and `D_d(z_d)` for reconstruction

---

## 📁 Project Structure

```
SENNA/
├── scripts/                      # Main scripts
│   ├── train.py                  # Train SENNA model with BERT encoder
│   ├── generate.py               # Generate synthetic data with Gemini API
│   ├── cluster.py                # K-means clustering on z_d embeddings
│   ├── validate.py               # Validate synthetic data with k-NN on z_c
│   ├── embed.py                  # Extract z_c and z_d representations
│   └── hyperparameter_tuning.py  # Grid/Random search for hyperparameters
├── src/                          # Source modules
│   ├── data_loader.py            # Load and preprocess datasets
│   ├── prompts.py                # Prompts for LLM generation
│   ├── sentence_templates.json   # Templates for converting tabular to text
│   └── visualization.py          # Visualization utilities
├── components/                   # Core model components
│   ├── models.py                 # CD_Model and layers
│   ├── dataset.py                # TextDataset class
│   ├── losses.py                 # HSIC, KL, Reconstruction losses
│   └── training.py               # Training loop utilities
├── fairness_evaluation/          # Fairness evaluation
│   ├── exp_german_credit.ipynb   # Experiment notebook
│   └── my_useful_functions.py    # Fairness metrics (SP, EO, EOpp, etc.)
├── data/                         # Raw datasets (5 datasets)
├── processed_data/               # Preprocessed text data
├── models/                       # Saved model checkpoints
└── sampled_data/                 # Clustered samples for ICL
```

---

## 📐 Loss Functions

### 1. Task Loss (L_task)

**Objective**: `z_c` should predict label `y` well

$$L_{task} = CrossEntropy(f_{task}(z_c), y)$$

### 2. Independence Loss (L_adv) - HSIC(z_d, y)

**Objective**: `z_d` should be independent of label `y`

$$L_{adv} = HSIC(z_d, y)$$

### 3. Orthogonality Loss (L_ortho) - HSIC(z_c, z_d)

**Objective**: `z_c` and `z_d` should be statistically independent

$$L_{ortho} = HSIC(z_c, z_d)$$

### 4. Reconstruction Loss (L_rec)

**Objective**: Reconstruct BERT embeddings from both spaces

$$L_{rec} = ||D_c(z_c) + D_d(z_d) - h||^2$$

### 5. KL Divergence Loss (L_kl)

**Objective**: Regularize latent distributions to standard normal

$$L_{kl} = KL(q(z_c|x) || N(0,I)) + KL(q(z_d|x) || N(0,I))$$

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt

# Create .env file with Gemini API key
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### Step 2: Train the Disentanglement Model

```bash
python scripts/train.py \
    --dataset german-credit-data \
    --epochs 100 \
    --batch-size 64 \
    --lambda-task 3.0 \
    --lambda-adv 0.1 \
    --lambda-ortho 1.0
```

**Output**: Model saved to `models/{dataset}/best_model_*.pth`

### Step 3: Cluster Data based on z_d

```bash
python scripts/cluster.py \
    --dataset german-credit-data \
    --output-dir sampled_data \
    --n-examples 10
```

**Output**: Sampled sentences saved to `sampled_data/{dataset}/*/sampled_sentences_combined.csv`

### Step 4: Generate Synthetic Data

```bash
python scripts/generate.py \
    --dataset german-credit-data \
    --csv-file sampled_data/german-credit-data/*/sampled_sentences_combined.csv \
    --n-examples 1000 \
    --model gemini-2.0-flash-exp
```

**Notes**:
- `--n-examples`: Number of synthetic samples to generate
- All examples in CSV are used as in-context examples (shuffled)
- Clustering ensures diversity in ICL examples

 Generate with Fairness Constraints

```bash
python scripts/generate.py \
    --dataset german-credit-data \
    --csv-file sampled_data/german-credit-data/*/sampled_sentences_combined.csv \
    --n-examples 1000 \
    --fair \
    --sensitive-attrs sex
    --visualization
```

**Fairness Features**:
- **Fairness Governor**: Automatically balances output across demographic groups
- **Batch-level Label Balancing**: Ensures 50/50 label ratio after validation
- **Post-generation Balancing**: Balances sensitive attributes within each label group

---

## ⚙️ Parameters

### Full Configuration Used in the Paper

This is the exact configuration behind every SENNA number reported in the paper.
Values are read from the defaults in the scripts listed in the last column; the
only flags overridden at run time are `--dataset`, `--n-examples`, and `--model`.

**Encoder and latent space**

| Component | Value | Source |
|-----------|-------|--------|
| BERT encoder | `bert-base-uncased` (768-d), fully fine-tuned | [components/models.py](components/models.py) |
| Max sequence length | 256 tokens | [components/tokenization.py](components/tokenization.py) |
| Latent dim of z_c, z_d | 128 each | `--latent-dim` |
| Pooling | learned attention over tokens, stop words masked | [components/models.py](components/models.py) |
| VAE bottleneck | 768 → 256 → 128, reparameterized | [components/models.py](components/models.py) |

**Loss weights**

| Component | Value | Source |
|-----------|-------|--------|
| λ_task | 8.0 | `--lambda-task` |
| λ_adv — HSIC(z_d, y) | 1.0 | `--lambda-adv` |
| λ_ortho — HSIC(z_c, z_d) | 0.1 | `--lambda-ortho` |
| λ_rec | 0.5 | `--lambda-rec` |
| λ_kl | 0.01 | `--lambda-kl` |
| λ_var — anti-collapse on z_d | 0.1 | `--lambda-var` |
| HSIC kernel | RBF | [components/losses.py](components/losses.py) |
| HSIC bandwidth | median heuristic, recomputed per mini-batch and per argument | [components/losses.py](components/losses.py) |

**Encoder training**

| Component | Value | Source |
|-----------|-------|--------|
| Optimizer (encoder, heads, decoders) | AdamW, weight decay 0.01 | [components/training.py](components/training.py) |
| Learning rate | 2e-5 (BERT), 4e-5 (heads/decoders) | `--lr`, ×2 for heads |
| Optimizer (classifiers) | Adam, lr 2e-4 | [components/training.py](components/training.py) |
| LR schedule | ReduceLROnPlateau (×0.5, patience 5, floor 1e-6) | [components/training.py](components/training.py) |
| Batch size B | 64 (auto-selected from GPU memory) | `--batch-size` |
| Epochs T | 100 | `--epochs` |
| Precision | mixed (AMP) on GPU | `--use-amp` |

**Exemplar selection**

| Component | Value | Source |
|-----------|-------|--------|
| Clustering space | z_d, clustered separately per class label | [scripts/cluster.py](scripts/cluster.py) |
| Number of clusters K | selected by silhouette over K ∈ [2, 20]; **K = 2** for every dataset–label group | `--method silhouette` |
| Representative-selection rule | uniform random draw **without replacement** within each cluster | [scripts/cluster.py](scripts/cluster.py) |
| Exemplars per cluster | 100 (Adult, Credit-Scoring); 20 (German Credit) | `--n-examples` |
| Resulting exemplar pool | 400 / 400 / 80 records | — |

**Generation and validation**

| Component | Value | Source |
|-----------|-------|--------|
| Decoding temperature | 0.7 | `--temperature` |
| Decoding top-p | 0.9 | `--top-p` |
| Max output tokens | 8192 | `--max-output-tokens` |
| Samples requested per call | 100 (50 via OpenRouter) | `--max-samples-per-call` |
| Concurrent workers | 8 | [scripts/generate.py](scripts/generate.py) |
| k in kNN consistency validation | 9 | `--k-neighbors` |
| Majority threshold τ | 0.5 | `--majority-threshold` |
| Retries | 3, **transport-level failures only**. A candidate rejected by the validator is discarded, not regenerated; additional batches are scheduled until the target count is met | [scripts/generate.py](scripts/generate.py) |
| Deduplication | exact sentence match, global across batches | [scripts/generate.py](scripts/generate.py) |
| Random seed | 42 | `--random-state` |

### Training (`scripts/train.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lambda-task` | 8.0 | Task loss weight (z_c → y accuracy) |
| `--lambda-adv` | 1.0 | Independence loss weight (HSIC(z_d, y)) |
| `--lambda-ortho` | 0.1 | Orthogonality loss weight (HSIC(z_c, z_d)) |
| `--lambda-rec` | 0.5 | Reconstruction loss weight |
| `--lambda-kl` | 0.01 | KL divergence weight |
| `--lambda-var` | 0.1 | Variance constraint on z_d (prevents collapse) |
| `--latent-dim` | 128 | Latent dimension for z_c and z_d |
| `--batch-size` | auto | Batch size (auto-detected from GPU memory) |
| `--epochs` | 100 | Training epochs |
| `--lr` | 2e-5 | Learning rate (BERT; heads use 2×) |

### Generation (`scripts/generate.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-examples` | 1 | Number of samples to generate |
| `--temperature` | 0.7 | Sampling temperature |
| `--top-p` | 0.9 | Nucleus sampling parameter |
| `--max-output-tokens` | 8192 | Max tokens per API response |
| `--max-samples-per-call` | 1000 | Samples per API call (capped to 100, or 50 via OpenRouter) |
| `--fair` | False | Enable fairness features |
| `--sensitive-attrs` | "sex" | Sensitive attribute for Fairness Governor |
| `--k-neighbors` | 9 | k-NN neighbors for label validation |
| `--majority-threshold` | 0.5 | Min fraction of neighbors sharing the label |
| `--no-shuffle` | False | Don't shuffle ICL examples |
| `--model` | gemini-3-pro-preview | LLM model to use |
| `--provider` | gemini | `gemini` or `openrouter` |

### Clustering (`scripts/cluster.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-examples` | 10 | Number of examples sampled per cluster |
| `--k` | None | Fixed K; if unset, K is searched |
| `--min-k` | 2 | Minimum K for K-means search |
| `--max-k` | 20 | Maximum K for K-means search |
| `--method` | silhouette | Criterion for selecting K (`silhouette`, `elbow`, `davies_bouldin`) |
| `--random-state` | 42 | Random seed |
| `--resample-only` | False | Only resample, don't re-cluster |

---

## 📊 Supported Datasets

| Dataset | Samples | Sensitive Attr | Label |
|---------|---------|----------------|-------|
| German Credit | 1,000 | sex, age | Good/Bad credit risk |
| Adult Income | 48,842 | sex, race | Income >50K / <=50K |
| Credit Scoring | 8,755 | sex | Approved / Rejected |

---

## 🔄 Full Pipeline

SENNA follows a 4-stage pipeline:

### Stage 1: Data Preparation & Representation Learning

1. **Data Loading**: Load tabular dataset and convert to natural language sentences using templates
2. **Model Training** ([`scripts/train.py`](scripts/train.py)): Train the disentanglement model
   - Learn z_c (task-relevant) and z_d (task-independent) representations
   - Use BERT encoder with dual attention heads
   - Optimize with HSIC loss for independence

### Stage 2: Diverse Example Sampling

3. **K-means Clustering** ([`scripts/cluster.py`](scripts/cluster.py)): Cluster on z_d embeddings
   - Ensures diversity in demographic representations
   - Sample representatives from each cluster
   - Output: ICL examples for generation

### Stage 3: Synthetic Data Generation

4. **LLM Generation** ([`scripts/generate.py`](scripts/generate.py)): Generate synthetic samples using Gemini API
   - **Prompt Construction**: Build prompts with ICL examples
   - **Batch Generation**: Generate samples in batches
   - **Sentence Parsing**: Extract structured data from text
   - **Duplicate Checking**: Remove duplicates against original data
   - **Label Validation**: k-NN validation on z_c embeddings (see [Validation Process](#validation-process))
   - **Fairness Balancing**: Optional demographic balancing via FairnessGovernor

### Stage 4: Output & Evaluation

5. **Data Export**: Save validated samples in original CSV format
6. **Fairness Evaluation** ([`fairness_evaluation/`](fairness_evaluation/)): Compute metrics

---

## 🎲 Generation Process

![SENNA Generation Process](/assets/generation.png)

The generation process in [`scripts/generate.py`](scripts/generate.py) implements a sophisticated pipeline with the following key components:

### Key Steps

1. **Prompt Construction**
   - Load ICL examples from clustered data
   - Format system and user messages with dataset-specific templates
   - Include optional fairness instructions

2. **Batch Generation Loop**
   - Generate samples using Gemini API (or other LLM providers)
   - Parse structured output from natural language
   - Extract attributes and labels from generated sentences

3. **Multi-stage Validation**
   - **Duplicate Checking**: Against original data and previous batches
   - **Label Validation**: k-NN validation on z_c embeddings (see [Validation Process](#validation-process))
   - **Structure Validation**: Ensure all required fields are present

4. **Fairness Balancing** (Optional with `--fair`)
   - Track demographic distribution via `FairnessGovernor`
   - Guide generation toward underrepresented groups
   - Perform final balancing across sensitive attributes

### Output Formats

Generated data is saved in multiple formats:

- **Original CSV format**: `synthetic_data/{dataset}/{model}/{timestamp}_synthetic.csv`
- **With sentences**: `{timestamp}_synthetic_with_sentences.csv`
- **Metadata**: `{timestamp}_metadata.json` (includes generation stats, validation rates)

---

## ✅ Validation Process

![SENNA Validation Process](/assets/validation.png)

Validation in [`scripts/validate.py`](scripts/validate.py) ensures generated samples are both realistic and fair through multiple checks:

### Individual Fairness: k-NN Label Validation

The core validation mechanism uses k-nearest neighbors on z_c embeddings:

**Process:**
1. **Extract z_c embedding** for generated sample (without label description)
2. **Find k=9 nearest neighbors** in original data using cosine distance on z_c
3. **Check majority label** of neighbors
4. **Accept sample** only if majority_ratio > 0.5 (strict majority)

This ensures **individual fairness**: samples with similar task-relevant features (z_c) should have similar labels.

### Validation Pipeline

**Step 1: Label Extraction**
- Parse label from generated sentence using regex patterns
- Handle dataset-specific formats (e.g., "good credit risk" vs "income >50K")

**Step 2: Embedding Extraction**
- Remove label description from sentence
- Extract z_c embedding using trained SENNA model
- Use cached embeddings for original data

**Step 3: k-NN Label Check**
- Compute cosine distances to all original samples
- Select k=9 nearest neighbors
- Verify generated label matches majority

**Step 4: Duplicate Detection**
- Check against original dataset
- Check within current batch
- Check against previously generated samples

**Step 5: Structure Validation**
- Ensure all required attributes are present
- Verify data types and value ranges

### Validation Assets Caching

To speed up validation, original data embeddings are cached:

```
cache/embeddings/{dataset}/z_c_z_d_{model_hash}.pkl
├── z_c: [n_original, latent_dim]    # Task-relevant embeddings
├── z_d: [n_original, latent_dim]    # Task-independent embeddings  
├── sentences: [n_original]          # Original sentences
└── labels: [n_original]             # Original labels
```

---

## 🔬 Baseline Methods

SENNA is compared against state-of-the-art synthetic data generation methods:

### Generative Adversarial Networks
- **CTGAN** ([`scripts/generate_ctgan.py`](scripts/generate_ctgan.py)): Conditional Tabular GAN
- **TabFairGAN** ([`scripts/generate_tabfairgan.py`](scripts/generate_tabfairgan.py)): Fairness-aware GAN for tabular data

### LLM-based Methods
- **GREAT** ([`scripts/generate_great.py`](scripts/generate_great.py)): GeneRation and Editing of data with Language Models
- **TabGEN-ICL** ([`scripts/generate_tabgen.py`](scripts/generate_tabgen.py)): In-context learning without disentanglement
- **CLLM** ([`scripts/generate_cllm.py`](scripts/generate_cllm.py)): Causal Language Models

### Fairness-aware Methods
- **FLAi** ([`scripts/generate_flai.py`](scripts/generate_flai.py)): Fair Learning AI

All baseline methods can be run with similar commands:
```bash
python scripts/generate_<method>.py --dataset <dataset_name> --n-examples 1000
```

---

## ⚖️ Fairness Evaluation

Comprehensive fairness evaluation scripts for all datasets:

### Evaluation Scripts

| Script | Dataset | Metrics |
|--------|---------|---------|
| [`fairness_evaluation/german-credit-dataset.py`](fairness_evaluation/german-credit-dataset.py) | German Credit | F1, SP, EO, PP, Theil, ABROCA |
| [`fairness_evaluation/adult-dataset.py`](fairness_evaluation/adult-dataset.py) | Adult Income | F1, SP, EO, PP, Theil, ABROCA |
| [`fairness_evaluation/credit-scoring-dataset.py`](fairness_evaluation/credit-scoring-dataset.py) | Credit Scoring | F1, SP, EO, PP, Theil, ABROCA |

### Fairness Metrics

Implemented in [`fairness_evaluation/my_useful_functions.py`](fairness_evaluation/my_useful_functions.py):

- **Statistical Parity (SP)**: P(Ŷ=1|A=0) - P(Ŷ=1|A=1)
- **Equalized Odds (EO)**: |TPR₀ - TPR₁| + |FPR₀ - FPR₁|
- **Equal Opportunity**: |TPR₀ - TPR₁|
- **Predictive Parity (PP)**: |PPV₀ - PPV₁|
- **Disparate Impact**: P(Ŷ=1|A=0) / P(Ŷ=1|A=1)
- **Theil Index**: Entropy-based measure of inequality
- **ABROCA**: Absolute Between-ROC Area ([`fairness_evaluation/compute_abroca.py`](fairness_evaluation/compute_abroca.py))

### Quality Metrics

Implemented in [`fairness_evaluation/quality_metrics.py`](fairness_evaluation/quality_metrics.py):

- **Density**: Fraction of real data neighbors around synthetic samples
- **Coverage**: Fraction of real data covered by synthetic data
- **Statistical Similarity**: KL divergence, Wasserstein distance
- **Utility**: ML efficacy on downstream tasks

### Running Fairness Evaluation

```bash
# Evaluate German Credit
python fairness_evaluation/german-credit-dataset.py

# Evaluate Adult Income
python fairness_evaluation/adult-dataset.py

# Evaluate Credit Scoring
python fairness_evaluation/credit-scoring-dataset.py
```

### Fairness-aware Training Methods

The framework includes implementations of various fairness-aware ML algorithms in [`fairness_evaluation/`](fairness_evaluation/):

- **AdaFair** ([`AdaFair.py`](fairness_evaluation/AdaFair.py)): Adaptive Fair Classification
- **Exponentiated Gradient Reduction** ([`exponentiated_gradient_reduction.py`](fairness_evaluation/exponentiated_gradient_reduction.py)): Constraints-based fairness

---

## 📈 Performance Metrics

### Disentanglement Quality
- **Task Accuracy (z_c → y)**: ~85-95% (Task-Relevant captures predictive info)
- **Adversarial Accuracy (z_d → y)**: ~45-55% (Task-Independent is independent)
- **HSIC(z_d, y) & HSIC(z_c, z_d)**: < 0.01 (Statistical independence)

### Fairness Metrics
- **Statistical Parity**: P(Ŷ=1|A=0) - P(Ŷ=1|A=1)
- **Equalized Odds**: |TPR₀ - TPR₁| + |TNR₀ - TNR₁|
- **Equal Opportunity**: |TPR₀ - TPR₁|
- **Disparate Impact**: P(Ŷ=1|A=0) / P(Ŷ=1|A=1)

---

## 📚 References

- HSIC: Hilbert-Schmidt Independence Criterion
- VAE: Variational Autoencoder
- BERT: Bidirectional Encoder Representations from Transformers
- Gemini API: Google's Large Language Model

---

## 📊 Visualization & Analysis

### Reproducing Paper Figures

```bash
# Reproduce plots from existing results
python reproduce_plot.py
```

### Visualization Utilities

The framework includes visualization tools in [`src/visualization.py`](src/visualization.py) and [`fairness_evaluation/utils.py`](fairness_evaluation/utils.py):

- **ABROCA Plots**: ROC curves comparison between demographic groups
- **Fairness Metrics Dashboard**: Bar charts comparing different methods
- **Distribution Comparisons**: Real vs synthetic data distributions
- **Embedding Visualizations**: t-SNE/UMAP of z_c and z_d spaces

### Regenerating Visualizations

```bash
# Regenerate visualizations from existing data
python scripts/regenerate_visualizations.py \
    --dataset german-credit-data \
    --output-dir visualizations/
```

---

## 🔧 Advanced Usage

### Hyperparameter Tuning

Use grid search or random search to find optimal hyperparameters:

```bash
python scripts/hyperparameter_tuning.py \
    --dataset german-credit-data \
    --search-type grid \
    --n-trials 50
```

The script will search over:
- `lambda_task`: [1.0, 3.0, 5.0]
- `lambda_adv`: [0.05, 0.1, 0.5]
- `lambda_ortho`: [0.5, 1.0, 2.0]
- `latent_dim`: [16, 32, 64]

Results are saved to `tuning_results/{dataset}/`.

### Extracting Embeddings

Extract z_c and z_d embeddings for analysis:

```bash
python scripts/embed.py \
    --dataset german-credit-data \
    --output-file embeddings.pkl
```

Output format: `{'z_c': array, 'z_d': array, 'labels': array, 'texts': list}`

---

## 🐛 Troubleshooting

### Common Issues

**Issue 1: `ZeroDivisionError` in fairness metrics**
- **Cause**: Imbalanced data with no samples in certain demographic groups
- **Fix**: Check your data has sufficient samples per group, or adjust fairness metric implementations

**Issue 2: Model hangs during MLPClassifier training**
- **Cause**: Features not scaled properly
- **Fix**: Ensure features are standardized using `StandardScaler` before training

**Issue 3: ABROCA plotting fails**
- **Cause**: `inclusive` parameter error in `pd.between()`
- **Fix**: Update to pandas >= 1.5.0 which supports string values for `inclusive`

**Issue 4: Gemini API rate limits**
- **Cause**: Too many requests in short time
- **Fix**: Add `--delay` parameter to generation scripts, or use batch processing

**Issue 5: Low diversity in generated samples**
- **Cause**: Insufficient clustering or poor ICL examples
- **Fix**: Increase `--k-max` in clustering, or use more diverse seed data

### Debug Mode

Enable verbose logging:

```bash
export DEBUG=1
python scripts/generate.py --dataset german-credit-data --n-examples 100
```

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📋 Requirements

See [`requirements.txt`](requirements.txt) for full dependencies. Key requirements:

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Google GenAI 0.2.0+
- scikit-learn, pandas, numpy
- For fairness evaluation: pomegranate, networkx, aif360

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🙏 Acknowledgments

- BERT pre-trained models from [Hugging Face](https://huggingface.co/)
- Gemini API from [Google AI](https://ai.google.dev/)
- Fairness metrics implementations inspired by [AIF360](https://github.com/Trusted-AI/AIF360)
- Baseline methods: CTGAN, GREAT, TabFairGAN, CLLM, FLAi

---

---

**Star ⭐ this repo if you find it useful!**
