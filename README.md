# SENNA

A framework for generating fair synthetic tabular data using disentangled representations and LLM-based generation.

## 🎯 Overview

SENNA tackles the challenge of generating fair synthetic data for tabular datasets. The core idea is to:

1. **Convert tabular data to natural language sentences** - Making it suitable for LLM processing
2. **Learn disentangled representations** - Separating task-relevant information from non-task relevant information
3. **Generate synthetic data using LLMs** - With in-context learning from diverse examples
4. **Data validation for multiple fairness** - Including individual fairness and group fairness
5. **Deconvert back to tabular format** - Validate and convert generated text back to structured data

[View SENNA Architecture (PDF)](SENNA.pdf)

## 🏗️ Architecture

### Disentangled Representation Learning

The framework uses a shared BERT encoder with two separate attention heads:

- **z_c (Task-Relevant Space)**: Captures information needed to predict the label (high accuracy ~85-95%)
- **z_d (Non-Task Relevant Space)**: Captures information independent of the label (accuracy ~50%, random guess)

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
3. **DemographicHead**: Attention pooling + VAE → `z_d` (non-task relevant information)
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

### Step 5: Generate with Fairness Constraints

```bash
python scripts/generate.py \
    --dataset german-credit-data \
    --csv-file sampled_data/german-credit-data/*/sampled_sentences_combined.csv \
    --n-examples 1000 \
    --fair \
    --sensitive-attrs sex
```

**Fairness Features**:
- **Fairness Governor**: Automatically balances output across demographic groups
- **Batch-level Label Balancing**: Ensures 50/50 label ratio after validation
- **Post-generation Balancing**: Balances sensitive attributes within each label group

---

## ⚙️ Parameters

### Training (`scripts/train.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lambda-task` | 3.0 | Task loss weight (z_c → y accuracy) |
| `--lambda-adv` | 0.1 | Adversarial loss weight (HSIC(z_d, y)) |
| `--lambda-ortho` | 1.0 | Orthogonality loss weight (HSIC(z_c, z_d)) |
| `--lambda-rec` | 0.5 | Reconstruction loss weight |
| `--lambda-kl` | 0.1 | KL divergence weight |
| `--latent-dim` | 32 | Latent dimension for z_c and z_d |
| `--epochs` | 100 | Training epochs |
| `--lr` | 2e-5 | Learning rate |

### Generation (`scripts/generate.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-examples` | 1 | Number of samples to generate |
| `--fair` | False | Enable fairness features |
| `--sensitive-attrs` | "sex" | Sensitive attribute for Fairness Governor |
| `--k-neighbors` | 9 | k-NN neighbors for label validation |
| `--no-shuffle` | False | Don't shuffle ICL examples |
| `--model` | gemini-2.0-flash-exp | LLM model to use |

### Clustering (`scripts/cluster.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-examples` | 10 | Number of examples per cluster |
| `--k-min` | 2 | Minimum K for K-means |
| `--k-max` | 10 | Maximum K for K-means |
| `--resample-only` | False | Only resample, don't re-cluster |

---

## 📊 Supported Datasets

| Dataset | Samples | Sensitive Attr | Label |
|---------|---------|----------------|-------|
| German Credit | 1,000 | sex, age | Good/Bad credit risk |
| Adult Income | 48,842 | sex, race | Income >50K / <=50K |
| Bank Marketing | 45,211 | age, marital | Subscribed / Not |
| Credit Scoring | 8,755 | sex | Approved / Rejected |
| Dutch Census | 60,420 | sex | High/Low occupation |

---

## 📈 Evaluation Metrics

### Disentanglement Quality
- **Task Accuracy (z_c → y)**: ~85-95% (Task-Relevant captures predictive info)
- **Adversarial Accuracy (z_d → y)**: ~45-55% (Non-Task Relevant is independent)
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

---