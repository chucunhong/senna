# Fairness Learning with Disentangled Representations (LIRD Framework)

Hệ thống học công bằng (fairness learning) sử dụng biểu diễn tách rời (disentangled representations) từ dữ liệu văn bản tự nhiên. Framework này tách thông tin thành hai thành phần:
- **z_c (Content)**: Thông tin liên quan đến task (có thể dự đoán label tốt)
- **z_d (Demographic)**: Thông tin độc lập với label (không thể dự đoán label, accuracy ~50%)

## 📁 Cấu trúc thư mục

```
Dist_Learn_Fair_Tab/
├── scripts/                      # Các script chính để chạy
│   ├── train.py                  # Train LIRD model với BERT encoder + Attention
│   ├── generate.py               # Generate synthetic data với Gemini API
│   ├── cluster.py                # K-means clustering và sampling từ z_d embeddings
│   ├── validate.py               # Validate synthetic data bằng k-NN trên z_c
│   ├── embed.py                  # Extract z_c và z_d representations từ model
│   ├── hyperparameter_tuning.py  # Grid/Random search cho hyperparameters
│   └── regenerate_visualizations.py  # Regenerate visualizations từ model
├── src/                          # Source modules
│   ├── data_loader.py            # Load và preprocess datasets
│   ├── prompts.py                # Prompts cho LLM generation
│   └── visualization.py          # Visualization utilities
├── components/                   # Core model components
│   ├── models.py                 # CD_Model và các layers
│   ├── dataset.py                # TextDataset class
│   ├── losses.py                 # HSIC, KL, Reconstruction losses
│   ├── tokenization.py           # Tokenization và caching utilities
│   ├── training.py               # Training loop utilities
│   └── utils.py                  # General utilities
├── fairness_evaluation/          # Đánh giá fairness
│   ├── exp_german_credit.ipynb   # Experiment notebook cho German Credit
│   ├── my_useful_functions.py    # Fairness metrics (SP, EO, EOpp, PP, etc.)
│   ├── AdaFair.py                # AdaFair fair boosting classifier
│   └── exponentiated_gradient_reduction.py  # EGR từ Fairlearn
├── data/                         # Raw data files
├── processed_data/               # Preprocessed data
├── models/                       # Saved model checkpoints
├── visualizations/               # Generated visualizations
└── sampled_data/                 # Sampled data for generation
```

## 📋 Mục lục

1. [Kiến trúc Model](#kiến-trúc-model)
2. [Công thức Loss Functions](#công-thức-loss-functions)
3. [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
4. [Example: End-to-End Workflow](#example-end-to-end-workflow)
5. [Fairness Evaluation](#fairness-evaluation)
6. [Tham số quan trọng](#tham-số-quan-trọng)

---

## 🏗️ Kiến trúc Model

### Architecture Overview

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
        Adversarial Classifier → y_pred (low accuracy ~50%)
    
    z_c, z_d
        ↓
    D_c(z_c) + D_d(z_d) → h_reconstructed (additive reconstruction)
```

### Components

1. **Shared BERT Encoder**: `bert-base-uncased` - mã hóa input text thành embeddings
2. **ContentHead**: Attention pooling + VAE → `z_c` (task-relevant information)
3. **DemographicHead**: Attention pooling + VAE → `z_d` (demographic information, independent of label)
4. **Task Classifier**: Dự đoán label từ `z_c` (target: high accuracy)
5. **Adversarial Classifier**: Dự đoán label từ `z_d` (target: ~50% accuracy, random guess)
6. **Decoders**: `D_c(z_c)` và `D_d(z_d)` để reconstruct BERT embeddings

---

## 📐 Công thức Loss Functions

### 1. Task Loss (L_task)

**Mục tiêu**: `z_c` có thể dự đoán label `y` tốt (high accuracy)

$$
L_{\text{task}} = \text{CrossEntropy}(f_{\text{task}}(z_c), y)
$$

Trong đó:
- $f_{\text{task}}(z_c)$: Task classifier output từ `z_c`
- $y$: True labels

**Target**: Minimize → Accuracy cao (~1.0)

---

### 2. Adversarial Loss (L_adv) - HSIC(z_d, y)

**Mục tiêu**: `z_d` độc lập với label `y` (không thể dự đoán label)

$$
L_{\text{adv}} = \text{HSIC}(z_d, y) + \text{Penalty}(f_{\text{adv}}(z_d), y)
$$

Trong đó **HSIC** (Hilbert-Schmidt Independence Criterion):

$$
\text{HSIC}(X, Y) = \frac{1}{(n-1)^2} \text{tr}(K_X H K_Y H)
$$

Với:
- $K_X, K_Y$: RBF kernel matrices
- $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^T$: Centering matrix
- RBF kernel: $K(x_i, x_j) = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)$

**Random guess penalty**:

$$
\text{Penalty} = \text{CrossEntropy}(f_{\text{adv}}(z_d), y) + \text{KL}(P(y|z_d) \| \text{Uniform})
$$

**Target**: Minimize → HSIC ≈ 0, Accuracy ≈ 50% (random guess)

---

### 3. Orthogonality Loss (L_ortho) - HSIC(z_c, z_d)

**Mục tiêu**: `z_c` và `z_d` độc lập thống kê (statistically independent)

$$
L_{\text{ortho}} = \text{HSIC}(z_c, z_d)
$$

**Target**: Minimize → HSIC ≈ 0 (z_c ⊥ z_d)

---

### 4. Reconstruction Loss (L_rec)

**Mục tiêu**: Reconstruct BERT embeddings từ `z_c` và `z_d`

$$
L_{\text{rec}} = \|D_c(z_c) + D_d(z_d) - h\|^2
$$

Trong đó:
- $h$: Original BERT CLS token embedding [768-dim]
- $D_c(z_c)$: Decoder output từ `z_c`
- $D_d(z_d)$: Decoder output từ `z_d`

**Target**: Minimize → MSE ≈ 0

---

### 5. KL Divergence Loss (L_kl)

**Mục tiêu**: Regularize latent distributions về standard normal $\mathcal{N}(0,I)$

$$
L_{\text{kl}} = \text{KL}(q(z_c|x) \| \mathcal{N}(0, I)) + \text{KL}(q(z_d|x) \| \mathcal{N}(0, I))
$$

Với VAE formulation:

$$
\text{KL}(q(z|x) \| \mathcal{N}(0, I)) = -\frac{1}{2}\sum_{i=1}^{d} \left[1 + \log(\sigma_i^2) - \mu_i^2 - \sigma_i^2\right]
$$

Trong đó:
- $q(z|x) = \mathcal{N}(\mu(x), \sigma^2(x))$: Encoder output
- $d$: Latent dimension

**Lợi ích**:
- Smooth latent space (tốt cho interpolation)
- Proper sampling (có thể sample z ~ N(0,I) và decode để generate data)
- Normalized distribution (tránh sampling ngoài learned manifold)

---

### 6. Variance Constraint (L_var)

**Mục tiêu**: Ngăn `z_d` collapse về zero/noise

$$
L_{\text{var}} = \max(0, \text{min\_variance} - \text{Var}(z_d))
$$

**Target**: Ensure `z_d` có đủ variance để carry information

---

### 7. Combined Loss Function

$$
L_{\text{total}} = \lambda_{\text{task}} \cdot L_{\text{task}} 
                - \lambda_{\text{adv}} \cdot L_{\text{adv}}
                + \lambda_{\text{ortho}} \cdot L_{\text{ortho}}
                + \lambda_{\text{rec}} \cdot L_{\text{rec}}
                + \lambda_{\text{kl}} \cdot L_{\text{kl}}
                + \lambda_{\text{var}} \cdot L_{\text{var}}
$$

**Default weights**:
| Parameter | Value | Mô tả |
|-----------|-------|-------|
| $\lambda_{\text{task}}$ | 3.0 | Ưu tiên cao cho task accuracy |
| $\lambda_{\text{adv}}$ | 0.1 | Minimize HSIC(z_d, y) |
| $\lambda_{\text{ortho}}$ | 1.0 | Minimize HSIC(z_c, z_d) |
| $\lambda_{\text{rec}}$ | 0.5 | Reconstruction quality |
| $\lambda_{\text{kl}}$ | 0.1 | VAE regularization |
| $\lambda_{\text{var}}$ | 0.1 | Prevent collapse |

---

## 🚀 Hướng dẫn sử dụng

### Bước 1: Train Model trên Original Data

Train model để học disentangled representations từ dữ liệu gốc.

```bash
python scripts/train.py \
    --dataset german-credit-data \
    --epochs 100 \
    --batch-size 64 \
    --lambda-task 3.0 \
    --lambda-adv 0.1 \
    --lambda-rec 0.5 \
    --lambda-ortho 1.0 \
    --lambda-kl 0.1 \
    --lambda-var 0.1
```

**Output**: 
- Model được lưu tại `models/{dataset}/best_model_{timestamp}.pth`
- Visualizations tại `visualizations/{dataset}/`

**Tham số quan trọng**:
- `--dataset`: Tên dataset (`german-credit-data`, `adult`)
- `--epochs`: Số epochs training
- `--batch-size`: Batch size (auto-detect nếu không chỉ định)
- `--use-amp`: Sử dụng Mixed Precision để tăng tốc (mặc định: enabled trên GPU)
- `--lambda-*`: Các trọng số loss functions

---

### Bước 2: Cluster Original Data để hiểu Distribution

Cluster dữ liệu dựa trên `z_d` embeddings để tìm các patterns và sample representative examples.

```bash
python scripts/cluster.py \
    --dataset german-credit-data \
    --output-dir sampled_data \
    --n-examples 10
```

**Lần đầu tiên** (clustering + sampling):
- Load model từ `models/{dataset}/best_model_*.pth`
- Generate `z_d` embeddings cho tất cả data
- Split theo label (0 và 1)
- Với mỗi label group:
  - Tìm optimal K (số clusters) bằng silhouette score và Davies-Bouldin index
  - K-means clustering trên `z_d`
  - Sample `n_examples` từ mỗi cluster
- Lưu cluster info để resample sau này

**Resample từ existing clusters** (nhanh hơn):
```bash
python scripts/cluster.py \
    --dataset german-credit-data \
    --output-dir sampled_data \
    --n-examples 20 \
    --resample-only
```

**Output**:
- `sampled_data/{dataset}/{timestamp}_{n_examples}/sampled_sentences_combined.csv`
- `sampled_data/{dataset}/{timestamp}_{n_examples}/sampled_sentences_combined_metadata.json`
- `sampled_data/{dataset}/clusters/label_0/` và `label_1/` (cluster info)

---

### Bước 3: Generate Synthetic Data từ Sampled Sentences

Sử dụng LLM (Mistral, Llama, etc.) để generate synthetic data từ in-context examples.

```bash
python scripts/generate.py \
    --dataset german-credit-data \
    --csv-file sampled_data/german-credit-data/20260130_111716_10/sampled_sentences_combined.csv \
    --output-file synthetic_data.json \
    --n-examples 100 \
    --model gemini-2.0-flash-exp
```

**Lưu ý**:
- `--n-examples`: Số lượng data samples muốn generate (KHÔNG phải số ICL examples)
- Tất cả examples trong CSV file sẽ được dùng làm in-context examples (không giới hạn)
- Script tự động tìm CSV file nếu bạn pass JSON metadata file

**Output**:
- `synthetic_data.json`: Generated samples với metadata
- `sampled_data/{dataset}/label_0_data.csv` và `label_1_data.csv` (để reuse)

**Các models hỗ trợ**:
- `mistralai/Mistral-7B-Instruct-v0.2` (default)
- `meta-llama/Llama-2-7b-chat-hf`
- `microsoft/Phi-3-mini-4k-instruct`
- Và các models khác từ Hugging Face

---

### Bước 4: Phát sinh dữ liệu tích hợp tính năng Công bằng (Fairness)

Sử dụng tham số `--fair` để kích hoạt các tính năng công bằng nâng cao như Fairness Governor, nhắm mục tiêu ICL, và kiểm định phản thực tế (counterfactual).

```bash
python scripts/generate.py \
    --dataset german-credit-data \
    --csv-file sampled_data/german-credit-data/label_0_data.csv \
    --n-examples 100 \
    --fair \
    --sensitive-attrs sex
```

**Tính năng nổi bật**:
- **Fairness Governor**: Tự động cân bằng tỉ lệ giữa các nhóm (VD: 25% cho mỗi tổ hợp Nhãn x Giới tính).
- **Targeted Prompting**: Tự động chỉ định LLM tập trung vào các nhóm đang thiếu hụt trong tập dữ liệu.
- **Counterfactual Validation**: Kiểm tra tính ổn định của quyết định khi thay đổi thuộc tính nhạy cảm.

---

## ⚙️ Tham số quan trọng

### Training Parameters (`scripts/train.py`)

| Parameter | Default | Mô tả |
|-----------|---------|-------|
| `--lambda-task` | 3.0 | Trọng số task loss (z_c → y accuracy) |
| `--lambda-adv` | 0.1 | Trọng số adversarial loss (HSIC(z_d, y)) |
| `--lambda-ortho` | 1.0 | Trọng số orthogonality loss (HSIC(z_c, z_d)) |
| `--lambda-rec` | 0.5 | Trọng số reconstruction loss |
| `--lambda-kl` | 0.1 | Trọng số KL divergence (VAE regularization) |
| `--lambda-var` | 0.1 | Trọng số variance constraint |
| `--latent-dim` | 32 | Dimensionality của z_c và z_d |
| `--batch-size` | Auto | Batch size (auto-detect based on GPU) |
| `--epochs` | 100 | Số epochs training |
| `--lr` | 2e-5 | Learning rate |

### Generation & Fairness Parameters (`scripts/generate.py`)

| Parameter | Default | Mô tả |
|-----------|---------|-------|
| `--n-examples` | 1 | Số lượng mẫu cần phát sinh |
| `--fair` | False | Kích hoạt tính năng công bằng nâng cao |
| `--sensitive-attrs` | "sex" | Các thuộc tính cần cân bằng (phân cách bởi dấu phẩy) |
| `--k-neighbors` | 9 | Số láng giềng k-NN để kiểm định nhãn |
| `--visualize` | False | Tạo visualization PCA 2D cho z_c và z_d |

### Clustering Parameters (`scripts/cluster.py`)

| Parameter | Default | Mô tả |
|-----------|---------|-------|
| `--n-examples` | 10 | Số examples sample từ mỗi cluster |
| `--k-min` | 2 | Minimum K cho K-means |
| `--k-max` | 10 | Maximum K cho K-means |
| `--resample-only` | False | Chỉ resample, không re-cluster |

### Generation Parameters (`generate_synthetic_data.py`)

| Parameter | Default | Mô tả |
|-----------|---------|-------|
| `--n-examples` | 1 | **Số lượng data samples muốn generate** |
| `--model` | mistralai/Mistral-7B-Instruct-v0.2 | Hugging Face model name |
| `--max-new-tokens` | 512 | Maximum tokens to generate |
| `--temperature` | 0.7 | Sampling temperature |
| `--top-p` | 0.9 | Nucleus sampling parameter |
| `--no-shuffle` | False | Không shuffle ICL examples |

---

## � Example: End-to-End Workflow

Đây là ví dụ đầy đủ để tạo synthetic data cho dataset **German Credit**:

### Bước 1: Chuẩn bị môi trường

```bash
# Install dependencies
pip install -r requirements.txt

# Tạo file .env với Gemini API key
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### Bước 2: Train LIRD Model

```bash
python scripts/train.py \
    --dataset german-credit-data \
    --epochs 100 \
    --batch-size 64

# Output:
# - models/german-credit-data/best_model_*.pth
# - visualizations/german-credit-data/
```

### Bước 3: Cluster và Sample từ z_d embeddings

```bash
python scripts/cluster.py \
    --dataset german-credit-data \
    --output-dir sampled_data \
    --n-examples 10

# Output:
# - sampled_data/german-credit-data/{timestamp}_10/sampled_sentences_combined.csv
```

### Bước 4: Generate Synthetic Data với Gemini

```bash
python scripts/generate.py \
    --dataset german-credit-data \
    --csv-file sampled_data/german-credit-data/{timestamp}_10/sampled_sentences_combined.csv \
    --n-examples 1000 \
    --fair \
    --sensitive-attrs sex

# Output:
# - synthetic_data/german-credit-data/{timestamp}/synthetic_data.json
```

### Bước 5: Đánh giá Fairness

```bash
# Mở notebook để chạy fairness evaluation
jupyter notebook fairness_evaluation/exp_german_credit.ipynb
```

---

## 📊 Fairness Evaluation

### Các Metrics trong `my_useful_functions.py`

| Metric | Mô tả | Công thức |
|--------|-------|-----------|
| **Statistical Parity** | Tỷ lệ positive predictions bằng nhau | SP = P(Ŷ=1\|A=0) - P(Ŷ=1\|A=1) |
| **Equalized Odds** | TPR và TNR bằng nhau giữa groups | EO = \|TPR₀ - TPR₁\| + \|TNR₀ - TNR₁\| |
| **Equal Opportunity** | TPR bằng nhau giữa groups | EOpp = \|TPR₀ - TPR₁\| |
| **Predictive Parity** | PPV bằng nhau giữa groups | PP = \|PPV₀ - PPV₁\| |
| **Predictive Equality** | FPR bằng nhau giữa groups | PE = \|FPR₀ - FPR₁\| |
| **Treatment Equality** | FN/FP ratio bằng nhau | TE = FN₀/FP₀ - FN₁/FP₁ |

### Fair Classifiers

#### AdaFair (`AdaFair.py`)
- AdaBoost-based fair classifier
- Tự động điều chỉnh sample weights để cân bằng TPR/TNR giữa protected groups
- Parameter `c`: Trade-off giữa accuracy và fairness

#### Exponentiated Gradient Reduction (`exponentiated_gradient_reduction.py`)
- Wrapper cho Fairlearn's ExponentiatedGradient
- Supports: DemographicParity, EqualizedOdds
- In-processing technique để reduce bias

### Sử dụng

```python
from fairness_evaluation.my_useful_functions import (
    calculate_performance_statistical_parity,
    calculate_performance_equalized_odds,
    calculate_performance_equal_opportunity
)

# Tính fairness metrics
result = calculate_performance_statistical_parity(
    data=X_test,        # Test features
    labels=y_test,      # True labels
    predictions=y_pred, # Model predictions
    saIndex=0,          # Index of sensitive attribute
    saValue=1           # Value của protected group
)

print(f"Accuracy: {result['accuracy']:.4f}")
print(f"Fairness (SP): {result['fairness']:.4f}")
```

---

### Disentanglement Metrics
- **Task Accuracy (z_c → y)**: ~0.85-0.95 (Chất lượng nội dung cao)
- **Adversarial Accuracy (z_d → y)**: ~0.45-0.55 (Thông tin nhạy cảm được tách rời hoàn toàn)
- **HSIC(z_d, y) & HSIC(z_c, z_d)**: < 0.01 (Tính độc lập thống kê cao)

### Fairness Metrics (with `--fair`)
- **Statistical Parity Difference**: Giảm đáng kể nhờ Fairness Governor.
- **Disparate Impact Ratio**: Tiệm cận 1.0 (Công bằng tối ưu).
- **Counterfactual Stability**: > 95% mẫu giữ nguyên nhãn khi đổi thuộc tính nhạy cảm.

---

##  Requirements

Xem `requirements.txt` để biết danh sách dependencies.

**Chính**:
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- pandas
- numpy

---

## 📝 Notes

1. **VAE vs Adversarial**: Framework này sử dụng HSIC-based LIRD, không dùng adversarial training với GRL
2. **ICL Examples**: Tất cả examples trong CSV file được dùng làm in-context examples (không giới hạn)
3. **Model Selection**: Có thể dùng bất kỳ Hugging Face model nào hỗ trợ chat format

---

## 📚 References

- LIRD Framework: Learning Independent Representations for Disentanglement
- HSIC: Hilbert-Schmidt Independence Criterion
- VAE: Variational Autoencoder