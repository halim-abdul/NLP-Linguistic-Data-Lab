# Group Information

- **Group name:** Linguistic Data Lab (Group 10)
- **Group code:** `g10`
- **Group repository:** https://github.com/halim-math/Linguistic-Data-Lab
- **Tutor responsible:** Hamza Ahmed Siddiqui
- **Group team leader:** Shibli (``, GitHub: [Shibli2316](https://github.com/Shibli2316))
- **Group members:**
  * **Shibli** (``, GitHub: [Shibli2316](https://github.com/Shibli2316))
  * **Nallani** (``, GitHub: [sunilnallani116](https://github.com/sunilnallani116))
  * **Halim** (``, GitHub: [halim-math](https://github.com/halim-math))


> Note: **Amjad** (GitHub: [Abdullah-Amjad](https://github.com/Abdullah-Amjad)) contributed the Part 1 Paraphrase Generation (PTG) baseline but is no longer a member of the group.

---

# Setup Instructions

This section describes how to set up the environment, run unit tests, and reproduce all experimental results across the five target tasks.

### 1. Environment Setup

The codebase is compatible with Python 3.10 and PyTorch 2.0+ on both local workstations (Windows/Linux) and High-Performance Computing clusters (e.g., GWDG Grete cluster with NVIDIA A100 GPUs).

```bash
# Clone the repository
git clone https://github.com/halim-math/Linguistic-Data-Lab.git
cd Linguistic-Data-Lab

# Activate the course conda environment
conda activate dnlp
```

### 2. Platform-Specific Fixes (Windows CUDA Dynamic Link Library)
During environment configuration on Windows, default package resolution may link CPU binaries that conflict with local NVIDIA drivers (`WinError 182: fbgemm.dll`). To resolve this, install explicit CUDA 12.1-compiled PyTorch wheels:
```bash
conda activate dnlp
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Implementation Verification via Sanity Tests
Before initiating model training, verify the custom optimizer and Transformer encoder implementations:
```bash
cd sanity_test
python optimizer_test.py
python sanity_check.py
cd ..
```

### 4. Canonical Execution Commands for All Experiments

```bash
# ==============================================================================
# Task 1: Quora Question Pairs (QQP)
# ==============================================================================
# Part 1 Baseline Fine-Tuning (Siamese Head, 10 Epochs; peak result reported at Epoch 4)
python multitask_classifier.py --option finetune --task qqp --use_gpu --batch_size 8 --epochs 10 --lr 1e-5

# Part 2 Enhanced Bidirectional Fine-Tuning (Joint Cross-Attention, 10 Epochs)
python multitask_classifier.py --option finetune --task qqp --qqp_profile bidirectional --epochs 10 --lr 1e-5 --batch_size 8 --seed 11711 --use_gpu --local_files_only

# ==============================================================================
# Task 2: Stanford Sentiment Treebank (SST)
# ==============================================================================
# Part 1 Baseline Fine-Tuning (Linear Head, 10 Epochs)
python multitask_classifier.py --option finetune --task sst --use_gpu --epochs 10 --batch_size 64 --lr 1e-5

# Part 2 Enhanced (Heterogeneous Heads, LLRD, EMA & Multi-Seed Ensemble)
python multitask_classifier.py --option finetune --task sst --sst_profile readme_final_tuned --epochs 10 --batch_size 64 --lr 1e-5 --seed 11711 --use_gpu --local_files_only

# ==============================================================================
# Task 3: Semantic Textual Similarity (STS)
# ==============================================================================
# Part 1 Baseline Fine-Tuning (Linear Regression Head, 10 Epochs)
python multitask_classifier.py --option finetune --task sts --use_gpu --epochs 10 --batch_size 64 --lr 1e-5

# Part 2 Enhanced Sentence-BERT (Mean-Pooling Cosine + Elementwise Difference Head)
python multitask_classifier.py --option finetune --task sts --use_gpu --epochs 10 --batch_size 64 --lr 1e-5 --seed 11711 --local_files_only

# ==============================================================================
# Task 4: Paraphrase Type Detection (PTD)
# ==============================================================================
# Part 1 Baseline (BART-Large Single-Token Head, 5 Epochs)
python bart_detection.py --use_gpu --epochs 5 --batch_size 8 --lr 1e-5

# Part 2 Enhanced (Multi-Scale Pooling + GELU MLP + Top-3 Ensemble + Calibration)
python bart_detection.py --use_gpu --epochs 10 --batch_size 8 --lr 1e-5 --seed 11711

# ==============================================================================
# Task 5: Paraphrase Type Generation (PTG)
# ==============================================================================
# Part 1 Baseline (Sequence-to-Sequence Greedy Search, 5 Epochs)
python bart_generation.py --use_gpu --epochs 5 --batch_size 8 --lr 1e-5

# Part 2 Enhanced (Structured Prefix + Diverse Beam Search Decoding)
python bart_generation.py --use_gpu --epochs 10 --batch_size 8 --lr 1e-5 --seed 11711
```

---

# Methodology

This section outlines the architectural principles, mathematical formulations, and engineering implementations developed across the project.

### 1. Decoupled Weight Decay Optimizer (`optimizer.py`)
Standard Adam couples weight decay with gradient updates, causing suboptimal parameter regularization in Transformer models. We implemented the AdamW algorithm following Loshchilov & Hutter (2017), decoupling weight decay from first and second gradient moments:

$$
\theta_t \leftarrow \theta_{t-1} - \alpha \lambda \theta_{t-1} - \alpha_t \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

where bias-corrected step size scaling is:

$$
\alpha_t = \alpha \cdot \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}
$$

```python
# Core AdamW update step in optimizer.py
for p in group["params"]:
    if p.grad is None:
        continue
    grad = p.grad.data
    state = self.state[p]
    if len(state) == 0:
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p.data)
        state["exp_avg_sq"] = torch.zeros_like(p.data)

    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
    beta1, beta2 = group["betas"]
    state["step"] += 1

    # Apply decoupled weight decay
    if group["weight_decay"] != 0:
        p.data.mul_(1 - group["lr"] * group["weight_decay"])

    # Update momentum and variance running averages
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    # Bias correction
    bias_correction1 = 1 - beta1 ** state["step"]
    bias_correction2 = 1 - beta2 ** state["step"]
    step_size = group["lr"] * (bias_correction2 ** 0.5) / bias_correction1

    denom = exp_avg_sq.sqrt().add_(group["eps"])
    p.data.addcdiv_(exp_avg, denom, value=-step_size)
```

### 2. Bidirectional Transformer Encoder (`bert.py`)
We implemented the primary structural components of the 12-layer minBERT encoder from scratch:
* **Multi-Head Scaled Dot-Product Attention**: Computed attention scores with padding mask projection ($M_{i,j} = -1\times 10^9$ for padding tokens):

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

* **Transformer Layer Add-Norm Residual Connections**: Configured residual normalization wrapping self-attention and position-wise feed-forward networks:

$$
\mathbf{h}_{\mathrm{inter}} = \mathrm{LayerNorm}(\mathbf{x} + \mathrm{Dropout}(\mathrm{SubLayer}(\mathbf{x})))
$$

### 3. Task Architectural Formulations

1. **Quora Question Pairs (QQP)**:
   - *Part 1*: Siamese dual-encoder extracting pooled representations $\mathbf{u}, \mathbf{v} \in \mathbb{R}^{768}$ concatenated with absolute difference:

     $$\mathbf{h}_{\text{QQP}} = [\mathbf{u}, \mathbf{v}, \vert \mathbf{u} - \mathbf{v}\vert] \in \mathbb{R}^{2304}$$

   - *Part 2*: Joint sequence cross-attention ($[\text{CLS}] \circ q_1 \circ [\text{SEP}] \circ q_2 \circ [\text{SEP}]$) allowing bidirectional token-level interactions across all 12 self-attention layers, trained with symmetric bidirectional loss:

     $$\mathcal{L}_{\text{bidirectional}} = \frac{1}{2}\left[\mathcal{L}(q_1, q_2) + \mathcal{L}(q_2, q_1)\right]$$

2. **Stanford Sentiment Treebank (SST)**:
   - *Part 1*: For the Part 1 baseline, we pass each input sentence through minBERT and use the pooled `[CLS]` representation. The resulting 768-dimensional representation is passed through a linear classification layer that produces five logits, one for each sentiment class. We train the model using multiclass cross-entropy loss.

     The Part 1 fine-tuned baseline achieved a best development accuracy of **0.524**, close to the project baseline target of **0.526($\pm 0.004$)**. The best result was reached at epoch 5 in P1_submission branch (after merged all). Before that, I acheived 0.5208 at epoch 3 in P1_SST branch. After this point, training accuracy continued to increase while development accuracy stopped improving and later decreased, indicating overfitting.

   - *Part 2*: For Part 2, our goal was to improve both development accuracy and the stability of fine-tuning. We first experimented with dropout, label smoothing, different learning rates, and `ReduceLROnPlateau`. Since these changes gave only limited improvement, we moved to different sentence representations, including `raw_cls`, `pooler`, `cls_mean`, and `gated_attn` [(See details)](https://github.com/halim-math/Linguistic-Data-Lab/blob/P2_SST/readme_files/SST_RESULTS.md).


     We also introduced Layer-wise Learning Rate Decay, together with gradient clipping, weight decay, early stopping, EMA/model averaging, and soft-voting ensemble prediction.

     The best single model achieved **0.5400** development accuracy. The final ensemble achieved **0.5431**, improving over both our Part 1 baseline of **0.524** and the project baseline target of **0.526**. The submitted SST development predictions reproduce the final **0.5431** accuracy (3-seed ensemble) and **0.5441**(5-seed ensemble) [(See details)](https://github.com/halim-math/Linguistic-Data-Lab/blob/P2_SST/README.md).

3. **Semantic Textual Similarity (STS)**:
   - *Part 1*: Linear regression head over concatenated embeddings $[\mathbf{u}, \mathbf{v}]$ scaled by sigmoid:

     $$\text{score} = 5.0 \cdot \sigma(\mathbf{W}[\mathbf{u}, \mathbf{v}] + b)$$

   - *Part 2*: Sentence-BERT metric learning using attention-masked mean pooling, direct cosine similarity mapping

     $$\text{score}_{\mathrm{cos}} = 2.5 \cdot (\cos(\mathbf{u}, \mathbf{v}) + 1.0)$$

     and an elementwise absolute difference correction MLP ($\vert \mathbf{u} - \mathbf{v}\vert \to 64 \to 1$).

4. **Paraphrase Type Detection (PTD)**:
   - *Part 1*: Single-token representation $\mathbf{h}_0 \in \mathbb{R}^{1024}$ from `facebook/bart-large` projected through a linear classifier with universal $0.5$ decision threshold.
   - *Part 2*: Multi-scale feature pooling (concatenating CLS, Mean, and Max pooled vectors across 3,072 dimensions) combined with 4 linguistic features (Jaccard similarity, character edit distance, length ratio, token count difference), a 3-layer GELU MLP ($3076 \to 768 \to 256 \to 26$), Focal Loss ($\gamma = 1.0$), top-3 checkpoint ensembling, and per-class calibrated decision thresholds $\tau_c \in [0.35, 0.65]$.
   - *Metric note*: the MCC reported for PTD is the **macro-average of the 26 per-class MCC values** (`compute_ptd_metrics`'s `macro_mcc` in `bart_detection.py`), not a single MCC computed on the flattened 26-way label matrix - the two are not comparable (the flattened version is dominated by one near-universal label and reads far higher).

5. **Paraphrase Type Generation (PTG)**:
   - *Part 1*: Raw stringified type ID serialization with greedy sequence-to-sequence beam search (`num_beams = 5`).
   - *Part 2*: Structured linguistic prompt prefixes (`"paraphrase types: 11, 25 | source: <sentence1>"`), dynamic length truncation to 128 tokens, and anti-copy Diverse Beam Search decoding ($G=5$ beam groups, diversity penalty $D=0.5$, repetition penalty $1.15$, trigram no-repeat constraints).
   - *Data note*: ETPC ships with only `train`/`test` splits (no official dev set). Both PTD and PTG derive their "dev" set identically, via an 80/20 split of `etpc-paraphrase-train.csv` with `random_state=11711` (`safe_train_test_split`), giving 2,184/546 rows - used consistently for checkpoint selection and the dev metrics reported throughout this README.

---

# Experiments

This section provides an in-depth analysis of all experiments conducted across the five tasks, detailing expectations, modifications, empirical results, and linguistic error analyses.

---

### Task 1: Quora Question Pairs (QQP)

*Detailed Task Documentation & Visualizations: [`visualisation/qqp/README.md`](./visualisation/qqp/README.md)*

![QQP Validation Metrics](./visualisation/qqp/02_precision_recall_f1_metrics.png)
*Figure 1: Precision, Recall, F1-Score, Accuracy, and Specificity across QQP controlled ablations against the 76.50% course target.*

- **Linguistic Challenge**: The dataset comprises 202,700 question pairs (135,134 train, 33,783 dev, 33,783 test). Models must distinguish high lexical overlap with differing intent (*"How to make money online?"* vs. *"Why did I lose money online?"*) from lexically diverse paraphrases.
- **What Was Changed**:
  1. *Part 1 Baseline*: Siamese encoder with tri-vector concatenation $[\mathbf{u}, \mathbf{v}, \vert\mathbf{u} - \mathbf{v}\vert] \in \mathbb{R}^{2304}$ trained over 10 epochs.
  2. *Part 2 Enhanced*: Stitched inputs into joint cross-attention sequences ($[\mathrm{CLS}] \circ q_1 \circ [\mathrm{SEP}] \circ q_2 \circ [\mathrm{SEP}]$) and optimized with symmetric bidirectional loss:

$$
\mathcal{L}_{\mathrm{bidirectional}} = \frac{1}{2}\left[\mathcal{L}(q_1, q_2) + \mathcal{L}(q_2, q_1)\right]
$$

- **Motivation, Expectation, and Empirical Validation**:
  - *Motivation*: In Part 1, the Siamese dual-encoder processed questions in isolation, preventing token-level cross-alignment until the final pooling layer and introducing order-dependent directional bias. We were motivated to concatenate both queries into a single cross-attention sequence and enforce symmetric bidirectional training.
  - *Expectation*: We hypothesized that allowing all 12 bidirectional self-attention layers to cross-attend across both question spans simultaneously would capture fine-grained semantic discrepancies and boost duplicate detection accuracy well beyond 85%.
  - *Outcome (Did it happen?)*: Yes, the hypothesis was strongly confirmed. Symmetric bidirectional training accelerated convergence and reached **`88.85%`** Dev Accuracy (Paraphrase F1: **`0.8489`**, MCC: **`0.7606`**) at Epoch 3, delivering a **+16.14%** relative gain over the 76.50% course target.
- **Empirical Findings & Discussion**:
  - *Part 1 Baseline*: Achieved peak validation accuracy of **`84.16%`** at Epoch 4 (surpassing the 76.5% target).
  - *Part 2 Final Submitted Model (Exp 03 Bidirectional)*: Rapidly converged to **`88.85%`** Dev Accuracy at Epoch 3. This is the exact model used to produce `predictions/bert/quora-paraphrase-{dev,test}-output.csv`, independently re-verified by scoring the submitted dev predictions against `data/quora-paraphrase-dev.csv`.
  - *Note on ensembling*: We also explored averaging predictions across all 4 QQP ablation checkpoints during experimentation, which trended toward ~89% accuracy. We kept the single bidirectional model as our submitted pipeline to ensure clean end-to-end reproducibility.
- **Limitations**:
  - Full joint cross-attention incurs quadratic computational complexity $\mathcal{O}((L_1 + L_2)^2)$ with respect to sequence length, precluding pre-computed vector indexing for fast semantic retrieval.
  - The model remains susceptible to false positives on adversarial pairs sharing high keyword/entity overlap but differing in critical conditional qualifiers (e.g., *"Can I travel to France without a visa?"* vs. *"Can I travel to France with an expired visa?"*).

---

### Task 2: Stanford Sentiment Treebank (SST)

*Detailed Task Documentation & Visualizations: [`visualisation/sst/README.md`](./visualisation/sst/README.md)*

<p align="center">
  <img src="./visualisation/sst/02_accuracy_vs_epoch.png" width="49%" alt="SST Accuracy vs Epoch" />
  <img src="./visualisation/sst/03_loss_vs_epoch.png" width="49%" alt="SST Loss vs Epoch" />
</p>
<p align="center">
  <em>Figure 2: (Left) Train vs. Validation Accuracy trajectory across 10 epochs displaying peak generalization at Epoch 5. (Right) Multi-class Cross-Entropy Loss decay illustrating steep early optimization dynamics.</em>
</p>

* **In-Depth Visual & Empirical Analysis**:
  * **Accuracy Trajectory (`02_accuracy_vs_epoch.png`)**: Training accuracy climbs monotonically from 34.5% up to 89.5% by Epoch 10. Development accuracy rises rapidly during early epochs, reaching its peak of **`52.38%`** (`0.524`) at **Epoch 5** (matching the official `0.526` baseline target). Beyond Epoch 5, validation accuracy plateaus and drifts down to 50.5%, signaling that minBERT begins memorizing specific training review idioms rather than learning compositional syntax.
  * **Loss Dynamics (`03_loss_vs_epoch.png`)**: Cross-entropy loss drops steadily from `1.472` to `0.305`. The steepest descent occurs between Epochs 1 and 3 ($\Delta \mathcal{L} = -0.470$), where the self-attention heads rapidly separate strong positive and negative polarities before tuning subtle neutral boundaries.
  * **Part 2 Multi-Head & Ensemble Breakthrough**: By introducing 4 heterogeneous representation heads (`raw_cls`, `pooler`, `cls_mean`, `gated_attn`), Layer-wise Learning Rate Decay ($\alpha = 0.95$), and temperature-calibrated ($T = 1.1$) soft-voting ensembling across 5 random seeds, Part 2 lifted validation accuracy to **`54.31%`** (`0.5431`), achieved an adjacent accuracy ($\pm 1$ class step) of **`93.26%`**, and suppressed severe polar inversions to under **`0.96%`**.

- **Linguistic Challenge**: 11,847 Rotten Tomatoes review sentences (7,898 train, 1,974 dev, 1,975 test) for fine-grained 5-class polarity classification (Very Negative to Very Positive). The core challenge is handling compositional negation, irony, and clause-level sentiment shifts.
- **What Was Changed**:
  1. *Part 1 Baseline*: Fine-tuned full minBERT backbone with a 5-class linear projection head on pooled `[CLS]`.
  2. *Part 2 Enhanced*: Evaluated 4 heterogeneous representation heads (`raw_cls`, `pooler`, `cls_mean`, `gated_attn`), introduced Layer-wise Learning Rate Decay (LLRD, $\alpha = 0.95$), Exponential Moving Average ($\beta = 0.995$), dynamic LR scheduling (`ReduceLROnPlateau`), and soft-voting ensembling with temperature calibration ($T = 1.1$).
- **Motivation, Expectation, and Empirical Validation**:
  - *Motivation*: In Part 1, single `[CLS]` fine-tuning suffered from rapid training-set memorization and severe generalization plateau past Epoch 5. We were motivated to evaluate regularizations (Dropout tuning, label smoothing, LLRD, EMA, `ReduceLROnPlateau`) and diverse representation heads to prevent catastrophic forgetting.
  - *Expectation*: We hypothesized that decay schedules and EMA would stabilize foundational encoder representations, while soft-voting ensemble averaging across heterogeneous heads would reduce variance and push fine-grained accuracy past 54%.
  - *Outcome (Did it happen?)*: Yes. While standard dropout and label smoothing slightly dampened peak performance, `ReduceLROnPlateau` raised single-model accuracy to 0.529, and the 3-head soft-voting ensemble achieved **`54.31%`** (further extended to **`54.41%`** with the 5-head ensemble), reducing polar inversions to under **`0.96%`**.
- **Empirical Findings & Discussion**:
  - *Part 1 Baseline*: Reached **`52.38%`** Dev Accuracy at Epoch 5 (matching the 52.6% course target) before overfitting degraded performance.
  - *Part 2 Top Single Model (`seed42-raw_cls`)*: Reached **`54.00%`** Dev Accuracy at Epoch 4.
  - *Part 2 Winning Ensemble*: Calibrated 3-head soft-voting ensemble achieved **`54.31%`** Dev Accuracy and **`93.26%`** adjacent accuracy, with the 5-head ensemble reaching **`54.41%`**.
- **Limitations**:
  - Fine-grained 5-class sentiment inherently contains human annotation subjectivity, especially on Neutral (Class 2), where F1 remains lowest (33.3%).
  - Subtle pragmatic sarcasm without overt lexical contrast remains challenging without broader contextual discourse grounding.

---

### Task 3: Semantic Textual Similarity (STS)

*Detailed Task Documentation & Visualizations: [`visualisation/sts/README.md`](./visualisation/sts/README.md)*

![STS Scatter Correlation Fit](./visualisation/sts/01_scatter_correlation_fit.png)
*Figure 3: Predicted vs. gold-standard similarity scores across 1,430 development sentence pairs* ($r = 0.8190, \rho = 0.8008$).

- **Linguistic Challenge**: 8,579 sentence pairs (5,719 train, 1,430 dev, 1,430 test) with continuous similarity scores $y \in [0.0, 5.0]$. The model must preserve graded semantic nuance while penalizing lexical overlap without semantic equivalence.
- **What Was Changed**:
  1. *Part 1 Baseline*: Concatenated embeddings $[\mathbf{u}, \mathbf{v}]$ mapped through a linear regression layer scaled by sigmoid.
  2. *Part 2 Enhanced*: Replaced linear head with Sentence-BERT cosine similarity $\text{score}_{\mathrm{cos}} = 2.5 \cdot (\cos(\mathbf{u}, \mathbf{v}) + 1.0)$ using attention-masked mean pooling, combined with an auxiliary elementwise difference MLP ($|\mathbf{u} - \mathbf{v}| \to 64 \to 1$).
- **Motivation, Expectation, and Empirical Validation**:
  - *Motivation*: Part 1's linear regression over concatenated embeddings lacked an explicit metric space geometry, causing embeddings to drift in an unconstrained Euclidean space and stagnating at a poor correlation of $r = 0.371$. We were motivated to enforce a geometric cosine similarity constraint over pooled token representations.
  - *Expectation*: We hypothesized that Sentence-BERT cosine metric mapping with attention-masked mean pooling would dramatically boost correlation past 0.65, and adding an elementwise difference MLP ($|\mathbf{u} - \mathbf{v}|$) would resolve localized lexical discrepancies to reach $r > 0.75$.
  - *Outcome (Did it happen?)*: Yes, the empirical gain was our largest across the entire project. Cosine mean-pooling immediately lifted Pearson $r$ to 0.664, and the residual difference head drove final correlation to **`0.8190` Pearson $r$** and **`0.8008` Spearman $\rho$** (**+136.71% relative gain**, 2.37× the course target).
- **Empirical Findings & Discussion**:
  - *Part 1 Baseline*: Stuck at $r = 0.371$ at Epoch 3 before decaying to $0.331$.
  - *Part 2 Mean-Pool Cosine*: Jumped immediately to $r = 0.664$.
  - *Part 2 Final (Mean-Pool + Diff Head)*: Reached **`0.8190` Pearson $r$** and **`0.8008` Spearman $\rho$** at Epoch 9.
- **Limitations**:
  - Metric mapping can be biased by asymmetric sentence lengths where one short clause is paired with an elaborately verbose sentence with peripheral details.
  - Highly domain-specific technical jargon absent from minBERT pre-training can lead to compressed cosine representations.

---

### Task 4: Paraphrase Type Detection (PTD)

*Detailed Task Documentation & Visualizations: [`visualisation/ptd/README.md`](./visualisation/ptd/README.md)*

![PTD Label Co-occurrence Heatmap](./visualisation/ptd/02_label_cooccurrence_heatmap.png)
*Figure 4: Label co-occurrence structure across ETPC multi-label paraphrase categories.*

- **Linguistic Challenge**: Multi-label classification across 26 fine-grained paraphrase categories in the Enriched Twitter Paraphrase Corpus (2,184 train, 546 dev, 585 test).
- **The Majority-Negative Paradox**: Negative label prevalence accounts for 90.25% of all binary entries. A naive baseline predicting all zeros scores 90.25% accuracy but achieves 0.00 MCC and 0.00 F1.
- **What Was Changed**:
  1. *Part 1 Baseline*: Single-token representation $\mathbf{h}_0 \in \mathbb{R}^{1024}$ from `facebook/bart-large` with linear layer and fixed 0.5 decision threshold.
  2. *Part 2 Enhanced*: Multi-scale pooling (CLS, Mean, and Max pooling across 3,072 dims) + 4 explicit linguistic difference features + 3-layer GELU MLP ($3076 \to 768 \to 256 \to 26$) + Focal Loss ($\gamma = 1.0$) + top-3 checkpoint ensembling + per-class threshold calibration ($\tau_c \in [0.35, 0.65]$).
- **Motivation, Expectation, and Empirical Validation**:
  - *Motivation*: Part 1's single `[CLS]` token pooling and universal 0.5 decision threshold suffered from severe majority-negative collapse, failing to detect rare syntactic transformations and yielding poor macro-MCC (0.0824) and low Exact Match (5.50%).
  - *Expectation*: We hypothesized that concatenating CLS, Mean, and Max pooled vectors with explicit surface-level linguistic features (Jaccard, edit distance, length delta) and training with Focal Loss would force the model to focus on hard, sparse positive instances.
  - *Outcome (Did it happen?)*: Yes. Multi-scale feature concatenation and threshold calibration elevated **macro-averaged MCC to `0.1311`** (**+59.1% relative gain**), increased **Exact Match to `7.88%`**, and lifted **Micro-F1 to `0.6913`** while maintaining **`91.24%`** Mean Binary Accuracy.
- **Empirical Findings & Discussion**:
  - *Part 1 Baseline*: Reached 90.70% accuracy, but MCC was only 0.0824 and Exact Match was 5.50%.
  - *Part 2 Final Model*: Reached **`91.24%` Binary Accuracy**, boosted **Exact Match to `7.88%`**, lifted **Micro-F1 to `0.6913`**, and surged **macro-averaged MCC to `0.1311`**. Values match `predictions/bart/etpc-paraphrase-detection-test-output.csv` exactly.
- **Limitations**:
  - Severe label sparsity in the smallest categories (several classes have $<20$ positive instances in the training split) imposes a fundamental ceiling on rare-category generalization.
  - Multi-scale pooling expands feature dimensionality to 3,076, requiring careful dropout regularization to prevent over-indexing on Twitter-specific orthographic artifacts.

---

### Task 5: Paraphrase Type Generation (PTG)

*Detailed Task Documentation & Visualizations: [`visualisation/ptg/README.md`](./visualisation/ptg/README.md)*

![PTG Epoch Convergence](./visualisation/ptg/02_epoch_convergence_trajectory.png)
*Figure 5: Reference BLEU, Penalized BLEU, and Input Novelty across all 10 training epochs for Part 2 prefix-conditioned BART, read directly from `runs/ptg_history.csv`.*

- **Linguistic Challenge & The Copy Trap**: Given source sentence $s$ and target paraphrase type IDs, synthesize an altered paraphrase $\hat{s}$. Standard sequence-to-sequence beam search defaults to copying the source sentence verbatim, causing official Penalized BLEU to collapse to zero:

$$
\mathrm{Penalized\ BLEU} = \frac{\mathrm{BLEU}_{\mathrm{ref}} \times (100 - \mathrm{BLEU}_{\mathrm{in}})}{52}
$$

- **What Was Changed**:
  1. *Part 1 Baseline*: Raw stringified type ID lists with standard greedy beam search (`num_beams = 5`).
  2. *Part 2 Enhanced*: Structured linguistic prompt prefixing, dynamic sequence length truncation to 128 tokens, and anti-copy Diverse Beam Search ($G=5$ groups, diversity penalty $D=0.5$, repetition penalty 1.15, trigram no-repeat constraints).
- **Motivation, Expectation, and Empirical Validation**:
  - *Motivation*: In Part 1, standard sequence-to-sequence decoding succumbed to the "copy trap": the model achieved 47.10 Reference BLEU by simply echoing the source sentence verbatim, resulting in 0.00% novelty and a Penalized BLEU of 0.00.
  - *Expectation*: We hypothesized that structured prompt prefixes (`"paraphrase types: ... | source: ..."`) would enforce type conditioning, while Diverse Beam Search with trigram blocking and diversity penalties would prevent verbatim copying and force novel structural edits.
  - *Outcome (Did it happen?)*: Yes. In Epoch 1, Reference BLEU peaked at **`48.05`** (surpassing the 47.50 course target). Across 10 epochs, anti-copy Diverse Beam Search expanded paraphrastic Novelty monotonically from 3.37% to **`22.63%`** (+571% increase), cutting verbatim copying from 78.02% to 21.98% and driving official Penalized BLEU to **`19.65`** (+530.7% increase).
- **Empirical Findings & Discussion**:
  - *Part 1 Baseline*: Collapsed into verbatim copying (Penalized BLEU = $0.00$).
  - *Part 2 Epoch 1 (Semantic Transfer Peak)*: Reference BLEU reached **`48.05`**, outperforming the official course target of `47.50`.
  - *Part 2 Epoch 10 (Novelty Peak)*: Paraphrastic Novelty grew monotonically from `3.37%` to **`22.63%`** and Penalized BLEU reached **`19.65`**.
- **Limitations**:
  - High diversity penalties create an intrinsic tension between lexical creativity and reference fidelity; aggressive rewriting occasionally alters nuanced named entities or minor numbers.
  - Conditioned generation on combinations of 3+ simultaneous paraphrase types can occasionally cause the model to satisfy dominant lexical shifts while omitting subtle syntactic reorderings.

---

# Results

Below are the verified experimental results for each task reported with three-digit precision.

### Stanford Sentiment Treebank (SST)

| *Model Configuration* | *Dev Accuracy* |
| --- | ---: |
| Part 1 Baseline | `0.524` |
| Baseline + default Dropout | `0.521` |
| Learning rate + Dropout | `0.517` |
| Learning rate + Dropout + Label smoothing | `0.512` |
| ReduceLROnPlateau | `0.529` |
| 3-Head Soft-Voting Ensemble | `0.5431` |
| *5-Head Soft-Voting Ensemble* | *`0.5441`* |

The 5-head ensemble achieved the highest SST accuracy, while the 3-head ensemble can also be enough as the final model; the 5-head version is an optional further improvement.

### Quora Question Pairs (QQP)

| Model Configuration | Dev Accuracy | Paraphrase F1 | Macro F1 | MCC | Dev Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Course Baseline Target** | `0.765` | - | - | - | - |
| **Part 1 Baseline (Siamese Head)** | `0.842` | `0.790` | `0.832` | `0.664` | `0.371` |
| **Part 2: Joint Cross-Attention (Single-Pass)** | `0.886` | `0.848` | `0.878` | `0.757` | `0.292` |
| **Part 2: Joint Cross-Attention + Gradient Clip** | `0.885` | `0.847` | `0.877` | `0.755` | `0.290` |
| **Part 2: Final Bidirectional Symmetric Training (submitted)** | **`0.889`** | **`0.849`** | **`0.880`** | **`0.761`** | **`0.282`** |

*The Final Bidirectional row is the exact model behind `predictions/bert/quora-paraphrase-{dev,test}-output.csv` (0.8885 accuracy, 0.8489 F1, 0.7606 MCC).*

### Semantic Textual Similarity (STS)

| Model Configuration | Pearson Correlation ($r$) | Spearman Rank ($\rho$) | Dev MSE Loss |
| :--- | :---: | :---: | :---: |
| **Course Baseline Target** | `0.346` | - | - |
| **Part 1 Baseline (Linear Regression Head)** | `0.371` | `0.354` | `0.678` |
| **Part 2: Cosine Similarity + Max Pooling** | `0.742` | `0.725` | `0.341` |
| **Part 2: Cosine Similarity + Attention Pooling** | `0.805` | `0.789` | `0.265` |
| **Part 2: Cosine Similarity + Mean Pooling** | `0.664` | `0.651` | `0.412` |
| **Part 2: Final Mean-Pool + Difference Head** | **`0.819`** | **`0.801`** | **`0.228`** |

### Paraphrase Type Detection (PTD)

| Model Configuration | Mean Binary Acc | Exact Match | Macro F1 | Micro F1 | MCC | Hamming Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Course Baseline Target** | `0.913` | - | - | - | - | - |
| **Part 1 Baseline (Single-Token Head)** | `0.907` | `0.055` | `0.112` | `0.641` | `0.082` | `0.093` |
| **Part 2: Multi-Scale 3072 + GELU MLP** | `0.910` | `0.070` | `0.164` | `0.683` | `0.072` | `0.090` |
| **Part 2: Multi-Scale 3076 + Focal Loss** | `0.911` | `0.070` | `0.159` | `0.681` | `0.107` | `0.089` |
| **Part 2: Top-3 Ensemble + Calibrated Thresholds (submitted)** | **`0.912`** | **`0.079`** | **`0.205`** | **`0.691`** | **`0.131`** | **`0.088`** |

*MCC above is macro-averaged across the 26 per-class MCCs (see Methodology). Row values regenerated via `python bart_detection.py --use_gpu --eval_only --seed 11711`, matching the submitted `predictions/bart/etpc-paraphrase-detection-test-output.csv` exactly.*

### Paraphrase Type Generation (PTG)

| Model Configuration | Reference BLEU | Novelty ($100 - \mathrm{BLEU}_{\mathrm{in}}$) | Penalized BLEU | ROUGE-1 F1 | ROUGE-L F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Course Baseline Target** | `47.500` | - | - | - | - |
| **Part 1 Baseline (Standard Beam Search)** | `47.100` | 0.00% | `0.000` | `0.612` | `0.584` |
| **Part 2: Structured Prefix (Epoch 1 Peak)** | **`48.050`** | 3.37% | `3.120` | `0.641` | `0.602` |
| **Part 2: Diverse Beam Search (Epoch 5)** | `46.950` | 15.24% | `13.760` | `0.672` | `0.628` |
| **Part 2: Final Diverse Beam Search (Epoch 10)** | `45.140` | **`22.63%`** | **`19.647`** | **`0.695`** | **`0.645`** |

---

### Master Cross-Task Benchmark Synthesis Table

| Task Name | Model Architecture | Baseline Target | Part 1 Baseline | Part 2 Improved (Ours) | Relative Gain / Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **QQP** | Joint Cross-Attention + Bidirectional Loss | `0.765` | `0.842` | **`0.889`** Accuracy (submitted) | **+16.1% above target** |
| **SST** | 3-Head Soft-Voting Ensemble ($T = 1.1$) | `0.526` | `0.524` | **`0.543`** Dev Accuracy | **+3.25% above target** |
| **STS** | Mean-Pool Cosine + Difference Head | `0.346` | `0.371` | **`0.819`** Pearson Correlation | **+136.7% relative gain (2.37×)** |
| **PTD** | Multi-Scale BART + Calibrated Thresholds | `0.913` | `0.907` | **`0.912`** Accuracy, **`0.131`** macro MCC | **Near target (+59.1% MCC gain)** |
| **PTG** | Diverse Beam Search ($G=5, D=0.5$) | `47.500` | `47.100` | **`48.050`** Ref BLEU, **`19.647`** Pen BLEU | **+571% Novelty** |

---

### Detailed Experiment-by-Experiment Benchmark Table

The table above summarizes each task's final result; the table below expands it into the individual experiments/checkpoints behind each task, with per-row contributor attribution.

| Task Name | Model Architecture | Baseline Target | Part 1 Baseline | Part 2 Improved (Ours) | Relative Gain / Status | Primary Contributors |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **QQP (Part 1 baseline)** | minBERT + Siamese Head $[u, v, \vert u-v\vert]$ | `0.765` | `0.8416` (Accuracy) | - | +10.0% above target | A. R. Shibli (P1) |
| **QQP (Part 2, Self-Attention)** | minBERT + Joint $[\text{CLS}]\ q_1\ [\text{SEP}]\ q_2\ [\text{SEP}]$ | `0.765` | - | `0.886` (Acc), `0.848` (F1) | +15.8% above target | A. R. Shibli (P2), A. Halim (P2) |
| **QQP (Part 2, Final Bidirectional, submitted)** | minBERT + Joint Cross-Attn + Bidirectional | `0.765` | - | **`0.889`** (Acc), **`0.849`** (F1) | +16.1% above target (+5.57% over P1) | A. R. Shibli (P2), A. Halim (P2) |
| **SST (Part 1 baseline)** | minBERT + 5-Class Head | `0.526` | `0.524` (Accuracy) | - | Matches target at Epoch 5 | A. Halim (P1) |
| **SST (Part 2, Top Single Model)** | minBERT + Raw [CLS] + LLRD + EMA | - | - | `0.540` (Accuracy) | +2.7% above target | A. Halim (P2) |
| **SST (Part 2, Winning 3-Head Ensemble, submitted)** | 3-Head Soft-Voting Ensemble ($T=1.1$) | `0.526` | - | **`0.543`** (Accuracy) | +3.3% above target (93.3% Adj Acc) | A. Halim (P2) |
| **STS (Part 1 baseline)** | minBERT + Regression Head | `0.346` | `0.371` (Correlation) | - | Baseline | S. K. Nallani (P1) |
| **STS (Part 2, Final Diff Head, submitted)** | minBERT + Mean-Pool + Diff Head | `0.346` | - | **`0.819`** (Correlation) | +136.7% Relative Gain vs. Target (2.37×) | S. K. Nallani (P2) |
| **PTD (Part 1 baseline)** | Single-Token [CLS] + Linear Head | `0.913` | `0.9070` (Acc), `0.0824` (MCC) | - | Baseline reference | A. R. Shibli (P1) |
| **PTD (Part 2, Uncalibrated, Epoch 6)** | Multi-Scale (3072) + 2-Layer GELU MLP | - | - | `0.9102` (Acc), `0.0721` (MCC) | +0.35% Acc, -12.5% MCC (single checkpoint; ensembling and calibration recover MCC below) | A. R. Shibli (P2) |
| **PTD (Part 2, Focal Loss)** | Multi-Scale (3076) + 4 Features + Focal Loss | - | - | `0.9113` (Acc), `0.1074` (MCC) | +0.47% Acc, +30.3% MCC | A. R. Shibli (P2) |
| **PTD (Part 2, Final Calibrated, submitted)** | Multi-Scale + 2-Layer MLP + Threshold Sweep | `0.913` | - | **`0.9124`** (Accuracy), **`0.1311`** (macro-MCC) | Near 0.913 target (-0.06pp), +59.1% Relative macro-MCC Gain | A. R. Shibli (P2) |
| **PTG (Part 1 baseline)** | Raw String List + Standard Beam Search | `47.5` | `47.10` (Ref BLEU), `0.00` (Penalized) | - | Collapses to exact copying ($100-\text{BLEU}_{\text{in}} \approx 0$) | A. Amjad (P1) |
| **PTG (Part 2, Early Epoch 1)** | Structured Prefix + Diverse Beam Search | `47.5` | - | **`48.05`** (Ref BLEU), `3.12` (Penalized) | Exceeds 47.5 target in Epoch 1 | A. R. Shibli (P2) |
| **PTG (Part 2, Mid Epoch 5)** | Structured Prefix + Diverse Beam + Trigram Anti-Repeat | - | - | `46.95` (Ref BLEU), `13.76` (Penalized) | Monotonic novelty rise ($3.37 \to 15.24$) | A. R. Shibli (P2) |
| **PTG (Part 2, Final Epoch 10, submitted)** | Structured Prefix + Diverse Beam Search + Cosine Decay | `47.5` | - | `45.14` (Ref BLEU), **`19.65`** (Penalized) | +571% Novelty ($3.37 \to 22.63$) | A. R. Shibli (P2) |

---

### Hyperparameter Optimization

Optimal hyperparameters were determined through systematic sweeps rather than unguided searches:
1. **Optimization & Learning Rates**: AdamW with $\beta_1 = 0.9, \beta_2 = 0.999$, decoupled weight decay $\lambda = 0.01$. Backbone learning rate $\eta = 1\times 10^{-5}$ combined with classification head multiplier $\eta_{\text{head}} = 3\times 10^{-5}$ on PTD and $2\times 10^{-5}$ on SST.
2. **Layer-wise Learning Rate Decay (LLRD)**: On SST, assigning exponential decay $\alpha = 0.95$ per layer ($\eta_l = \eta_{\text{base}} \cdot \alpha^{(12 - l)}$) prevented catastrophic forgetting in lower Transformer layers while allowing rapid classification head convergence.
3. **Exponential Moving Average (EMA)**: An EMA shadow weight decay rate of $\beta_{\text{EMA}} = 0.995$ on SST smoothed parameter trajectories across flat loss minima.
4. **Probability Temperature Calibration**: Softmax temperature scaling with $T = 1.1$ on SST reduced overconfident misclassifications on ambiguous reviews, dropping Negative Log-Likelihood from 1.2450 to 1.0637.
5. **Diverse Beam Search Configuration**: On PTG, setting 5 beam groups ($G=5$), diversity penalty $D=0.5$, repetition penalty $1.15$, and trigram no-repeat constraints balanced semantic fidelity with novel paraphrastic rewrites.

---

## Visualizations

*Detailed Cross-Task Benchmark Documentation & Visualizations: [`visualisation/benchmark/README.md`](./visualisation/benchmark/README.md)*

---

### Figure 6: Unified 5-Task Performance Benchmark Radar
![5-Task Grand Unified Performance Radar](./visualisation/benchmark/01_grand_unified_5task_radar.png)
* **Visual Overview**: Benchmark radar mapping normalized model performance against the 100% course target boundary across all five tasks.
* **Quantitative Benchmark Analysis**:
  * **Task 1 (QQP)**: 116.1% of course target (**`88.85%`** Dev Accuracy, final bidirectional model, vs. 76.50% baseline target).
  * **Task 2 (SST-5)**: 103.3% of course target (**`54.31%`** Dev Accuracy vs. 52.60% baseline target).
  * **Task 3 (STS)**: 236.7% of course target (**`0.8190`** Pearson Correlation vs. 0.3460 baseline target).
  * **Task 4 (PTD)**: 99.9% of course target (**`0.9124`** Accuracy vs. `0.9130` target, within 0.06 points), backed by a **+59.1%** relative macro-MCC gain (`0.0824` $\to$ `0.1311`) reflecting substantially improved minority-class detection.
  * **Task 5 (PTG)**: 101.2% of course target on official Reference BLEU (**`48.05`** vs. 47.50 target), accompanied by a **+530.7%** increase in Penalized BLEU across Part 2 training (`3.12` $\to$ `19.65`, Epoch 1 $\to$ Epoch 10) reflecting novel paraphrastic generation.

---

### Figure 7: Quora Question Pairs 4-Experiment Ablation Dynamics
![QQP Performance](./visualisation/qqp_performance.png)
* **Panel 1 (Training Loss Trajectory)**: Siamese baseline (Exp 01, gray dashed) decays gradually from `0.4907` to `0.0510`. Joint cross-attention (Exp 02, orange dash-dot) accelerates convergence, while Symmetric Bidirectional Training (Exp 03, blue solid) plunges smoothly from `0.3403` down to `0.0174` because both $(q_1, q_2)$ and $(q_2, q_1)$ permutations double gradient constraints per batch. Gradient clipping (Exp 04, cyan dotted) maintains stable bounded descent.
* **Panel 2 (Validation Accuracy Progression)**: All three joint attention architectures exceed 87.4% accuracy in Epoch 1, substantially outperforming Part 1 baseline (`80.42%`). Exp 03 (bidirectional) peaks at **`88.85%`** at Epoch 3 (**+16.1% relative gain over target**) - this is the checkpoint used for the final submitted QQP predictions.
* **Panel 3 (Paraphrase F1 & Matthews Correlation)**: Paraphrase F1 reaches **`0.8508`** at Epoch 5, and MCC surges from `0.6635` to **`0.7606`**, confirming balanced duplicate separation across 21,280 negative and 12,503 positive validation pairs.

---

### Figure 8: Stanford Sentiment Treebank Loss & Generalization Dynamics
![SST Performance](./visualisation/sst_performance.png)
* **Left Panel (Cross-Entropy Loss Dynamics)**: Multi-class loss drops steadily from `1.472` to `0.305`, with the steepest inflection between Epochs 1 and 3 ($\Delta \mathcal{L} = -0.470$), where self-attention heads rapidly learn coarse sentiment polarities before tuning subtle neutral boundaries.
* **Right Panel (Train vs. Dev Generalization Inflection)**: Training accuracy rises continuously from `0.345` to `0.895`. Validation accuracy reaches its generalization maximum of **`52.38%`** at **Epoch 5** (matching the `0.526` baseline target) before overfitting on training review idioms causes validation accuracy to drift down to `50.5%` by Epoch 10.

---

### Figure 9: Semantic Textual Similarity Part 1 vs. Part 2 Convergence
![STS Comparison](./visualisation/sts_comparison.png)
* **Left Panel (Loss Dynamics)**: Part 1 MSE loss descends slowly ($2.186 \to 0.678$) as the linear layer struggles with raw concatenation. In Part 2, cosine metric loss plummets from `2.113` down to `0.228`.
* **Middle Panel (Pearson Correlation Scaling)**: Part 1 baseline peaks early at Epoch 3 ($r = 0.371$) before decaying to `0.331`. Part 2 Sentence-BERT mean-pooling + difference MLP climbs monotonically from `0.713` to **`0.8190`** (**+136.7% relative gain**, 2.37× target).
* **Right Panel (Train vs. Dev Correlation)**: Dev correlation stabilizes at Epoch 9 ($r = 0.8190$) with a tightly bounded train-dev gap ($0.154$), confirming optimal stopping before severe memorization.

---

### Figure 10: Paraphrase Type Detection Multi-Scale Evaluation
![PTD Comparison](./visualisation/ptd_comparison.png)
* **Left Panel (BCE Loss Dynamics)**: Part 1 terminates at Epoch 5 (`0.2250`). Part 2 smoothly decays from `0.4316` down to `0.1263`, finding a flatter and more generalizable loss basin.
* **Middle Panel (Binary Accuracy & Calibration)**: Due to high negative label prevalence, both models operate in the 90.2% to 91.2% band; per-class calibrated thresholding pushes final accuracy to **`91.24%`**, closely matching the `0.9130` course target.
* **Right Panel (Matthews Correlation Coefficient Leap)**: Macro-averaged MCC surges from baseline `0.0824` to **`0.1311`** (**+59.1% relative gain**), and full 26-label Exact Match jumps from 5.50% to **`7.88%`**, demonstrating genuine minority-class detection.

---

### Figure 11: Paraphrase Type Generation Fidelity vs. Novelty
![PTG Comparison](./visualisation/ptg_comparison.png)
* **Left Panel (Seq2Seq Loss)**: Cross-entropy loss drops smoothly from `1.6386` to `0.6428`, with dynamic sequence length truncation (128 tokens) doubling training throughput.
* **Middle Panel (Reference BLEU Semantic Fidelity)**: Peaks early at **`48.05`** in Epoch 1 (surpassing the `47.50` target) and maintains strong semantic fidelity throughout training.
* **Right Panel (Paraphrastic Novelty Expansion)**: Paraphrastic Novelty ($100 - \text{BLEU}_{\text{in}}$) surges from `3.37%` to **`22.63%`** (**+571% increase**), slashing verbatim copying from `78.02%` down to `21.98%` and driving official Penalized BLEU to **`19.65`**.

---

### Figure 12: Master Cross-Task Executive Scoreboard
![Overall Benchmark Comparison](./visualisation/summary_benchmark.png)
* **Visual Summary**: Unified cross-task comparative scoreboard displaying achieved scores, course targets, and relative percentage margins across all five tasks in a single executive chart.

---

### Cross-Task Visual Analysis & Key Trends:
* **Convergence Dynamics**: Across all classification tasks (QQP, SST, STS), modern metric heads and joint cross-attention enabled rapid convergence within 3 to 5 epochs, avoiding training-set memorization.
* **Generalization Trade-offs**: In PTG, early epochs maximize lexical overlap (Reference BLEU peaks at 48.05), while later epochs with Diverse Beam Search force structural novelty (surging Novelty to 22.63% and Penalized BLEU to 19.65).
* **Metric Robustness**: In PTD, while raw binary accuracy remained stable in the 90-91% range due to label sparsity, macro-averaged MCC surged by +59.1% and Exact Match climbed from 5.50% to 7.88%, demonstrating genuine multi-label discovery.

---

# Members Contribution

| Member | Tasks Handled | Core Contributions |
| :--- | :--- | :--- |
| **Shibli**<br>*(Team Leader)* | **PTD (Part 1 & 2)**<br>**PTG (Part 2)**<br>**QQP (Part 1 & 2)** | • Designed and implemented multi-scale neural pooling (CLS + Mean + Max = 3072 dims) + 4 linguistic difference features + 3-layer GELU MLP + Focal Loss + threshold calibration for PTD, boosting macro-MCC to 0.1311 and Exact Match to 7.88%.<br>• Developed structured prompt conditioning and anti-copy Diverse Beam Search decoding for PTG, achieving 48.05 Ref BLEU and 19.65 Penalized BLEU.<br>• Co-developed joint cross-attention and symmetric bidirectional training for QQP, achieving 88.85% Dev Accuracy (the model behind the final submitted predictions).<br>• Directed repository lifecycle, git merge conflict resolution, environment onboarding & CUDA debugging, and authored publication-quality visualization suite. |
| **Nallani** | **STS (Part 1 & 2)** | • Spearheaded end-to-end implementation of STS Sentence-BERT metric learning framework.<br>• Conducted systematic pooling ablations (CLS, Max, Attention, Mean), establishing Mean Pooling as optimal.<br>• Designed elementwise absolute difference correction MLP ($\vert \mathbf{u} - \mathbf{v}\vert \to 64 \to 1$), driving Pearson correlation to **0.8190** (2.37× course target) and Spearman $\rho$ to 0.8008.<br>• Conducted exhaustive 10-epoch loss tracking and academic proofreading of master documentation. |
| **Halim** | **SST (Part 1 & 2)**<br>**QQP (Part 2)** | • Spearheaded SST Part 1 baseline and Part 2 5-class fine-grained sentiment classification.<br>• Evaluated 4 heterogeneous pooling heads (`raw_cls`, `pooler`, `cls_mean`, `gated_attn`), Layer-wise Learning Rate Decay ($\alpha = 0.95$), EMA ($\beta = 0.995$), and soft-voting ensemble ($T = 1.1$), achieving **54.31%** Dev Accuracy.<br>• Co-developed QQP joint sequence token serialization and conceptualized comparative multi-panel visual graph layouts. |
| **Amjad** | **PTG (Part 1)** | • Implemented Part 1 sequence-to-sequence paraphrase generation baseline using `facebook/bart-large` with teacher-forcing cross-entropy loss, achieving 47.10 Reference BLEU on the development set. |

---

# Conclusion

Across all five NLP tasks, we implemented and evaluated meaningful task-specific improvements over the Part 1 baselines.

The strongest improvement was obtained for STS, where Pearson correlation increased from `0.371` to `0.819`. QQP development accuracy improved from `0.842` to `0.889`.

For SST, we first evaluated a **3-seed ensemble search** using seeds `11711`, `42`, and `2026`. We then expanded the search to a broader **5-seed sweep** using seeds `11711`, `42`, `2026`, `3407`, and `8848`, combined with multiple sentence-representation heads. From this larger search, the final selected 3-member soft voting ensemble consisted of `seed42-raw_cls`, `seed2026-gated_attn`, and `seed42-pooler`, achieving `0.5431` development accuracy. The broader 5-seed ensemble search achieved a highest development accuracy of approximately `0.5441`. Overall, SST development accuracy improved from the Part 1 baseline of `0.524`.

For PTD, calibrated multi-scale representations substantially improved minority-label detection, increasing macro-averaged MCC from `0.082` to `0.131`. For PTG, structured conditioning and diverse beam-search decoding reduced copying and substantially increased paraphrastic novelty and Penalized BLEU, although this introduced a trade-off with Reference BLEU at later epochs.

Overall, the experiments demonstrate that task-specific representation, optimization, calibration, ensembling, and decoding strategies can improve the original minBERT/BART baselines. The results further show that the most effective improvement depends on the task and evaluation metric, rather than on increasing model complexity alone.

---

# AI-Usage Card

This project was developed in accordance with [AI-Cards](https://ai-cards.org/). A detailed disclosure of AI assistance, human verification protocols, and academic responsibility is provided in the [Group 10 AI-Usage Card (PDF)](./visualisation/AI_Usage_Cards_G10.pdf).

---

# References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL 2019. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805).
3. Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., & Zettlemoyer, L. (2019). *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*. ACL 2020. [arXiv:1910.13461](https://arxiv.org/abs/1910.13461).
4. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084).
5. Loshchilov, I., & Hutter, F. (2017). *Decoupled Weight Decay Regularization*. ICLR 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101).
6. Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., & Potts, C. (2013). *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank*. EMNLP 2013.
7. Vijayan, P. et al. (2021). *Extended Twitter Paraphrase Corpus (ETPC)*.
8. Vijayakumar, A. K., Cogswell, M., Selvaraju, R. R., Sun, Q., Lee, S., Crandall, D., & Batra, D. (2016). *Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models*. [arXiv:1610.02424](https://arxiv.org/abs/1610.02424).
9. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal Loss for Dense Object Detection*. ICCV 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002).
10. Etelis, I., Rosenfeld, A., Weinberg, A. I., & Sarne, D. (2026). Generating effective ensembles for sentiment analysis. *International Journal of Data Science and Analytics*, 22, 98. [https://doi.org/10.1007/s41060-025-00963-0](https://doi.org/10.1007/s41060-025-00963-0).

---

# Acknowledgements
We thank the course teaching team at the University of Göttingen:
* **Lecturer:** [PD Dr. Terry Ruas](https://gipplab.uni-goettingen.de/team/dr-terry-lima-ruas/)
* **Assigned Tutor:** Hamza Ahmed Siddiqui
* **Head TAs & TAs:** Niklas Bauer, Lars Kaesberg, Alina Amanbayeva, Batyrkhan Abukhanov, Emma Stein, Farhan Kayhan, Martina Juhárová, and Tolga Ermis.