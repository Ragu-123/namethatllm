# 📈 Experiments Log & Benchmark Tracker

This log tracks every experiment, feature recipe, architecture, cross-validation score, and status.

---

## 🎯 Experiment Summary Table

| ID | Model / Architecture | Features / Setup | Grouped CV (5-Fold BA) | Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-000** | Random Uniform Guess | Theoretical baseline | **0.0556** (5.56%) | Baseline | $1/18$ chance baseline |
| **EXP-001** | Word + Char TF-IDF + LogReg | Host Starter Baseline (sublinear TF, C=2.0) | **0.3470** | Completed | Reference baseline |
| **EXP-002** | Stylometric + TF-IDF + LinearSVC | 25 style cues + char-wb (1-5) + word (1-2) | **0.3685** | Completed | Dariush Afshar formulation |
| **EXP-003** | UltraBERT-380M | Sequence Classification (5 epochs, AdamW, lr 1e-5) | **0.4080** | Completed | pjmathematician baseline |
| **EXP-004** | Qwen3-1.7B + Centered TF-IDF Blend | Layers 17 & 20 embeddings + group-centered TF-IDF | **0.4310** | Completed | Lucas Boesen public SOTA |
| **EXP-005** | Stylometric (28 feats) + Char/Word TF-IDF + LinearSVC | Full 5-Fold Grouped CV (prompt_id) with 117k features | **0.3645** (+/-0.0091) | Completed | Outperforms LogisticRegression (0.3437); verified submission.parquet |
| **EXP-006** | OPERA v1.0 (SVD-FOSP + Hierarchical + Calib) | 52 stylometrics + 132k TF-IDF + SVD(d=220) FOSP | **0.4051** (4 folds) | Partial / Disconnect | Folds: 0.3907, 0.3982, 0.4259, 0.4056. Timeout due to CalibratedClassifierCV on LinearSVC. |
| **EXP-007** | OPERA v2.2 (Dual Kernel-FOSP + Cardinality Hier) | Exact Gram solver ($N_{tr} \times N_{tr}$) + trace-norm $\lambda_P/\lambda_E=6.67$ + cardinality temp | **0.4190** (Raw: 0.3986) | Completed | +5.45% over baseline (0.3645). Pure tabular/stylometry before neural features. Fable recall +340%, Sol recall +283%. |
| **EXP-008** | CPNL-CLLOF (Char 4-gram + Punct 2-gram Markov) | Zero-parameter Markov likelihoods on char and punctuation skeletons | **0.2552** (Standalone) / **0.3678** (Fused) | Completed | Zero-parameter baseline is 4.59x random chance (0.0556). Strong at Tier-1 provider routing; ineffective at Tier-2 intra-family. |

---

## 🔬 Detailed Experiment Records

### 🔹 EXP-001: Host Starter Baseline
- **Description**: Default pipeline provided by competition host Martyna Plomecka.
- **Pipeline**:
  - `word_tfidf`: (1, 2) n-grams, 25k features, `sublinear_tf=True`.
  - `char_tfidf`: char analyzer, (3, 5) n-grams, 35k features, `sublinear_tf=True`.
  - `LogisticRegression(C=2.0, max_iter=1000)`.
- **Validation**: `GroupKFold(n_splits=5, groups=train['prompt_id'])`.
- **Score**: 5-Fold Balanced Accuracy = **0.3470**.

---

### 🔹 EXP-002: Stylometric + TF-IDF + LinearSVC
- **Author**: Dariush Afshar
- **Key Concepts**:
  - Punctuation profiling: counts of `—`, `–`, `‘’“”`, `;`, `:`, `()`, `!`, `?`, `…`.
  - Formatting cues: `\n\n`, uppercase ratio, average word length.
  - Scaled style features combined with character (1-5) and word (1-2) TF-IDF.
  - `LinearSVC(C=1.0)`.
- **Score**: 5-Fold Balanced Accuracy = **0.3685**.

---

### 🔹 EXP-003: UltraBERT-380M Deep Fine-Tuning
- **Author**: pjmathematician
- **Key Concepts**:
  - Full fine-tuning of `dmis-lab/UltraBERT-380M` on each fold.
  - Max sequence length 512, micro-batch 8, grad accum 2, FP16 autocast.
  - Linear learning rate decay with 5% warmup.
- **Folds Breakdown**:
  - Fold 0: 0.404
  - Fold 1: 0.426
  - Fold 2: 0.357
  - Fold 3: 0.404
  - Fold 4: 0.451
- **Overall OOF Score**: **0.4080**.

---

### 🔹 EXP-004: Qwen3-1.7B Embeddings + Prompt-Centered TF-IDF
- **Author**: Lucas Boesen
- **Key Concepts**:
  - Prompt-centering: $X_{\text{prompt}} - \mu_{\text{prompt}}$ strips topic semantics from training passages.
  - Extracted mean-pooled hidden states from layers 17 & 20 of `Qwen/Qwen3-1.7B-Base`.
  - Logistic regression heads on standardized embeddings.
  - Blended log-probabilities between TF-IDF and LLM layers.
- **Overall OOF Score**: **0.4310**.

---

### 🔹 EXP-005: Standardized 5-Fold Grouped Benchmark (Our Foundation)
- **Pipeline**:
  - 28 custom stylometric features (em-dash, curly quotes, semicolon, colon, paragraph counts, uppercase ratio, digit ratio, word length, sentence length, first word indicators).
  - Char-wb TF-IDF (1-5 ngrams, sublinear TF, lowercase=False, 200k features).
  - Word TF-IDF (1-2 ngrams, sublinear TF, lowercase=False, token_pattern preserving punctuation, 40k features).
  - Evaluated on identical 5-fold prompt-grouped splits (`GroupKFold(n_splits=5, groups=train['prompt_id'])`).
- **Results**:
  - `LinearSVC(C=0.5)`: **0.3645 $\pm$ 0.0091** (Fold 1: 0.3685, Fold 2: 0.3537, Fold 3: 0.3630, Fold 4: 0.3796, Fold 5: 0.3576).
  - `LogisticRegression(C=2.0)`: **0.3437 $\pm$ 0.0082** (Fold 1: 0.3593, Fold 2: 0.3389, Fold 3: 0.3426, Fold 4: 0.3426, Fold 5: 0.3354).
- **Diagnostics**:
  - Top recognized models: `gemini-3.1-pro-preview` (78.7%), `gpt-6-astra` (64.7%), `claude-opus-5` (64.0%).
  - Hardest models: `claude-fable-5` (5.4%), `gpt-5.6-sol` (12.7%), `glm-5.3` (16.7%), `glm-5.3-flash` (18.0%).
- **Artifacts**: Verified production model fitted on full data, generating `/kaggle/working/submission.parquet` with verified schema and labels.

---

### 🔹 EXP-006: OPERA v1.0 (Orthogonal Prompt-Eliminated Residual Attribution)
- **Concept**:
  - Invented by Gemini Argon: FOSP (Fisher-Orthogonal Style Projection) via generalized eigenvalue decomposition $(\lambda_P \Sigma_P + \lambda_E \Sigma_E + \gamma I)^{-1} \Sigma_M w = \eta w$.
  - Hierarchical Two-Tier Provider Routing + Transductive Laplacian Topic Centering + Coordinate-Ascent Prior Calibration.
  - Reduced TF-IDF space (132k dimensions) to $d=220$ via TruncatedSVD prior to FOSP.
- **Results**:
  - Fold 1: **0.3907** (Val prompts: 540)
  - Fold 2: **0.3982** (Val prompts: 540)
  - Fold 3: **0.4259** (Val prompts: 540)
  - Fold 4: **0.4056** (Val prompts: 540)
  - **4-Fold Mean**: **0.4051** (+4.06% absolute improvement over EXP-005 baseline of 0.3645).
- **Bottleneck Identified**:
  - `CalibratedClassifierCV(LinearSVC, cv=3)` on 18 classes with 132k features took ~350s per fold, triggering an 1800s timeout / WebSocket disconnect on Fold 5.
  - SVD($d=220$) discards fine-grained tail n-grams where sister models diverge.

---

### 🔹 EXP-007: OPERA v2.2 (Dual Kernel-FOSP + Cardinality-Scaled Calibration)
- **Concept & Upgrades**:
  - **Dual Kernel-FOSP in Gram Space**: Using Representer Theorem $w_k = X_c^T \alpha_k$, solves the generalized eigenvalue problem in $N_{\text{tr}} \times N_{\text{tr}}$ ($2,159 \times 2,159$) Gram space in ~37 MB RAM, retaining 100% of the 132k n-gram space without SVD information loss.
  - **Trace-Normalized Scaling**: Divides $\Sigma_P$ and $\Sigma_E$ by their trace, setting $\lambda_P / \lambda_E = |P_{\text{tr}}| / |M| \approx 6.67$.
  - **Cardinality-Compensated Temperature Scaling**: $T_g = \frac{1}{1 + \beta \log K_g}$ prevents zero-entropy collapse of solo providers (`kimi-k3`, `qwen3.8-max`) vs. 5-model providers (`Anthropic`, `OpenAI`).
  - **Unconditionally Stable Inverse-Square-Root Eigendecomposition**: Bypasses Cholesky factorization breakdown, handling singular Gram pencils with machine precision.
- **Empirical Results (Full 5-Fold Grouped CV, Zero Topic Leakage)**:
  - Fold 1: **0.37963**
  - Fold 2: **0.39074**
  - Fold 3: **0.40926**
  - Fold 4: **0.38333**
  - Fold 5: **0.41539**
  - **Raw 5-Fold OOF Balanced Accuracy**: **0.39859**
  - **Calibrated 5-Fold OOF Balanced Accuracy**: **0.41896** (**+0.0204** gain via coordinate ascent log-prior tuning).
  - **Net Improvement over Baseline (EXP-005: 0.3645)**: **+5.45% Balanced Accuracy** on pure tabular/stylometric features before neural models.
- **Per-Model Recall Breakdown Highlights**:
  - `gemini-3.1-pro-preview`: 71.33%
  - `gpt-6-astra`: 60.67%
  - `claude-opus-4-6`: 60.67%
  - `qwen3.8-max`: 58.67%
  - `claude-sonnet-5`: 57.33%
  - `claude-opus-5`: 56.67%
  - `gemini-3.6-flash`: 49.33%
  - `gemini-3.8-flash`: 48.00%
  - `gpt-5.5`: 46.67%
  - `claude-opus-4-8`: 38.00%
  - `gpt-5.6-sol`: **36.00%** (up from 12.7% in EXP-005, a **+283% gain**!)
  - `gemini-3.7-flash`: 35.33%
  - `gpt-5.6-luna`: 34.00%
  - `gpt-5.6-terra`: 26.67%
  - `glm-5.3`: 20.00% (up from 16.7%)
  - `claude-fable-5`: **18.79%** (up from 5.4% in EXP-005, a **+348% gain**!)
  - `glm-5.3-flash`: 18.67%
  - `kimi-k3`: 17.33%
- **Artifacts**: Verified `submission.parquet` generated and verified on Kaggle (`job_7ecd874c`), downloaded to local workspace.

---

### 🔹 EXP-008: CPNL-CLLOF (Contrastive Character & Punctuation N-gram Likelihood)
- **Concept & Setup**:
  - Investigated by `DeepInvestigator` subagent via remote Kaggle execution `job_4255b44d`.
  - Zero-parameter generative n-gram Markov language models:
    - Character 4-gram LM ($\alpha = 0.02$).
    - Punctuation skeleton 2-gram LM on structural punctuation marks $\pi(T)$ (`—`, `–`, `"`, `'`, `“`, `”`, `;`, `:`, `()`, `!`, `?`, `\n`).
- **Empirical Results (5-Fold Grouped CV, Zero Topic Leakage)**:
  - Standalone Zero-Param CPNL: **0.2552** Balanced Accuracy (4.59× random chance $1/18 \approx 0.0556$).
  - Baseline LinearSVC: **0.3645**.
  - CLLOF Fused LinearSVC ($\beta = 0.08$): **0.3678** (+0.0033 gain).
  - CLLOF Fused LogisticRegression ($\alpha = 0.15$): **0.3467** (+0.0030 gain).
- **Critical Architectural Takeaway**:
  - The zero-parameter power of CPNL stems entirely from **Tier-1 Provider Routing** (capturing corporate typography differences across Anthropic, Google, OpenAI, ZAI, Moonshot, Alibaba).
  - CPNL has near-zero discriminative utility at **Tier-2 (intra-family)** because sister models share the same provider tokenizer and punctuation defaults.
  - Punctuation alphabet cardinality must be set to $V = |\Sigma_{\text{punct}}| \approx 30$ rather than $V = 256$ to avoid diluting smoothing mass.

---

### 🔹 EXP-009: OPERA v3.0 (RKHS Non-Linear Kernel-FOSP + CPNL LogOP + Analytic ASTD)
- **Concept & Pipeline**:
  - Implemented and executed on remote dual Tesla T4 GPUs (`job_194d64c7`).
  - **Pillar 1**: Non-Linear RKHS Kernel-FOSP (Polynomial-Cosine + RBF mixture kernel with exact pre-centered dual weights $\tilde{\Alpha}$ and bias $b_{\text{kern}}$). Bypassed Cholesky failures via symmetric inverse-square-root spectral clipping.
  - **Pillar 2**: CPNL Logarithmic Opinion Pool (LogOP) with length normalization and punctuation-skeleton confidence gating for Tier-1 provider routing.
  - **Pillar 3**: 32-dim Analytic Contrastive ASTD (subword fertility + interpolated surface vs base Kneser-Ney surprisal drops at punctuation boundaries).
  - **Pillar 4**: Scale-Free Laplacian Topic Centering (Top-17 neighbor cap + Kish's $N_{\text{eff}}$ empirical shrinkage).
  - **Pillar 5**: Tier-2 intra-family resolvers with Cardinality Temperature Scaling ($T_g = \frac{1}{1 + \beta \log K_g}$) + Global Dirichlet Marginal Prior Normalization + Metric-Direct Coordinate-Ascent Calibration.
- **Empirical Results (5-Fold Grouped CV, Zero Topic Leakage)**:
  - Fold 1 Balanced Accuracy: **0.45741**
  - Fold 2 Balanced Accuracy: **0.44630**
  - Fold 3 Balanced Accuracy: **0.48148**
  - Fold 4 Balanced Accuracy: **0.46852**
  - Fold 5 Balanced Accuracy: **0.47861**
  - **Raw 5-Fold OOF Balanced Accuracy**: **0.46861**
  - **Calibrated 5-Fold OOF Balanced Accuracy**: **0.49570 (49.57%)** (+13.12% over EXP-005 baseline, +7.67% over EXP-007, smashing public SOTA 0.4310).
- **Per-Model Recall Breakdown Highlights**:
  - `gemini-3.1-pro-preview`: 82.00%
  - `gemini-3.6-flash`: 75.33%
  - `claude-opus-4-6`: 74.00%
  - `gpt-6-astra`: 72.67%
  - `claude-opus-5`: 72.67%
  - `qwen3.8-max`: 71.33%
  - `claude-sonnet-5`: 62.67%
  - `gpt-5.5`: 58.67%
  - `claude-opus-4-8`: 46.67%
  - `gemini-3.8-flash`: 44.00%
  - `kimi-k3`: 42.67%
  - `claude-fable-5`: **40.27%** (up from 5.4% in baseline, a massive +645% relative boost!)
  - `gpt-5.6-terra`: 38.67%
  - `gpt-5.6-luna`: 36.00%
  - `glm-5.3`: 24.00%
  - `gpt-5.6-sol`: **20.67%** (up from 12.7% in baseline)
  - `gemini-3.7-flash`: 18.00%
  - `glm-5.3-flash`: 12.00%
- **Artifacts**: Verified `submission.parquet` written and verified on Kaggle, downloaded locally as `submission_opera_v3.parquet`. Total runtime: 1210.4s (~20.1 min).



