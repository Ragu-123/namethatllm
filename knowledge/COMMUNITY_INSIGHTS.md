# 💡 Community Insights & State-of-the-Art Approaches

## 1. Key Forum Discussions

### 🧵 "Will the winner just be whoever can get the most extra training data?" (CoreyJamesLevinson)
- **Discussion Context**: Since the 18 model names are known frontier and open-weights models (OpenAI, Anthropic, Google, Alibaba, Zhipu, Moonshot), participants can query these model APIs or run open-weights checkpoints to generate tens of thousands of extra training pairs.
- **Strategic Implication**: Generating synthetic training data using the 18 target models on diverse pop-science prompts is a viable path for deep fine-tuning.

### 🧵 "Reconstructed train prompts for all 150 prompt_ids" (pjmathematician)
- **Breakthrough Discovery**:
  - Reconstructed the exact hidden instructions for all 150 `prompt_id`s by prompting Claude Sonnet 5 with the 18 responses per prompt.
  - Released dataset: [`ntl-train-prompts-reconstructed`](https://www.kaggle.com/datasets/pjmathematician/ntl-train-prompts-reconstructed).
  - High self-rated confidence (mean 0.81).
  - Established prompt format: "Explain concept X vividly", covering STEM, philosophy, history, and literature with random length directives.

### 🧵 "Beyond the pangram experience" (Big Sur)
- **Analysis**: Off-the-shelf AI detectors designed for long-form essays perform poorly on single short paragraphs (40–200 words). Stylometric feature engineering (micro-punctuation, syntax trees, token frequencies) outperforms generic perplexity-based detectors.

---

## 2. Top Public Notebook Architectures & Benchmarks

### 1. Host Starter Baseline (`martynaplomecka`)
- **Features**:
  - Word TF-IDF: n-grams (1, 2), max_features=25,000, sublinear_tf=True
  - Char TF-IDF: analyzer='char', n-grams (3, 5), max_features=35,000, sublinear_tf=True
- **Classifier**: `LogisticRegression(C=2.0, max_iter=1000)`
- **CV Result**: 5-Fold Grouped CV Balanced Accuracy $\approx \mathbf{0.347}$

### 2. Stylometric + TF-IDF + LinearSVC (`dariushafshar`)
- **Key Innovation**: Added 25 explicit stylometric features:
  - Punctuation counts: em-dashes (`—`), en-dashes (`–`), curly quotes (`‘’“”`), semicolons (`;`), colons (`:`), ellipses (`…`, `...`).
  - Formatting cues: paragraph count (`\n\n`), newline density, uppercase ratio.
  - Syntactic proxies: mean word length, question marks, exclamation marks.
- **Model**: `LinearSVC(C=1.0)` with scaled stylometric features + Char/Word TF-IDF.
- **CV Result**: 5-Fold Grouped CV Balanced Accuracy $\approx \mathbf{0.3685}$

### 3. UltraBERT-380M Fine-Tuning (`pjmathematician`)
- **Architecture**: [`dmis-lab/UltraBERT-380M`](https://huggingface.co/dmis-lab/UltraBERT-380M) sequence classification head.
- **Training Details**:
  - AdamW ($\beta_1=0.9, \beta_2=0.98, \text{wd}=0.1$), lr = $1\times 10^{-5}$, 5 epochs.
  - Effective batch size 16 (8 x 2 accumulation), max length 512 tokens.
  - Evaluated on 5 GroupKFold folds.
- **Results Across Folds**:
  - Fold 0: 0.404
  - Fold 1: 0.426
  - Fold 2: 0.357
  - Fold 3: 0.404
  - Fold 4: 0.451
  - **OOF Balanced Accuracy**: $\mathbf{0.408}$ (40.8%)

### 4. Qwen3-1.7B Hidden Embeddings + Prompt-Centering (`lucasboesen`)
- **Breakthrough Concept: Prompt-Centering (`group_center`)**:
  - In training, 18 models answer the exact same prompt. All topical words ("neutron star", "Jevons paradox") are shared.
  - By subtracting the mean embedding/feature vector of each prompt group:
    $$X_{\text{centered}} = X - \mu_{\text{prompt}}$$
    topical signals vanish, leaving **pure model-specific stylistic residuals**!
- **Features**:
  - Mean-pooled hidden states from layers 17 and 20 of `Qwen/Qwen3-1.7B-Base`.
  - Word + Char TF-IDF with case and punctuation preserved.
  - Individual Logistic Regression heads on standardized features.
- **Ensemble**: Weighted log-probability blend.
- **CV Result**: 5-Fold Grouped CV Balanced Accuracy $\approx \mathbf{0.431}$ (Current Public SOTA).

---

## 3. High-Value Modeling Directions

1. **Prompt-Centering Generalization**: How to handle test passages where prompt clusters are smaller or unknown (e.g. cosine-distance topic clustering).
2. **Multi-Model Embedding Fusion**: Combining hidden states from distinct architectures (e.g. Qwen3-1.7B + UltraBERT-380M + DeBERTa-v3-large).
3. **Dedicated Within-Provider Resolvers**: Cascaded architecture:
   - Stage 1: Predict provider family (Claude vs Gemini vs GPT vs GLM vs Kimi vs Qwen) — high accuracy expected.
   - Stage 2: Specialized intra-family classifiers (e.g., distinguishing `gemini-3.6-flash` vs `gemini-3.7-flash` vs `gemini-3.8-flash`).
4. **Synthetic Data Augmentation**: Expanding training set using reconstructed prompt styles.
