# Name That LLM: OPERA Algorithm Repository

Code and methodology for the Kaggle **"Name That LLM"** Competition.

## Algorithm Overview: OPERA (Orthogonal Prompt-Eliminated Residual Attribution)
Current public benchmarks hit a ceiling at ~0.4310 Balanced Accuracy due to the **unseen-prompt train-test covariate shift** and **sister-model collapse** (e.g., Claude-Fable vs. Claude-Opus, GPT-5.6-Sol vs. GPT-5.6-Luna).

OPERA solves this via:
1. **Pillar 1: RKHS Non-Linear Kernel-FOSP**:
   - Generalized eigenvalue prompt-annihilation projection in $N_{\text{tr}} \times N_{\text{tr}}$ dual Gram space.
   - Exact pre-centered dual weights $\tilde{\mathbf{A}}$ and kernel bias $b_{\text{kern}}$ enabling $O(N_{\text{tr}} N_{\text{te}})$ inference without test prompt labels.
2. **Pillar 2: CPNL Logarithmic Opinion Pool (LogOP)**:
   - High-confidence Markov likelihood routing for Tier-1 corporate provider discrimination with length normalization and punctuation gating.
3. **Pillar 3: Contrastive ASTD (Autoregressive Stylometric Trajectory Dynamics)**:
   - Token-by-token surprisal, subword fertility, and boundary entropy tracking.
4. **Pillar 4: Scale-Free Laplacian Topic Centering**:
   - Top-17 capped neighborhood with Kish's $N_{\text{eff}}$ empirical shrinkage.
5. **Pillar 5: Tier-2 Intra-Family Resolvers**:
   - Cardinality temperature scaling ($T_g = \frac{1}{1 + \beta \log K_g}$) + Dirichlet marginal prior normalization + coordinate-ascent calibration.

## Empirical Benchmark Performance (5-Fold Grouped CV, Zero Topic Leakage)
- **Baseline (LinearSVC)**: 0.3645 Balanced Accuracy
- **Community Public SOTA (Frozen Qwen3 + SVM)**: 0.4310 Balanced Accuracy
- **OPERA v2.2 (Dual Kernel-FOSP)**: 0.4190 Balanced Accuracy
- **OPERA v3.0 (RKHS Non-Linear Kernel-FOSP + LogOP + ASTD)**: **0.49570 Balanced Accuracy**
  - Fold 1: 0.45741
  - Fold 2: 0.44630
  - Fold 3: 0.48148
  - Fold 4: 0.46852
  - Fold 5: 0.47861
  - Claude-Fable-5 Recall: **40.27%** (up from 5.4%)
  - GPT-5.6-Sol Recall: **20.67%** (up from 12.7%)
  - Gemini-3.1-Pro-Preview Recall: **82.00%**

## Neural Backbone Integration: Qwen 3 (1.7B-Base)
Model path: `/kaggle/input/models/qwen-lm/qwen-3/transformers/1.7b-base/1`
Features: Mean-pooled hidden states from layers 17 and 20 projected through the non-linear FOSP style subspace operator.
