# 🤝 Agent Handover & Continuation Guide

> **Notice to Any Incoming Agent**:  
> Welcome! This repository is configured for autonomous pair programming on the Kaggle competition **Name That LLM**.  
> Everything you need to understand the state, resume experiments, and make submissions is documented below.

---

## 🧭 1. Project State at a Glance

- **Competition**: [Name That LLM](https://www.kaggle.com/competitions/name-that-llm) ($50,000 USD, Featured).
- **Goal**: Classify which of 18 LLMs generated an English text passage (18-way multi-class classification).
- **Data Location**: Mounted on Kaggle at `/kaggle/input/competitions/name-that-llm/`.
- **Target Metric**: **Balanced Accuracy Score** (`sklearn.metrics.balanced_accuracy_score`).
- **Critical Invariant**: **Test passages are from unseen prompts**. All validation must be 5-fold `GroupKFold` on `prompt_id`. Never use random splits.
- **Output Submission**: `/kaggle/working/submission.parquet` with columns `['id', 'model']`.
- **Current Frontrunner**: **OPERA v2.0 (Dual Kernel-FOSP + Cardinality-Scaled Calibration)**:
  - FOSP (Fisher-Orthogonal Style Projection) projects representations into the nullspace of prompt semantics ($\Sigma_P$) via exact $N_{\text{tr}} \times N_{\text{tr}}$ Gram solver ($w_k = X_c^T \alpha_k$).
  - OPERA v1.0 achieved **0.4051 4-fold BA** (Fold 3 reached **0.4259**), up from the 0.3645 baseline.
  - Active execution: `job_3aa80e1b` queued and running on Kaggle.

---

## ⚙️ 2. Execution Protocol & Remote Kaggle Workflow

All code execution is governed by the `kaggle-remote-jupyter` skill and `kaggle-agent-mcp` tools:

1. **Remote Hardware Available**:
   - 2x Tesla T4 GPUs (14.6 GB VRAM each).
   - 31.3 GB System RAM.
   - Kaggle Linux container with pre-installed PyTorch, Transformers, Scikit-Learn.
2. **The `code.py` Single-File Rule**:
   - Write your Python script into the workspace root file: `c:\Users\SEC\Downloads\kaggle\namethatllm\code.py`.
   - **Always overwrite `code.py` in-place.** Never spawn ad-hoc scripts (`temp.py`, `script1.py`).
3. **Execution Command**:
   - Call the tool `kaggle_remote_kernel_execute(file_path="c:\\Users\\SEC\\Downloads\\kaggle\\namethatllm\\code.py")`.
   - For long-running background jobs, call `kaggle_remote_kernel_execute_async` and monitor via `kaggle-job-watch` or `kaggle_remote_kernel_wait_for_job`.
4. **Live Monitoring**:
   - View live streaming logs and running cells at **`http://localhost:2004/logs`**.
5. **Retrieving Artifacts**:
   - To pull files (e.g. `submission.parquet`, plots, models) from `/kaggle/working/` to your local machine:
     ```python
     kaggle_remote_kernel_download_file(remote_path="submission.parquet")
     ```

---

## 🎯 3. Next High-Priority Experiments (Ranked Roadmap)

Here are the highest-ROI experiments ready to be executed:

### 🥇 Priority 1: Baseline Stylometric + Centered Linear Pipeline
- Combine the best of Dariush Afshar (25 stylometric features) with Lucas Boesen's prompt-centering (`group_center`).
- Extract:
  - 1-5 Char-wb TF-IDF (case-sensitive, min_df=2).
  - 1-2 Word TF-IDF (case-sensitive, punctuation preserved).
  - 25 Stylometric cues (em-dash, curly quotes, semicolon, colon, paragraph breaks, uppercase ratio).
- Apply prompt-centering to strip prompt topics:
  $$X_{\text{centered}} = X - \mu_{\text{prompt}}$$
- Train `LinearSVC(C=0.5)` or `LogisticRegression(C=2.0)`.
- Benchmark 5-Fold Grouped CV.

### 🥈 Priority 2: Hierarchical Two-Tier Classification
- The 18 models belong to 6 distinct providers:
  - `Anthropic` (5), `Google` (4), `OpenAI` (5), `ZAI` (2), `Moonshot` (1), `Alibaba` (1).
- **Tier 1 Classifier**: Predicts provider family (6 classes).
- **Tier 2 Classifiers**: 6 specialized intra-family classifiers trained exclusively to resolve subtle differences between sister models (e.g. distinguishing `gemini-3.6-flash` vs `3.7-flash` vs `3.8-flash`).
- Formulate joint probability:
  $$P(\text{model} = m \mid x) = P(\text{provider} = \text{prov}(m) \mid x) \times P(\text{model} = m \mid x, \text{provider} = \text{prov}(m))$$

### 🥉 Priority 3: Deep Feature Extraction with Pretrained Transformer Backbones
- Use pretrained language models to extract hidden states:
  - `Qwen/Qwen3-1.7B-Base` (layers 17 and 20).
  - `dmis-lab/UltraBERT-380M`.
  - `microsoft/deberta-v3-large`.
- Feed extracted pooled embeddings into calibrated linear heads with prompt-centering.

### 🏅 Priority 4: Test-Time Topic Clustering
- Since test passages do not have `prompt_id`, apply hierarchical cosine-distance clustering on test TF-IDF representations.
- For clusters of size $\ge 4$, apply centering using the cluster mean to strip test topic bias before classification!

---

## 📋 4. Pre-Submission Verification Checklist

Before creating a submission or submitting to the leaderboard, verify:
- [ ] GroupKFold CV was evaluated on all 5 folds with zero prompt leakage.
- [ ] Overall Out-of-fold Balanced Accuracy is computed and logged in [EXPERIMENTS_LOG.md](file:///c:/Users/SEC/Downloads/kaggle/namethatllm/knowledge/EXPERIMENTS_LOG.md).
- [ ] Output is saved as a Parquet file: `/kaggle/working/submission.parquet`.
- [ ] Submission columns are strictly `['id', 'model']`.
- [ ] Number of rows in `submission.parquet` matches `len(test_df)`.
- [ ] Zero null or NaN values in `submission['model']`.
- [ ] All predicted labels belong to the 18 valid model strings.
- [ ] Code runs offline without requiring active internet connection during scoring rerun.
