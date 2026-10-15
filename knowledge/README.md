# 🧠 Name That LLM — Knowledge Base & Agent Hub

> **Competition**: [Kaggle: Name That LLM](https://www.kaggle.com/competitions/name-that-llm)  
> **Prize**: $50,000 USD (Featured)  
> **Task**: 18-Class Sequence Classification (Identify which LLM generated a text passage)  
> **Evaluation Metric**: Macro-averaged Balanced Accuracy (`sklearn.metrics.balanced_accuracy_score`)  
> **Submission Format**: `submission.parquet` with columns `['id', 'model']`  
> **Primary Invariant**: **Unseen Prompts at Test Time** $\rightarrow$ Validation MUST use `GroupKFold(n_splits=5, groups=train['prompt_id'])`.

---

## 🚀 Quick Navigation for Agents

If you are an AI agent picking up this task mid-way, **start here**:

| Document | Purpose |
| :--- | :--- |
| [AGENT_HANDOVER.md](file:///c:/Users/SEC/Downloads/kaggle/namethatllm/knowledge/AGENT_HANDOVER.md) | **Step-by-step instructions** to resume work immediately without breaking state. |
| [COMPETITION_OVERVIEW.md](file:///c:/Users/SEC/Downloads/kaggle/namethatllm/knowledge/COMPETITION_OVERVIEW.md) | Core rules, timeline, metric definitions, and kernel constraints. |
| [DATA_SCHEMA_AND_EDA.md](file:///c:/Users/SEC/Downloads/kaggle/namethatllm/knowledge/DATA_SCHEMA_AND_EDA.md) | Full breakdown of `train.csv`, `test.csv`, prompt distribution, and text characteristics. |
| [VALIDATION_STRATEGY.md](file:///c:/Users/SEC/Downloads/kaggle/namethatllm/knowledge/VALIDATION_STRATEGY.md) | Strict CV protocol to prevent topic leakage and ensure leaderboard correlation. |
| [COMMUNITY_INSIGHTS.md](file:///c:/Users/SEC/Downloads/kaggle/namethatllm/knowledge/COMMUNITY_INSIGHTS.md) | Synthesis of discussions, top public notebooks, prompt reconstruction, and techniques. |
| [MODEL_PROFILES_AND_FINGERPRINTS.md](file:///c:/Users/SEC/Downloads/kaggle/namethatllm/knowledge/MODEL_PROFILES_AND_FINGERPRINTS.md) | Detailed analysis of the 18 target models across 6 provider families & stylometric signatures. |
| [EXPERIMENTS_LOG.md](file:///c:/Users/SEC/Downloads/kaggle/namethatllm/knowledge/EXPERIMENTS_LOG.md) | Comprehensive record of every baseline, model iteration, hyperparameter, and CV score. |
| [FAILURES_AND_PITFALLS.md](file:///c:/Users/SEC/Downloads/kaggle/namethatllm/knowledge/FAILURES_AND_PITFALLS.md) | Catalog of what failed, invalid assumptions, leakage bugs, and anti-patterns. |

---

## 📊 Current Leaderboard & Experiment Status

- **Chance / Random Guess Baseline**: $1 / 18 \approx 0.0556$ (5.56%)
- **Host Starter Baseline (Word+Char TF-IDF + LogisticRegression)**: CV ~0.347
- **Stylometric + Char/Word TF-IDF + LinearSVC (Dariush Afshar)**: CV ~0.3685
- **UltraBERT-380M Fine-tuned (dmis-lab)**: 5-Fold OOF CV ~0.408
- **Qwen3-1.7B Hidden States + Centered TF-IDF Blend (Lucas Boesen)**: CV ~0.431

---

## 🛠️ Remote Environment & Execution Rules

1. **Remote Execution Target**: Kaggle Jupyter Server (`2x Tesla T4 GPUs`, 31.3 GB RAM, Linux container).
2. **Golden Execution Invariant**:
   - Write all executable code into the single script file: `c:\Users\SEC\Downloads\kaggle\namethatllm\code.py`.
   - Overwrite `code.py` in-place. **Never create ad-hoc files** like `test1.py`, `script.py`.
   - Dispatch code using `kaggle_remote_kernel_execute(file_path="code.py")`.
   - Live stream and terminal outputs are accessible via the persistent browser log server at `http://localhost:2004/logs`.
3. **Artifact Handling**:
   - Files generated on `/kaggle/working/` (e.g. `submission.parquet`, `.pt` models, plots) stay in the remote Linux container.
   - Use `kaggle_remote_kernel_download_file` to fetch files locally when needed.
