# 🏆 Competition Overview: Name That LLM

## 1. Problem Statement
With the explosion of frontier and open-weights Large Language Models, each model develops subtle stylistic, orthographic, structural, and semantic "fingerprints" based on its pre-training corpora, tokenizer, RLHF/alignment tuning, system prompts, and decoding defaults.

The goal of **Name That LLM** is:
> **Given an unlabeled passage of English text generated in response to an unseen prompt, correctly predict which of the 18 target LLMs authored the text.**

- **Competition ID**: `170484`
- **Slug**: `name-that-llm`
- **URL**: [https://www.kaggle.com/competitions/name-that-llm](https://www.kaggle.com/competitions/name-that-llm)
- **Prize Pool**: $50,000 USD
- **Timeline**:
  - Launch Date: October 14, 2026
  - Entry & Team Merger Deadline: January 6, 2027
  - Final Submission Deadline: January 14, 2027

---

## 2. Evaluation Metric: Balanced Accuracy Score

The official evaluation metric is **Balanced Accuracy** (as implemented in `sklearn.metrics.balanced_accuracy_score`):

$$\text{Balanced Accuracy} = \frac{1}{K} \sum_{k=1}^{K} \text{Recall}_k = \frac{1}{K} \sum_{k=1}^{K} \frac{TP_k}{TP_k + FN_k}$$

where:
- $K = 18$ (the number of target LLM classes).
- $TP_k$ is the number of true positive predictions for class $k$.
- $FN_k$ is the number of false negative predictions for class $k$.
- If every sample is guessed as a single class, $BA = 1/18 \approx 0.0556$.
- If a model predicts random uniform guesses, expected $BA \approx 0.0556$.

### Critical Strategic Implications of Balanced Accuracy:
1. **Equal Weight for Every Model**: Regardless of test set class frequencies (which may vary slightly between public and private test sets), each of the 18 models contributes exactly $\frac{1}{18} \approx 5.56\%$ to the final score.
2. **Penalizes Neglecting Difficult Classes**: If our model yields 0% recall on 3 hard models (e.g. confusing `gemini-3.6-flash`, `gemini-3.7-flash`, and `gemini-3.8-flash`), our theoretical maximum score immediately drops from $1.00$ to $15/18 \approx 0.833$.
3. **Threshold / Prior Calibration**: If class probabilities or decision thresholds favor majority/easy classes, balanced accuracy drops. Threshold tuning or class balancing on out-of-fold validation is highly impactful.

---

## 3. Submission Protocol & Requirements

- **Type**: Code Competition (Notebooks/Kernels Submission Only).
- **Internet Access**: Disabled during scoring rerun. All models, tokenizers, weights, and packages must be pre-packaged in Kaggle Datasets or pip wheels.
- **Output File**: Must generate `submission.parquet` (NOT `.csv`) in `/kaggle/working/submission.parquet`.
- **Columns**: Exactly two columns:
  - `id`: string, matching the test row identifiers.
  - `model`: string, one of the 18 valid model label strings.
- **Example Format**:
  ```python
  import pandas as pd
  sub = pd.DataFrame({'id': test_df['id'], 'model': test_preds})
  sub.to_parquet('submission.parquet', index=False)
  ```
- **Max Daily Submissions**: 5 submissions per day.

---

## 4. Hardware & Runtime Constraints

- **Execution Environment**:
  - Kaggle Linux Docker Container (`gcr.io/kaggle-images/python`).
  - Available Accelerators:
    - **Dual Tesla T4 GPUs** (`2 x 14.6 GB = 29.2 GB VRAM`).
    - Alternatively: Single Nvidia P100 (16 GB) or TPU v3-8 (if configured).
  - RAM: ~31.3 GB System Memory.
  - Disk: 20 GB writable scratch space on `/kaggle/working/`.
- **Inference Runtime Limit**: Typically 9 hours for GPU sessions. Because test set contains hidden evaluation rows, inference must be vectorized, batch-processed, and lightweight.
