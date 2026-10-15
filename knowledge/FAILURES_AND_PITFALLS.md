# ⚠️ Failures, Negative Results & Pitfalls to Avoid

This document records known failures, traps, debugging gotchas, and architectural anti-patterns to prevent repeating past mistakes.

---

## 🚫 Trap 1: Random Split Validation Leakage (StratifiedKFold)
- **The Pitfall**: Using standard `KFold` or `StratifiedKFold` on `train.csv`.
- **Why It Fails**: In `train.csv`, all 18 models answered each prompt. Random splitting places 14 models answering prompt `p050` in the training split and 4 models answering `p050` in the validation split. The classifier memorizes topic keywords (e.g., *"Stern-Gerlach"*, *"epigenetics"*, *"slime mold"*) and achieves artificially high CV scores.
- **The Consequence**: The actual hidden test set contains **unseen prompts**. A model tuned on random CV will suffer a catastrophic score drop on the public/private leaderboard.
- **Enforcement**: **Always** use `GroupKFold(n_splits=5, groups=train['prompt_id'])`.

---

## 🚫 Trap 2: Punctuation Stripping & Aggressive Lowercasing
- **The Pitfall**: Using standard text preprocessing pipelines that do `text.lower()` and regex tokenizers like `\b\w+\b`.
- **Why It Fails**:
  1. Language model fingerprints reside heavily in **punctuation and orthography**:
     - Em-dashes (`—`) vs hyphens (`-`) vs en-dashes (`–`).
     - Curly quotes (`‘’“”`) vs straight quotes (`'"'`).
     - Semicolons (`;`) and colons (`:`).
     - Ellipses (`…` vs `...`).
  2. Capitalization habits (sentence-initial, proper nouns, title case) differ significantly between models.
- **Enforcement**:
  - Keep `lowercase=False` in vectorizers.
  - In Word TF-IDF, use `token_pattern=r"(?u)\b\w+\b|[^\w\s]"` to capture punctuation tokens.
  - Always pair with character-level n-grams (`analyzer='char'` or `analyzer='char_wb'`, n-gram range `(1, 5)`).

---

## 🚫 Trap 3: Submission File Format Mismatch (`.csv` vs `.parquet`)
- **The Pitfall**: Saving predictions to `/kaggle/working/submission.csv`.
- **Why It Fails**: The competition specification mandates `submission.parquet`.
- **The Consequence**: Immediate submission evaluation failure with error `Submission file not found` or `Evaluation Error`.
- **Enforcement**:
  ```python
  submission = pd.DataFrame({'id': test_df['id'], 'model': predictions})
  submission.to_parquet('/kaggle/working/submission.parquet', index=False)
  ```

---

## 🚫 Trap 4: CUDA Out-Of-Memory (OOM) on Kaggle Tesla T4
- **The Pitfall**: Fine-tuning transformer models (e.g. UltraBERT-380M, DeBERTa-v3-large) with `batch_size=16` or higher at `max_length=512`.
- **Why It Fails**: Kaggle Tesla T4 GPUs have 14.6 GB VRAM and lack hardware `bfloat16` support. FP32 or large batch sizes quickly exceed 14.6 GB VRAM.
- **Enforcement**:
  - Use `batch_size=8` with `gradient_accumulation_steps=2` (effective batch size 16).
  - Use `torch.autocast('cuda', dtype=torch.float16)` with `torch.amp.GradScaler`.
  - Delete intermediate tensors and run `torch.cuda.empty_cache()` between folds.

---

## 🚫 Trap 5: Forcing Fixed Class Priors on Test Set
- **The Pitfall**: Assuming the test set contains an exact equal distribution of all 18 models and applying rigid integer Hungarian matching.
- **Why It Fails**: In the test set, the prompt distribution and number of rows may not be an exact multiple of 18, and private test batches may have slight class skews. Hard integer constraints cause brittle edge-case failures.
- **Enforcement**: Apply soft temperature scaling or soft class prior adjustment rather than hard discrete slot filling.

---

## 🚫 Trap 6: Local vs. Remote Path Confusion
- **The Pitfall**: Attempting to read `/kaggle/working/submission.parquet` or check model checkpoints using local Python commands on Windows.
- **Why It Fails**: The Kaggle kernel runs inside an isolated remote Linux Docker container. Files at `/kaggle/working/` exist exclusively on Kaggle until explicitly downloaded.
- **Enforcement**: Use `kaggle_remote_kernel_download_file(remote_path="...")` to transfer files from Kaggle to the local workspace.
