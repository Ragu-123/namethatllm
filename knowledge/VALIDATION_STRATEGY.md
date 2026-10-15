# 🛡️ Validation Strategy: Grouped CV Protocol

## 1. The Core Validation Mandate: Unseen Prompts

The most critical realization in **Name That LLM** is that **the private leaderboard test set consists of passages generated from completely unseen prompts**.

If a validation split puts passages answering prompt `p042` into training and another passage answering `p042` into validation:
- The classifier learns strong associations between prompt-specific topical vocabulary (e.g. *"Jevons paradox"*, *"slime mold"*, *"neutron star"*, *"Stern-Gerlach"*) and arbitrary labels.
- This creates **massive topic leakage**, artificially inflating CV scores.
- Models developed under random splits collapse on the actual leaderboard when presented with novel prompts.

---

## 2. StratifiedKFold vs. GroupKFold Empirical Evidence

Public experiments (by Yang Kuang Ou and others) quantitatively proved this divergence:
- **Random Split (StratifiedKFold)**: Tests whether a model can recognize an LLM when the topic is already known/seen in training.
- **Grouped Split (`GroupKFold(n_splits=5, groups=train['prompt_id'])`)**: Tests whether a model can recognize an LLM purely by its stylistic, syntactic, and structural habits on a **never-before-seen topic**.

### The Rule for All Future Experiments:
> **ALL validation, tuning, and model selection MUST use `GroupKFold(n_splits=5)` grouped strictly by `train['prompt_id']`. Random k-fold, stratified k-fold, or train/test splits without prompt grouping are strictly prohibited.**

---

## 3. Standard CV Implementation & Split Recipe

Every experiment must use the following standard setup to ensure all models are directly comparable:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, recall_score, confusion_matrix

def get_grouped_folds(df, n_splits=5, seed=2026):
    """
    Returns a list of (train_idx, val_idx) tuples guaranteed to have
    zero prompt_id overlap between train and validation.
    """
    gkf = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(df, groups=df['prompt_id']))
    
    # Assert zero leakage
    for fold, (tr_idx, va_idx) in enumerate(folds):
        tr_prompts = set(df['prompt_id'].iloc[tr_idx])
        va_prompts = set(df['prompt_id'].iloc[va_idx])
        assert len(tr_prompts & va_prompts) == 0, f"Leakage detected in fold {fold}!"
        
    return folds
```

---

## 4. Evaluation & Diagnostic Metric Standard

When evaluating any fold or out-of-fold (OOF) prediction, always compute and log:

1. **Overall Balanced Accuracy**:
   ```python
   ba = balanced_accuracy_score(y_true, y_pred)
   print(f"Overall OOF Balanced Accuracy: {ba:.5f}")
   ```

2. **Per-Model Recall Breakdown**:
   ```python
   classes = sorted(df['model'].unique())
   recalls = recall_score(y_true, y_pred, labels=classes, average=None)
   recall_df = pd.Series(recalls, index=classes).sort_values()
   print("Per-Model Recall:\n", recall_df.round(4))
   ```

3. **Per-Fold Variance**:
   Log the mean and standard deviation across all 5 folds:
   $$\text{CV} = \mu \pm \sigma$$
   Because prompt difficulty varies across folds, individual fold scores typically range $\pm 0.04$ around the mean.

4. **Saving OOF Predictions**:
   Always save OOF prediction probability arrays (`oof_preds.npy` or `.npz`) along with ground truth labels. This enables ensembling and post-processing calibration.
