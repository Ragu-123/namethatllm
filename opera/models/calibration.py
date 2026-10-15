import numpy as np
from sklearn.metrics import balanced_accuracy_score

def calibrated_tier2_proba(probs, cluster_size, beta=0.35):
    """Cluster-cardinality temperature scaling: T_g = 1 / (1 + beta * log(K_g))."""
    if cluster_size <= 1:
        return np.ones((probs.shape[0], 1), dtype=np.float64)
    temp = 1.0 / (1.0 + beta * np.log(cluster_size))
    p_exp = np.power(np.clip(probs, 1e-12, 1.0), 1.0 / temp)
    return p_exp / p_exp.sum(axis=1, keepdims=True)

def normalize_hierarchical_marginals(oof_hier, test_hier, tau=0.65, alpha_dirichlet=1e-3):
    """Dirichlet-smoothed empirical marginal prior normalization."""
    n_classes = oof_hier.shape[1]
    target_prior = 1.0 / n_classes
    emp_marginal = oof_hier.mean(axis=0)
    scale_factors = ((target_prior + alpha_dirichlet) / (emp_marginal + alpha_dirichlet)) ** tau

    oof_cal = oof_hier * scale_factors[None, :]
    oof_cal /= oof_cal.sum(axis=1, keepdims=True)

    test_cal = test_hier * scale_factors[None, :]
    test_cal /= test_cal.sum(axis=1, keepdims=True)
    return oof_cal, test_cal

def optimize_class_offsets(oof_probs, y_true, n_rounds=4):
    """Direct coordinate-ascent prior optimization on macro-averaged Balanced Accuracy."""
    log_p = np.log(np.clip(oof_probs, 1e-7, 1.0))
    offsets = np.zeros(oof_probs.shape[1], dtype=np.float64)
    best_ba = balanced_accuracy_score(y_true, np.argmax(log_p, axis=1))
    grid = np.linspace(-0.5, 0.5, 25)
    for _ in range(n_rounds):
        improved = False
        for c in range(oof_probs.shape[1]):
            cur_best_val = offsets[c]
            for delta in grid:
                cand = offsets.copy()
                cand[c] = cur_best_val + delta
                preds = np.argmax(log_p + cand[None, :], axis=1)
                score = balanced_accuracy_score(y_true, preds)
                if score > best_ba + 1e-6:
                    best_ba = score
                    offsets[c] = cand[c]
                    improved = True
        if not improved:
            break
    return offsets, best_ba
