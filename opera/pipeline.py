import os
import time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import balanced_accuracy_score, recall_score

from .config import PROVIDER_MAP, CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, PROVIDERS, PROV_TO_IDX
from .features import extract_stylometric_matrix, AnalyticContrastiveASTD, extract_qwen_features
from .models import (
    CPNLProviderLogOP,
    calibrated_tier2_proba,
    normalize_hierarchical_marginals,
    optimize_class_offsets,
)

def run_opera_pipeline(data_dir, qwen_dir=None, output_dir=".", use_neural=True):
    print("=== OPERA v4.1: SOTA Dual-Branch Pipeline (Zero Distortion) ===", flush=True)
    t_start = time.time()

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    test_df["text"] = test_df["text"].fillna("").astype(str)
    print(f"Loaded data: Train={train_df.shape}, Test={test_df.shape}", flush=True)

    # 1. Feature Extraction (Identical transformations for Train and Test)
    print("\n--- Extracting Stylometric & Analytic Features ---", flush=True)
    X_style_tr_raw = extract_stylometric_matrix(train_df["text"].tolist())
    X_style_te_raw = extract_stylometric_matrix(test_df["text"].tolist())
    print(f"Stylometrics: train={X_style_tr_raw.shape}, test={X_style_te_raw.shape}", flush=True)

    astd_model = AnalyticContrastiveASTD()
    astd_model.fit(train_df["text"].tolist())
    X_astd_tr_raw = astd_model.transform(train_df["text"].tolist())
    X_astd_te_raw = astd_model.transform(test_df["text"].tolist())
    print(f"Analytic-ASTD: train={X_astd_tr_raw.shape}, test={X_astd_te_raw.shape}", flush=True)

    print("Building high-capacity character and word TF-IDF vectorizers...", flush=True)
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(1, 5), min_df=2, max_features=250000,
        sublinear_tf=True, lowercase=False
    )
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_features=40000,
        sublinear_tf=True, lowercase=False, token_pattern=r"(?u)\b\w+\b|[^\w\s]"
    )

    X_char_tr = char_vec.fit_transform(train_df["text"])
    X_char_te = char_vec.transform(test_df["text"])
    X_word_tr = word_vec.fit_transform(train_df["text"])
    X_word_te = word_vec.transform(test_df["text"])

    X_tfidf_tr = sparse.hstack([X_char_tr, X_word_tr], format="csr")
    X_tfidf_te = sparse.hstack([X_char_te, X_word_te], format="csr")
    X_tfidf_tr.data = np.sqrt(np.maximum(X_tfidf_tr.data, 0.0))
    X_tfidf_te.data = np.sqrt(np.maximum(X_tfidf_te.data, 0.0))
    print(f"Hellinger TF-IDF space: {X_tfidf_tr.shape[1]:,} features", flush=True)

    has_qwen = False
    if use_neural and qwen_dir and os.path.exists(qwen_dir):
        print(f"\n--- Extracting Neural Qwen 3 (1.7B-Base) Representations ---", flush=True)
        cache_tr = os.path.join(output_dir, "qwen_train_emb.npz")
        cache_te = os.path.join(output_dir, "qwen_test_emb.npz")
        qwen_h17_tr, qwen_h20_tr = extract_qwen_features(train_df["text"].tolist(), qwen_dir, cache_path=cache_tr)
        qwen_h17_te, qwen_h20_te = extract_qwen_features(test_df["text"].tolist(), qwen_dir, cache_path=cache_te)
        print(f"Qwen representations ready: Layer 17={qwen_h17_tr.shape}, Layer 20={qwen_h20_tr.shape}", flush=True)
        has_qwen = True

    # 2. 5-Fold Grouped Cross-Validation Protocol
    y_all = np.array([CLASS_TO_IDX[m] for m in train_df["model"]])
    prov_all = np.array([PROV_TO_IDX[PROVIDER_MAP[m]] for m in train_df["model"]])
    prompts_all = train_df["prompt_id"].values
    n_classes = len(CLASSES)

    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(train_df, y_all, groups=prompts_all))

    oof_svm = np.zeros((len(train_df), n_classes), dtype=np.float64)
    oof_qwen17 = np.zeros((len(train_df), n_classes), dtype=np.float64)
    oof_qwen20 = np.zeros((len(train_df), n_classes), dtype=np.float64)
    oof_hier = np.zeros((len(train_df), n_classes), dtype=np.float64)

    test_svm = np.zeros((len(test_df), n_classes), dtype=np.float64)
    test_qwen17 = np.zeros((len(test_df), n_classes), dtype=np.float64)
    test_qwen20 = np.zeros((len(test_df), n_classes), dtype=np.float64)
    test_hier = np.zeros((len(test_df), n_classes), dtype=np.float64)

    print("\n--- Commencing 5-Fold Grouped Cross-Validation (Zero Pseudo-Centering) ---", flush=True)

    for fold, (tr_idx, va_idx) in enumerate(folds):
        t_f = time.time()
        tr_prompts = prompts_all[tr_idx]
        va_prompts = prompts_all[va_idx]
        assert len(set(tr_prompts) & set(va_prompts)) == 0, f"Prompt leakage in fold {fold}!"

        # Strictly fit standardizers on training folds only
        scaler_s = StandardScaler()
        S_tr = scaler_s.fit_transform(X_style_tr_raw[tr_idx])
        S_va = scaler_s.transform(X_style_tr_raw[va_idx])
        S_te = scaler_s.transform(X_style_te_raw)

        scaler_a = StandardScaler()
        A_tr = scaler_a.fit_transform(X_astd_tr_raw[tr_idx])
        A_va = scaler_a.transform(X_astd_tr_raw[va_idx])
        A_te = scaler_a.transform(X_astd_te_raw)

        Dense_tr = np.hstack([S_tr, A_tr])
        Dense_va = np.hstack([S_va, A_va])
        Dense_te = np.hstack([S_te, A_te])

        # Combined features for Branch 1 (TF-IDF + Stylometrics + ASTD)
        X_tr_svm = sparse.hstack([X_tfidf_tr[tr_idx], sparse.csr_matrix(Dense_tr * 0.3)], format="csr")
        X_va_svm = sparse.hstack([X_tfidf_tr[va_idx], sparse.csr_matrix(Dense_va * 0.3)], format="csr")
        X_te_svm = sparse.hstack([X_tfidf_te, sparse.csr_matrix(Dense_te * 0.3)], format="csr")

        # Head 1: Calibrated LinearSVC Head
        base_svc = LinearSVC(C=1.0, max_iter=5000, random_state=42 + fold)
        cal_svc = CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv="prefit")
        base_svc.fit(X_tr_svm, y_all[tr_idx])
        cal_svc.fit(X_tr_svm, y_all[tr_idx])

        oof_svm[va_idx] = cal_svc.predict_proba(X_va_svm)
        test_svm += cal_svc.predict_proba(X_te_svm) / len(folds)

        # Head 2 & 3: Qwen 3 Layers 17 & 20 Heads
        if has_qwen:
            sc_17 = StandardScaler()
            Q17_tr = sc_17.fit_transform(qwen_h17_tr[tr_idx])
            Q17_va = sc_17.transform(qwen_h17_tr[va_idx])
            Q17_te = sc_17.transform(qwen_h17_te)

            clf_q17 = LogisticRegression(C=0.005, max_iter=5000, random_state=42 + fold)
            clf_q17.fit(Q17_tr, y_all[tr_idx])
            oof_qwen17[va_idx] = clf_q17.predict_proba(Q17_va)
            test_qwen17 += clf_q17.predict_proba(Q17_te) / len(folds)

            sc_20 = StandardScaler()
            Q20_tr = sc_20.fit_transform(qwen_h20_tr[tr_idx])
            Q20_va = sc_20.transform(qwen_h20_tr[va_idx])
            Q20_te = sc_20.transform(qwen_h20_te)

            clf_q20 = LogisticRegression(C=0.005, max_iter=5000, random_state=42 + fold)
            clf_q20.fit(Q20_tr, y_all[tr_idx])
            oof_qwen20[va_idx] = clf_q20.predict_proba(Q20_va)
            test_qwen20 += clf_q20.predict_proba(Q20_te) / len(folds)

        # Head 4: Tier-1 Provider Router & Intra-Family Resolver
        clf_tier1 = LogisticRegression(C=2.0, max_iter=1000, random_state=42 + fold)
        clf_tier1.fit(X_tr_svm, prov_all[tr_idx])
        prov_probs_va_disc = clf_tier1.predict_proba(X_va_svm)
        prov_probs_te_disc = clf_tier1.predict_proba(X_te_svm)

        cpnl_router = CPNLProviderLogOP(n_providers=len(PROVIDERS))
        cpnl_router.fit_markov(train_df["text"].iloc[tr_idx].tolist(), prov_all[tr_idx])
        b_c_va, b_p_va, g_va = cpnl_router.compute_bayes_factors(train_df["text"].iloc[va_idx].tolist())
        b_c_te, b_p_te, g_te = cpnl_router.compute_bayes_factors(test_df["text"].tolist())

        prov_probs_va = cpnl_router.fuse_proba(prov_probs_va_disc, b_c_va, b_p_va, g_va)
        prov_probs_te = cpnl_router.fuse_proba(prov_probs_te_disc, b_c_te, b_p_te, g_te)

        fold_hier_va = np.zeros((len(va_idx), n_classes), dtype=np.float64)
        fold_hier_te = np.zeros((len(test_df), n_classes), dtype=np.float64)

        for p_name, p_idx in PROV_TO_IDX.items():
            fam_models = [CLASS_TO_IDX[m] for m, pv in PROVIDER_MAP.items() if pv == p_name]
            k_size = len(fam_models)
            if k_size == 1:
                m_idx = fam_models[0]
                fold_hier_va[:, m_idx] = prov_probs_va[:, p_idx]
                fold_hier_te[:, m_idx] = prov_probs_te[:, p_idx]
            else:
                fam_mask = np.isin(y_all[tr_idx], fam_models)
                fam_tr_indices = tr_idx[fam_mask]

                clf_tier2 = LogisticRegression(C=1.5, max_iter=1000, random_state=42 + fold)
                clf_tier2.fit(X_tr_svm[fam_mask], y_all[fam_tr_indices])

                sub_va_raw = clf_tier2.predict_proba(X_va_svm)
                sub_te_raw = clf_tier2.predict_proba(X_te_svm)
                sub_va_probs = calibrated_tier2_proba(sub_va_raw, cluster_size=k_size, beta=0.35)
                sub_te_probs = calibrated_tier2_proba(sub_te_raw, cluster_size=k_size, beta=0.35)

                for local_i, m_idx in enumerate(clf_tier2.classes_):
                    fold_hier_va[:, m_idx] = prov_probs_va[:, p_idx] * sub_va_probs[:, local_i]
                    fold_hier_te[:, m_idx] = prov_probs_te[:, p_idx] * sub_te_probs[:, local_i]

        oof_hier[va_idx] = fold_hier_va
        test_hier += fold_hier_te / len(folds)

        if has_qwen:
            f_log = (
                0.40 * np.log(np.clip(oof_svm[va_idx], 1e-12, 1.0))
                + 0.25 * np.log(np.clip(oof_qwen17[va_idx], 1e-12, 1.0))
                + 0.25 * np.log(np.clip(oof_qwen20[va_idx], 1e-12, 1.0))
                + 0.10 * np.log(np.clip(oof_hier[va_idx], 1e-12, 1.0))
            )
        else:
            f_log = (
                0.75 * np.log(np.clip(oof_svm[va_idx], 1e-12, 1.0))
                + 0.25 * np.log(np.clip(oof_hier[va_idx], 1e-12, 1.0))
            )
        fold_ba = balanced_accuracy_score(y_all[va_idx], np.argmax(f_log, axis=1))
        print(f"  Fold {fold + 1}/5 Balanced Accuracy: {fold_ba:.5f} (Elapsed: {time.time() - t_f:.1f}s)", flush=True)

    # 3. Geometric Ensembling & Threshold Optimization
    log_svm_oof = np.log(np.clip(oof_svm, 1e-12, 1.0))
    log_svm_te = np.log(np.clip(test_svm, 1e-12, 1.0))

    log_hier_oof = np.log(np.clip(oof_hier, 1e-12, 1.0))
    log_hier_te = np.log(np.clip(test_hier, 1e-12, 1.0))

    if has_qwen:
        log_q17_oof = np.log(np.clip(oof_qwen17, 1e-12, 1.0))
        log_q17_te = np.log(np.clip(test_qwen17, 1e-12, 1.0))

        log_q20_oof = np.log(np.clip(oof_qwen20, 1e-12, 1.0))
        log_q20_te = np.log(np.clip(test_qwen20, 1e-12, 1.0))

        oof_logits = (
            0.40 * log_svm_oof
            + 0.25 * log_q17_oof
            + 0.25 * log_q20_oof
            + 0.10 * log_hier_oof
        )
        test_logits = (
            0.40 * log_svm_te
            + 0.25 * log_q17_te
            + 0.25 * log_q20_te
            + 0.10 * log_hier_te
        )
    else:
        oof_logits = 0.75 * log_svm_oof + 0.25 * log_hier_oof
        test_logits = 0.75 * log_svm_te + 0.25 * log_hier_te

    raw_ba = balanced_accuracy_score(y_all, np.argmax(oof_logits, axis=1))
    print(f"\n>> Raw 5-Fold OOF Balanced Accuracy: {raw_ba:.5f}", flush=True)

    # Metric-Direct Prior Calibration
    best_offsets, cal_ba = optimize_class_offsets(softmax(oof_logits, axis=1), y_all, n_rounds=4)
    print(f">> Calibrated 5-Fold OOF Balanced Accuracy: {cal_ba:.5f} (Net Gain: {cal_ba - raw_ba:+.4f})", flush=True)

    oof_final_preds = np.argmax(oof_logits + best_offsets[None, :], axis=1)
    recalls = recall_score(y_all, oof_final_preds, average=None)
    print("\nPer-Model Recall Breakdown:", flush=True)
    rec_series = pd.Series(recalls, index=CLASSES).sort_values()
    for m_name, r_val in rec_series.items():
        print(f"  {m_name:24s}: {r_val:.4f}", flush=True)

    # 4. Generate Submission with 100% Identical Distribution
    test_final_idx = np.argmax(test_logits + best_offsets[None, :], axis=1)
    test_pred_labels = [IDX_TO_CLASS[i] for i in test_final_idx]
    sub = pd.DataFrame({"id": test_df["id"].astype(str), "model": test_pred_labels})
    sub_path = os.path.join(output_dir, "submission.parquet")
    sub.to_parquet(sub_path, index=False)
    print(f"\nWrote verified predictions to {sub_path}", flush=True)

    assert os.path.exists(sub_path)
    check_sub = pd.read_parquet(sub_path)
    assert list(check_sub.columns) == ["id", "model"]
    assert len(check_sub) == len(test_df)
    assert check_sub["model"].isin(CLASSES).all()
    print("Verification passed! Preview:", flush=True)
    print(check_sub.head(), flush=True)
    print(f"\n=== Total Execution Time: {time.time() - t_start:.1f}s ===", flush=True)
    return cal_ba
