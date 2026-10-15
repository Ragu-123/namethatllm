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
from sklearn.metrics import balanced_accuracy_score, recall_score

from .config import PROVIDER_MAP, CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, PROVIDERS, PROV_TO_IDX
from .features import extract_stylometric_matrix, AnalyticContrastiveASTD, extract_qwen_features
from .models import (
    NonLinearKernelFOSP,
    CPNLProviderLogOP,
    scale_free_laplacian_center,
    group_center_sparse,
    group_center_dense,
    calibrated_tier2_proba,
    normalize_hierarchical_marginals,
    optimize_class_offsets,
)

def run_opera_pipeline(data_dir, qwen_dir=None, output_dir=".", use_neural=True):
    print("=== OPERA Modular Pipeline Execution ===", flush=True)
    t_start = time.time()

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    print(f"Loaded data: Train={train_df.shape}, Test={test_df.shape}", flush=True)

    # 1. Feature Extraction
    print("\n--- Extracting Features ---", flush=True)
    X_style_tr_raw = extract_stylometric_matrix(train_df["text"].tolist())
    X_style_te_raw = extract_stylometric_matrix(test_df["text"].tolist())
    print(f"Stylometrics: train={X_style_tr_raw.shape}, test={X_style_te_raw.shape}", flush=True)

    astd_model = AnalyticContrastiveASTD()
    astd_model.fit(train_df["text"].tolist())
    X_astd_tr_raw = astd_model.transform(train_df["text"].tolist())
    X_astd_te_raw = astd_model.transform(test_df["text"].tolist())
    print(f"Analytic-ASTD: train={X_astd_tr_raw.shape}, test={X_astd_te_raw.shape}", flush=True)

    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(1, 5), min_df=2, max_features=180000,
        sublinear_tf=True, lowercase=False
    )
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 3), min_df=2, max_features=65000,
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
        Qwen_tr_raw = np.nan_to_num(np.hstack([qwen_h17_tr, qwen_h20_tr]), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        Qwen_te_raw = np.nan_to_num(np.hstack([qwen_h17_te, qwen_h20_te]), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        print(f"Qwen representations: train={Qwen_tr_raw.shape}, test={Qwen_te_raw.shape}", flush=True)
        has_qwen = True

    # 2. Validation Setup
    y_all = np.array([CLASS_TO_IDX[m] for m in train_df["model"]])
    prov_all = np.array([PROV_TO_IDX[PROVIDER_MAP[m]] for m in train_df["model"]])
    prompts_all = train_df["prompt_id"].values

    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(train_df, y_all, groups=prompts_all))

    oof_global_lr = np.zeros((len(train_df), len(CLASSES)), dtype=np.float64)
    oof_global_svc = np.zeros((len(train_df), len(CLASSES)), dtype=np.float64)
    oof_nl_fosp = np.zeros((len(train_df), len(CLASSES)), dtype=np.float64)
    oof_qwen_head = np.zeros((len(train_df), len(CLASSES)), dtype=np.float64)
    oof_hier = np.zeros((len(train_df), len(CLASSES)), dtype=np.float64)

    test_global_lr = np.zeros((len(test_df), len(CLASSES)), dtype=np.float64)
    test_global_svc = np.zeros((len(test_df), len(CLASSES)), dtype=np.float64)
    test_nl_fosp = np.zeros((len(test_df), len(CLASSES)), dtype=np.float64)
    test_qwen_head = np.zeros((len(test_df), len(CLASSES)), dtype=np.float64)
    test_hier = np.zeros((len(test_df), len(CLASSES)), dtype=np.float64)

    print("\n--- Commencing 5-Fold Grouped Cross-Validation ---", flush=True)

    for fold, (tr_idx, va_idx) in enumerate(folds):
        t_f = time.time()
        tr_prompts = prompts_all[tr_idx]
        va_prompts = prompts_all[va_idx]
        assert len(set(tr_prompts) & set(va_prompts)) == 0, f"Prompt leakage in fold {fold}!"

        scaler_s = StandardScaler()
        S_tr = scaler_s.fit_transform(X_style_tr_raw[tr_idx])
        S_va = scaler_s.transform(X_style_tr_raw[va_idx])
        S_te = scaler_s.transform(X_style_te_raw)

        scaler_a = StandardScaler()
        A_tr = scaler_a.fit_transform(X_astd_tr_raw[tr_idx])
        A_va = scaler_a.transform(X_astd_tr_raw[va_idx])
        A_te = scaler_a.transform(X_astd_te_raw)

        if has_qwen:
            scaler_q = StandardScaler()
            Q_tr = scaler_q.fit_transform(np.clip(Qwen_tr_raw[tr_idx], -1e4, 1e4))
            Q_va = scaler_q.transform(np.clip(Qwen_tr_raw[va_idx], -1e4, 1e4))
            Q_te = scaler_q.transform(np.clip(Qwen_te_raw, -1e4, 1e4))
            Dense_tr = np.hstack([S_tr, A_tr, Q_tr * 0.4])
            Dense_va = np.hstack([S_va, A_va, Q_va * 0.4])
            Dense_te = np.hstack([S_te, A_te, Q_te * 0.4])
        else:
            Dense_tr = np.hstack([S_tr, A_tr])
            Dense_va = np.hstack([S_va, A_va])
            Dense_te = np.hstack([S_te, A_te])

        D_tr_c = group_center_dense(Dense_tr, tr_prompts)
        D_va_c = group_center_dense(Dense_va, va_prompts)
        D_te_c = scale_free_laplacian_center(Dense_te, test_df["text"].tolist(), train_texts=train_df["text"].tolist())

        T_tr_c = group_center_sparse(X_tfidf_tr[tr_idx], tr_prompts)
        T_va_c = group_center_sparse(X_tfidf_tr[va_idx], va_prompts)
        T_te_c = scale_free_laplacian_center(X_tfidf_te, test_df["text"].tolist(), train_texts=train_df["text"].tolist())

        X_tr_comb = sparse.hstack([T_tr_c, sparse.csr_matrix(D_tr_c * 0.25)], format="csr")
        X_va_comb = sparse.hstack([T_va_c, sparse.csr_matrix(D_va_c * 0.25)], format="csr")
        X_te_comb = sparse.hstack([T_te_c, sparse.csr_matrix(D_te_c * 0.25)], format="csr")

        X_tr_norm = normalize(X_tr_comb, norm="l2")
        X_va_norm = normalize(X_va_comb, norm="l2")
        X_te_norm = normalize(X_te_comb, norm="l2")

        # Head 1: Global Logistic Regression
        clf_lr = LogisticRegression(C=2.5, max_iter=1000, random_state=42 + fold)
        clf_lr.fit(X_tr_norm, y_all[tr_idx])
        oof_global_lr[va_idx] = clf_lr.predict_proba(X_va_norm)
        test_global_lr += clf_lr.predict_proba(X_te_norm) / len(folds)

        # Head 2: Global LinearSVC
        base_svc = LinearSVC(C=0.45, dual=False, max_iter=3000, random_state=42 + fold)
        base_svc.fit(X_tr_norm, y_all[tr_idx])
        svc_va_dec = base_svc.decision_function(X_va_norm)
        svc_te_dec = base_svc.decision_function(X_te_norm)
        oof_global_svc[va_idx] = softmax(svc_va_dec / 1.5, axis=1)
        test_global_svc += softmax(svc_te_dec / 1.5, axis=1) / len(folds)

        # Head 3: Non-Linear Kernel-FOSP in RKHS
        nl_fosp = NonLinearKernelFOSP(n_components=36 if has_qwen else 32, lam_p_ratio=6.67, gamma_rel=0.06, rbf_gamma=2.0)
        Z_tr = nl_fosp.fit_transform(X_tr_comb, tr_prompts, y_all[tr_idx])
        Z_va = nl_fosp.transform(X_va_comb)
        Z_te = nl_fosp.transform(X_te_comb)

        z_scaler = StandardScaler()
        Z_tr = z_scaler.fit_transform(Z_tr)
        Z_va = z_scaler.transform(Z_va)
        Z_te = z_scaler.transform(Z_te)

        clf_fosp = LogisticRegression(C=0.8, max_iter=1000, random_state=42 + fold)
        clf_fosp.fit(Z_tr, y_all[tr_idx])
        oof_nl_fosp[va_idx] = clf_fosp.predict_proba(Z_va)
        test_nl_fosp += clf_fosp.predict_proba(Z_te) / len(folds)

        # Head 4: Specialized Neural Qwen Head
        if has_qwen:
            Q_tr_c = group_center_dense(Q_tr, tr_prompts)
            Q_va_c = group_center_dense(Q_va, va_prompts)
            Q_te_c = scale_free_laplacian_center(Q_te, test_df["text"].tolist(), train_texts=train_df["text"].tolist())

            clf_qwen = LogisticRegression(C=0.015, max_iter=1000, random_state=42 + fold)
            clf_qwen.fit(Q_tr_c, y_all[tr_idx])
            oof_qwen_head[va_idx] = clf_qwen.predict_proba(Q_va_c)
            test_qwen_head += clf_qwen.predict_proba(Q_te_c) / len(folds)

        # Head 5: CPNL LogOP Tier-1 Provider Router
        clf_tier1 = LogisticRegression(C=2.0, max_iter=1000, random_state=42 + fold)
        clf_tier1.fit(X_tr_norm, prov_all[tr_idx])
        prov_probs_va_disc = clf_tier1.predict_proba(X_va_norm)
        prov_probs_te_disc = clf_tier1.predict_proba(X_te_norm)

        cpnl_router = CPNLProviderLogOP(n_providers=len(PROVIDERS))
        cpnl_router.fit_markov(train_df["text"].iloc[tr_idx].tolist(), prov_all[tr_idx])
        b_c_va, b_p_va, g_va = cpnl_router.compute_bayes_factors(train_df["text"].iloc[va_idx].tolist())
        b_c_te, b_p_te, g_te = cpnl_router.compute_bayes_factors(test_df["text"].tolist())

        prov_probs_va = cpnl_router.fuse_proba(prov_probs_va_disc, b_c_va, b_p_va, g_va)
        prov_probs_te = cpnl_router.fuse_proba(prov_probs_te_disc, b_c_te, b_p_te, g_te)

        # Head 6: Tier-2 Intra-Family Resolvers
        fold_hier_va = np.zeros((len(va_idx), len(CLASSES)), dtype=np.float64)
        fold_hier_te = np.zeros((len(test_df), len(CLASSES)), dtype=np.float64)

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
                fam_tr_prompts = prompts_all[fam_tr_indices]

                T_fam_tr = group_center_sparse(X_tfidf_tr[fam_tr_indices], fam_tr_prompts)
                D_fam_tr = group_center_dense(Dense_tr[fam_mask], fam_tr_prompts)
                X_fam_tr = normalize(
                    sparse.hstack([T_fam_tr, sparse.csr_matrix(D_fam_tr * 0.3)], format="csr"), norm="l2"
                )

                clf_tier2 = LogisticRegression(C=1.8, max_iter=1000, random_state=42 + fold)
                clf_tier2.fit(X_fam_tr, y_all[fam_tr_indices])

                sub_va_raw = clf_tier2.predict_proba(X_va_norm)
                sub_te_raw = clf_tier2.predict_proba(X_te_norm)
                sub_va_probs = calibrated_tier2_proba(sub_va_raw, cluster_size=k_size, beta=0.35)
                sub_te_probs = calibrated_tier2_proba(sub_te_raw, cluster_size=k_size, beta=0.35)

                for local_i, m_idx in enumerate(clf_tier2.classes_):
                    fold_hier_va[:, m_idx] = prov_probs_va[:, p_idx] * sub_va_probs[:, local_i]
                    fold_hier_te[:, m_idx] = prov_probs_te[:, p_idx] * sub_te_probs[:, local_i]

        oof_hier[va_idx] = fold_hier_va
        test_hier += fold_hier_te / len(folds)

        if has_qwen:
            fold_blend = (
                0.25 * oof_global_lr[va_idx]
                + 0.15 * oof_global_svc[va_idx]
                + 0.20 * oof_nl_fosp[va_idx]
                + 0.20 * oof_qwen_head[va_idx]
                + 0.20 * oof_hier[va_idx]
            )
        else:
            fold_blend = (
                0.30 * oof_global_lr[va_idx]
                + 0.20 * oof_global_svc[va_idx]
                + 0.25 * oof_nl_fosp[va_idx]
                + 0.25 * oof_hier[va_idx]
            )
        fold_ba = balanced_accuracy_score(y_all[va_idx], np.argmax(fold_blend, axis=1))
        print(f"  Fold {fold + 1}/5 Balanced Accuracy: {fold_ba:.5f} (Elapsed: {time.time() - t_f:.1f}s)", flush=True)

    # 3. Marginal Prior Normalization & Calibration
    oof_hier_norm, test_hier_norm = normalize_hierarchical_marginals(oof_hier, test_hier, tau=0.65)

    if has_qwen:
        oof_ensemble = (
            0.25 * oof_global_lr
            + 0.15 * oof_global_svc
            + 0.20 * oof_nl_fosp
            + 0.20 * oof_qwen_head
            + 0.20 * oof_hier_norm
        )
        test_ensemble = (
            0.25 * test_global_lr
            + 0.15 * test_global_svc
            + 0.20 * test_nl_fosp
            + 0.20 * test_qwen_head
            + 0.20 * test_hier_norm
        )
    else:
        oof_ensemble = (
            0.30 * oof_global_lr
            + 0.20 * oof_global_svc
            + 0.25 * oof_nl_fosp
            + 0.25 * oof_hier_norm
        )
        test_ensemble = (
            0.30 * test_global_lr
            + 0.20 * test_global_svc
            + 0.25 * test_nl_fosp
            + 0.25 * test_hier_norm
        )

    raw_ba = balanced_accuracy_score(y_all, np.argmax(oof_ensemble, axis=1))
    print(f"\n>> Raw 5-Fold OOF Balanced Accuracy: {raw_ba:.5f}", flush=True)

    best_offsets, cal_ba = optimize_class_offsets(oof_ensemble, y_all, n_rounds=4)
    print(f">> Calibrated 5-Fold OOF Balanced Accuracy: {cal_ba:.5f} (Net Gain: {cal_ba - raw_ba:+.4f})", flush=True)

    oof_final_preds = np.argmax(np.log(np.clip(oof_ensemble, 1e-7, 1.0)) + best_offsets[None, :], axis=1)
    recalls = recall_score(y_all, oof_final_preds, average=None)
    print("\nPer-Model Recall Breakdown:", flush=True)
    rec_series = pd.Series(recalls, index=CLASSES).sort_values()
    for m_name, r_val in rec_series.items():
        print(f"  {m_name:24s}: {r_val:.4f}", flush=True)

    # 4. Generate Submission
    test_final_idx = np.argmax(np.log(np.clip(test_ensemble, 1e-7, 1.0)) + best_offsets[None, :], axis=1)
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
