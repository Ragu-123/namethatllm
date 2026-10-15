import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

def scale_free_laplacian_center(X_test, texts_test, train_texts=None, k_max=17, sim_tau=0.18, beta=4.0, kappa=2.5, chunk_size=1024):
    """Scale-free Laplacian topic centering using Kish's Neff shrinkage."""
    n_test = len(texts_test)
    if n_test <= 1:
        return X_test

    topic_vec = TfidfVectorizer(
        ngram_range=(1, 2), max_features=20000, lowercase=True, stop_words="english", sublinear_tf=True
    )
    if train_texts is not None and len(train_texts) > 0:
        topic_vec.fit(list(train_texts) + list(texts_test))
    else:
        topic_vec.fit(texts_test)

    U_test = normalize(topic_vec.transform(texts_test), norm="l2", copy=False)
    k_eff = min(k_max, n_test - 1)

    nbr_indices = np.zeros((n_test, k_eff), dtype=np.int32)
    nbr_sims = np.zeros((n_test, k_eff), dtype=np.float32)

    for start in range(0, n_test, chunk_size):
        end = min(start + chunk_size, n_test)
        sim_block = (U_test[start:end] @ U_test.T).toarray()
        for local_i in range(end - start):
            sim_block[local_i, start + local_i] = -1.0

        if k_eff < n_test - 1:
            part_idx = np.argpartition(sim_block, -k_eff, axis=1)[:, -k_eff:]
        else:
            part_idx = np.argsort(sim_block, axis=1)[:, -k_eff:]

        part_sims = np.take_along_axis(sim_block, part_idx, axis=1)
        order = np.argsort(part_sims, axis=1)[:, ::-1]
        nbr_indices[start:end] = np.take_along_axis(part_idx, order, axis=1)
        nbr_sims[start:end] = np.take_along_axis(part_sims, order, axis=1)

    excess = np.maximum(0.0, (nbr_sims - sim_tau) / (1.0 - sim_tau))
    raw_w = np.square(excess) * np.exp(beta * np.maximum(nbr_sims, 0.0))

    w_sum = raw_w.sum(axis=1)
    w_sq_sum = np.square(raw_w).sum(axis=1)

    n_eff = np.where(w_sq_sum > 1e-12, np.square(w_sum) / np.maximum(w_sq_sum, 1e-12), 0.0)
    strength_gate = np.tanh(w_sum / 0.35)
    omega = (17.0 / 18.0) * (n_eff / (n_eff + kappa)) * strength_gate

    norm_w = np.where(w_sum[:, None] > 1e-12, raw_w / np.maximum(w_sum[:, None], 1e-12), 0.0)
    rho = np.sqrt((17.0 / 18.0) / (1.0 + np.where(n_eff > 1e-6, np.square(omega) / np.maximum(n_eff, 1.0), 0.0)))

    if sparse.issparse(X_test):
        X_csr = X_test.tocsr()
        row_list, col_list, val_list = [], [], []
        for i in range(n_test):
            row_list.append(i); col_list.append(i); val_list.append(float(rho[i]))
            if omega[i] > 1e-6:
                for m_j in range(k_eff):
                    w_ij = norm_w[i, m_j]
                    if w_ij > 1e-6:
                        row_list.append(i); col_list.append(int(nbr_indices[i, m_j]))
                        val_list.append(float(-rho[i] * omega[i] * w_ij))
        L_op = sparse.csr_matrix((val_list, (row_list, col_list)), shape=(n_test, n_test), dtype=np.float32)
        return L_op @ X_csr

    X_out = np.empty_like(X_test, dtype=np.float32)
    for i in range(n_test):
        if omega[i] > 1e-6:
            peer_mean = (norm_w[i, :, None] * X_test[nbr_indices[i]]).sum(axis=0)
            X_out[i] = rho[i] * (X_test[i] - omega[i] * peer_mean)
        else:
            X_out[i] = rho[i] * X_test[i]
    return X_out

def group_center_sparse(X, groups):
    """Subtract prompt group mean for sparse matrices."""
    df_g = pd.DataFrame({"g": groups})
    codes, uniques = pd.factorize(df_g["g"])
    n_samples = X.shape[0]
    n_groups = len(uniques)
    counts = np.bincount(codes)
    G = sparse.csr_matrix(
        (np.ones(n_samples, dtype=np.float32), (codes, np.arange(n_samples))),
        shape=(n_groups, n_samples)
    )
    inv_counts = sparse.diags(1.0 / np.maximum(counts, 1).astype(np.float32))
    group_means = inv_counts @ (G @ X)
    sample_means = G.T @ group_means
    return (X - sample_means).tocsr()

def group_center_dense(X, groups):
    """Subtract prompt group mean for dense matrices."""
    X_out = X.copy()
    df_g = pd.Series(groups)
    for _, idx in df_g.groupby(df_g).groups.items():
        indices = np.asarray(list(idx))
        if len(indices) > 1:
            X_out[indices] -= X_out[indices].mean(axis=0, keepdims=True)
    return X_out
