import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import eigh
from sklearn.preprocessing import normalize

class NonLinearKernelFOSP:
    """Fisher-Orthogonal Style Projection in Reproducing Kernel Hilbert Space (RKHS)."""
    def __init__(self, n_components=32, lam_p_ratio=6.67, gamma_rel=0.06, rbf_gamma=2.0):
        self.n_components = n_components
        self.lam_p_ratio = lam_p_ratio
        self.gamma_rel = gamma_rel
        self.rbf_gamma = rbf_gamma
        self.X_tr_norm = None
        self.Alpha_tilde = None
        self.b_kern = None

    def _compute_kernel_from_cosine(self, C):
        C2 = np.square(C)
        C3 = C2 * C
        K_rbf = np.exp(-self.rbf_gamma * (1.0 - C))
        return (0.45 * C + 0.25 * C2 + 0.10 * C3 + 0.20 * K_rbf).astype(np.float64)

    def fit_transform(self, X_tr, prompts, models):
        self.X_tr_norm = normalize(X_tr, norm="l2", copy=True)
        n_samples = self.X_tr_norm.shape[0]

        C_tr = (self.X_tr_norm @ self.X_tr_norm.T)
        if sparse.issparse(C_tr):
            C_tr = C_tr.toarray()
        K_tr = self._compute_kernel_from_cosine(C_tr)

        row_means = K_tr.mean(axis=1, keepdims=True)
        col_means = K_tr.mean(axis=0, keepdims=True)
        grand_mean = K_tr.mean()
        K_c = K_tr - row_means - col_means + grand_mean

        p_codes, p_uniques = pd.factorize(prompts)
        m_codes, m_uniques = pd.factorize(models)
        n_p = len(p_uniques)
        n_m = len(m_uniques)

        G_P = np.zeros((n_samples, n_p), dtype=np.float64)
        G_P[np.arange(n_samples), p_codes] = 1.0
        H_P = (G_P / np.maximum(G_P.sum(axis=0, keepdims=True), 1.0)) @ G_P.T

        G_M = np.zeros((n_samples, n_m), dtype=np.float64)
        G_M[np.arange(n_samples), m_codes] = 1.0
        H_M = (G_M / np.maximum(G_M.sum(axis=0, keepdims=True), 1.0)) @ G_M.T

        H_E = np.eye(n_samples, dtype=np.float64) - H_P - H_M

        A_dual = (K_c @ H_M @ K_c) / n_m
        B_P = (K_c @ H_P @ K_c) / n_p
        B_E = (K_c @ (H_E.T @ H_E) @ K_c) / n_samples

        B_P_norm = B_P / max(np.trace(B_P), 1e-12)
        B_E_norm = B_E / max(np.trace(B_E), 1e-12)
        B_comb = self.lam_p_ratio * B_P_norm + 1.0 * B_E_norm

        gamma = self.gamma_rel * (np.trace(B_comb) / max(np.trace(K_c), 1e-12))
        B_dual = B_comb + gamma * K_c

        # Robust Eigendecomposition of B_dual (bypasses Cholesky LinAlgError)
        A_dual = 0.5 * (A_dual + A_dual.T)
        B_dual = 0.5 * (B_dual + B_dual.T)

        evals_b, evecs_b = eigh(B_dual)
        max_b = max(float(np.max(evals_b)), 1e-12)
        min_b_thresh = max(max_b * 1e-3, 1e-6)
        evals_b_clipped = np.maximum(evals_b, min_b_thresh)

        inv_sqrt_b = evecs_b @ np.diag(1.0 / np.sqrt(evals_b_clipped)) @ evecs_b.T
        M = inv_sqrt_b @ A_dual @ inv_sqrt_b
        M = 0.5 * (M + M.T)

        evals_m, evecs_m = eigh(M)
        top_idx = np.argsort(evals_m)[::-1][:self.n_components]
        Alpha = inv_sqrt_b @ evecs_m[:, top_idx]

        rkhs_norms = np.sqrt(np.maximum(np.sum(Alpha * (K_c @ Alpha), axis=0), 1e-12))
        Alpha = Alpha / rkhs_norms[None, :]

        self.Alpha_tilde = (Alpha - Alpha.mean(axis=0, keepdims=True)).astype(np.float32)
        self.b_kern = (self.Alpha_tilde.T @ K_tr.mean(axis=1)).astype(np.float32)

        Z_tr = (K_tr @ self.Alpha_tilde) - self.b_kern[None, :]
        return Z_tr.astype(np.float32)

    def transform(self, X_te):
        X_te_norm = normalize(X_te, norm="l2", copy=True)
        C_te = (X_te_norm @ self.X_tr_norm.T)
        if sparse.issparse(C_te):
            C_te = C_te.toarray()
        K_te_tr = self._compute_kernel_from_cosine(C_te).astype(np.float32)
        Z_te = (K_te_tr @ self.Alpha_tilde) - self.b_kern[None, :]
        return Z_te.astype(np.float32)
