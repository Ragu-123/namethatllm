import os
import re
import sys
import time
import socket
import numpy as np
import pandas as pd
from collections import Counter
from scipy import sparse
from scipy.linalg import eigh
from scipy.special import softmax
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score, recall_score

print("=== OPERA v3.0: RKHS Non-Linear Kernel-FOSP + CPNL LogOP + Analytic-ASTD ===", flush=True)
t_start = time.time()

# 1. Environment & Hardware Discovery
print("\n--- System & Hardware Discovery ---", flush=True)
print(f"Python: {sys.version.split()[0]}", flush=True)
if os.path.exists("/kaggle/input"):
    print(f"Available in /kaggle/input: {os.listdir('/kaggle/input')}", flush=True)

# 2. Locate Data Directory
data_candidates = [
    '/kaggle/input/competitions/name-that-llm',
    '/kaggle/input/name-that-llm',
]
DATA_DIR = next((d for d in data_candidates if os.path.exists(os.path.join(d, 'train.csv'))), None)
if DATA_DIR is None:
    raise FileNotFoundError("Could not locate train.csv in candidate paths!")

print(f"Loading competition data from: {DATA_DIR}", flush=True)
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
print(f"Train: {train_df.shape}, Test: {test_df.shape}", flush=True)

# 3. Provider Taxonomy & Class Mapping
PROVIDER_MAP = {
    "claude-fable-5": "Anthropic",
    "claude-opus-4-6": "Anthropic",
    "claude-opus-4-8": "Anthropic",
    "claude-opus-5": "Anthropic",
    "claude-sonnet-5": "Anthropic",
    "gemini-3.1-pro-preview": "Google",
    "gemini-3.6-flash": "Google",
    "gemini-3.7-flash": "Google",
    "gemini-3.8-flash": "Google",
    "gpt-5.5": "OpenAI",
    "gpt-5.6-luna": "OpenAI",
    "gpt-5.6-sol": "OpenAI",
    "gpt-5.6-terra": "OpenAI",
    "gpt-6-astra": "OpenAI",
    "glm-5.3": "ZAI",
    "glm-5.3-flash": "ZAI",
    "kimi-k3": "Moonshot",
    "qwen3.8-max": "Alibaba",
}

CLASSES = sorted(train_df["model"].unique())
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

PROVIDERS = sorted(list(set(PROVIDER_MAP.values())))
PROV_TO_IDX = {p: i for i, p in enumerate(PROVIDERS)}

PUNCT_SKELETON_SET = set("—–-;:,!?()[]“”‘’\"'….\n")
TOKEN_RE = re.compile(r"(?u)\b\w+\b|[^\w\s]")
SYLLABLE_RE = re.compile(r"[aeiouyAEIOUY]+")
MAJOR_PUNCT = {"—", "–", ";", ":", ",", "(", ")"}

# 4. Stylometric Feature Extractor (54 Fine-Grained Dimensions)
def extract_stylometric_matrix(texts):
    FUNCTION_WORDS = [
        "the", "and", "of", "to", "in", "is", "that", "it", "for", "as",
        "with", "on", "be", "this", "by", "are", "from", "or", "an", "at",
        "not", "but", "have", "which", "they", "you", "one", "all", "their",
        "can", "her", "has", "there", "been", "if", "more", "when", "will"
    ]
    feats = []
    for text in texts:
        t = str(text)
        n_chars = max(len(t), 1)
        words = t.split()
        n_words = max(len(words), 1)
        scale_100w = 100.0 / n_words

        sents = [s.strip() for s in re.split(r'[.!?]+', t) if s.strip()]
        n_sents = max(len(sents), 1)
        sent_lens = [len(s.split()) for s in sents] if sents else [1]
        word_lens = [len(w.strip(".,!?;:\"'()[]—-")) for w in words] if words else [1]

        em_dash = t.count("—") * scale_100w
        en_dash = t.count("–") * scale_100w
        hyphen = t.count("-") * scale_100w
        semicolon = t.count(";") * scale_100w
        colon = t.count(":") * scale_100w
        comma = t.count(",") * scale_100w
        period = t.count(".") * scale_100w
        exclam = t.count("!") * scale_100w
        quest = t.count("?") * scale_100w
        ellipsis_uni = t.count("…") * scale_100w
        ellipsis_dot = t.count("...") * scale_100w
        lparen = t.count("(") * scale_100w
        rparen = t.count(")") * scale_100w
        curly_squote = (t.count("‘") + t.count("’")) * scale_100w
        curly_dquote = (t.count("“") + t.count("”")) * scale_100w
        straight_squote = t.count("'") * scale_100w
        straight_dquote = t.count('"') * scale_100w
        slash = t.count("/") * scale_100w
        asterisk = t.count("*") * scale_100w
        newline = t.count("\n") * scale_100w
        double_nl = t.count("\n\n") * scale_100w
        em_dash_spaced = t.count(" — ") * scale_100w
        em_dash_tight = (t.count("—") - t.count(" — ")) * scale_100w

        upper_ratio = sum(1 for c in t if c.isupper()) / n_chars
        digit_ratio = sum(1 for c in t if c.isdigit()) / n_chars
        space_ratio = sum(1 for c in t if c.isspace()) / n_chars
        punct_ratio = sum(1 for c in t if c in ".,!?;:\"'()[]—–-…") / n_chars

        mean_wlen = float(np.mean(word_lens))
        std_wlen = float(np.std(word_lens))
        max_wlen = float(np.max(word_lens))
        long_words_6 = sum(1 for l in word_lens if l >= 6) / n_words
        long_words_10 = sum(1 for l in word_lens if l >= 10) / n_words
        short_words_3 = sum(1 for l in word_lens if l <= 3) / n_words

        lower_words = [w.lower().strip(".,!?;:\"'()[]—-") for w in words]
        unique_words = len(set(lower_words))
        ttr = unique_words / n_words
        herdan_c = np.log(max(unique_words, 2)) / np.log(max(n_words, 2))
        hapax = sum(1 for w in set(lower_words) if lower_words.count(w) == 1) / n_words

        mean_slen = float(np.mean(sent_lens))
        std_slen = float(np.std(sent_lens))
        max_slen = float(np.max(sent_lens))
        min_slen = float(np.min(sent_lens))
        cv_slen = std_slen / (mean_slen + 1e-5)
        first_sent_len = sent_lens[0] / (mean_slen + 1e-5)
        last_sent_len = sent_lens[-1] / (mean_slen + 1e-5)

        title_words = sum(1 for w in words[1:] if w and w[0].isupper()) / n_words
        all_caps_words = sum(1 for w in words if len(w) > 1 and w.isupper()) / n_words

        first_word = lower_words[0] if lower_words else ""
        starts_with_the = 1.0 if first_word == "the" else 0.0
        starts_with_in = 1.0 if first_word in ("in", "inside", "within") else 0.0
        starts_with_imagine = 1.0 if first_word in ("imagine", "picture", "consider", "suppose") else 0.0
        starts_with_to = 1.0 if first_word == "to" else 0.0
        starts_with_a = 1.0 if first_word in ("a", "an") else 0.0

        fw_counts = [lower_words.count(fw) * scale_100w for fw in FUNCTION_WORDS[:30]]

        row = [
            np.log1p(n_chars), np.log1p(n_words), np.log1p(n_sents),
            em_dash, en_dash, hyphen, semicolon, colon, comma, period,
            exclam, quest, ellipsis_uni, ellipsis_dot, lparen, rparen,
            curly_squote, curly_dquote, straight_squote, straight_dquote,
            slash, asterisk, newline, double_nl, em_dash_spaced, em_dash_tight,
            upper_ratio, digit_ratio, space_ratio, punct_ratio,
            mean_wlen, std_wlen, max_wlen, long_words_6, long_words_10, short_words_3,
            ttr, herdan_c, hapax,
            mean_slen, std_slen, max_slen, min_slen, cv_slen, first_sent_len, last_sent_len,
            title_words, all_caps_words,
            starts_with_the, starts_with_in, starts_with_imagine, starts_with_to, starts_with_a
        ] + fw_counts
        feats.append(row)
    return np.asarray(feats, dtype=np.float32)

# 5. Pillar 3: Analytic Zero-GPU Contrastive ASTD (32 Features)
class AnalyticContrastiveASTD:
    def __init__(self):
        self.uni_base = Counter()
        self.bi_base = Counter()
        self.total_base = 1
        self.uni_surf = Counter()
        self.bi_surf = Counter()
        self.tri_surf = Counter()
        self.total_surf = 1

    def _estimate_subword_fertility(self, word):
        n_c = len(word)
        if n_c <= 3:
            return 1.0
        n_vowel_groups = len(SYLLABLE_RE.findall(word))
        has_camel_or_hyphen = 1.0 if (any(c.isupper() for c in word[1:]) or "-" in word) else 0.0
        return max(1.0, 0.45 * n_vowel_groups + 0.12 * n_c + 0.5 * has_camel_or_hyphen)

    def fit(self, texts):
        self.uni_base.clear()
        self.bi_base.clear()
        self.uni_surf.clear()
        self.bi_surf.clear()
        self.tri_surf.clear()

        for text in texts:
            surf_toks = ["<BOS>", "<BOS>"] + TOKEN_RE.findall(str(text)) + ["<EOS>"]
            base_toks = ["<bos>"] + [t.lower() for t in surf_toks[2:-1] if t.isalnum()] + ["<eos>"]

            for i in range(len(base_toks)):
                self.uni_base[base_toks[i]] += 1
                if i >= 1:
                    self.bi_base[(base_toks[i-1], base_toks[i])] += 1

            for i in range(len(surf_toks)):
                self.uni_surf[surf_toks[i]] += 1
                if i >= 1:
                    self.bi_surf[(surf_toks[i-1], surf_toks[i])] += 1
                if i >= 2:
                    self.tri_surf[(surf_toks[i-2], surf_toks[i-1], surf_toks[i])] += 1

        self.total_base = max(sum(self.uni_base.values()), 1)
        self.total_surf = max(sum(self.uni_surf.values()), 1)
        return self

    def transform(self, texts):
        v_base = max(len(self.uni_base), 1000)
        v_surf = max(len(self.uni_surf), 1000)
        feats = np.zeros((len(texts), 32), dtype=np.float32)

        for idx, text in enumerate(texts):
            raw_tokens = TOKEN_RE.findall(str(text))
            if len(raw_tokens) < 4:
                continue

            surf_toks = ["<BOS>", "<BOS>"] + raw_tokens + ["<EOS>"]
            n_tok = len(raw_tokens)

            s_base = np.zeros(n_tok, dtype=np.float32)
            s_surf = np.zeros(n_tok, dtype=np.float32)
            fertility = np.zeros(n_tok, dtype=np.float32)
            punct_indices = []

            prev_alnum = "<bos>"
            for t_i, tok in enumerate(raw_tokens):
                w_low = tok.lower()
                p_u_base = (self.uni_base[w_low] + 1.0) / (self.total_base + v_base)
                c_prev_b = self.uni_base[prev_alnum]
                p_b_base = (self.bi_base[(prev_alnum, w_low)] + 0.5) / (c_prev_b + 0.5 * v_base)
                p_interp_base = 0.70 * p_b_base + 0.30 * p_u_base
                s_base[t_i] = -np.log(max(p_interp_base, 1e-9))
                if tok.isalnum():
                    prev_alnum = w_low

                w_m2 = surf_toks[t_i]
                w_m1 = surf_toks[t_i + 1]
                p_u_surf = (self.uni_surf[tok] + 1.0) / (self.total_surf + v_surf)
                p_b_surf = (self.bi_surf[(w_m1, tok)] + 0.4) / (self.uni_surf[w_m1] + 0.4 * v_surf)
                c_bi = self.bi_surf[(w_m2, w_m1)]
                p_t_surf = (self.tri_surf[(w_m2, w_m1, tok)] + 0.2) / (c_bi + 0.2 * v_surf)
                p_interp_surf = 0.55 * p_t_surf + 0.30 * p_b_surf + 0.15 * p_u_surf
                s_surf[t_i] = -np.log(max(p_interp_surf, 1e-9))

                fertility[t_i] = self._estimate_subword_fertility(tok)
                if tok in MAJOR_PUNCT:
                    punct_indices.append(t_i)

            ds = s_base - s_surf
            q_cut = max(n_tok // 5, 1)
            ds_diff1 = np.abs(np.diff(ds))
            ds_diff2 = np.abs(np.diff(ds, n=2)) if n_tok > 4 else np.array([0.0], dtype=np.float32)
            sb_diff1 = np.abs(np.diff(s_base))
            fert_diff1 = np.abs(np.diff(fertility))

            if punct_indices:
                p_arr = np.array(punct_indices)
                s_at_punct = s_surf[p_arr].mean()
                ds_at_punct = ds[p_arr].mean()
                pre_idx = np.maximum(p_arr - 1, 0)
                post_idx = np.minimum(p_arr + 1, n_tok - 1)
                pre_post_drop = (s_surf[pre_idx] - s_surf[post_idx]).mean()
            else:
                s_at_punct = s_surf.mean()
                ds_at_punct = ds.mean()
                pre_post_drop = 0.0

            feats[idx] = [
                s_base.mean(), s_base.std(), np.percentile(s_base, 10), np.percentile(s_base, 50), np.percentile(s_base, 90),
                s_surf.mean(), s_surf.std(), np.percentile(s_surf, 10), np.percentile(s_surf, 50), np.percentile(s_surf, 90),
                ds.mean(), ds.std(), np.percentile(ds, 10), np.percentile(ds, 25),
                np.percentile(ds, 50), np.percentile(ds, 75), np.percentile(ds, 90),
                ds[:q_cut].mean(), ds[q_cut:-q_cut].mean() if n_tok > 2 * q_cut else ds.mean(), ds[-q_cut:].mean(),
                ds_diff1.mean(), ds_diff1.std(), ds_diff2.mean(),
                sb_diff1.mean(), (ds > 0.0).mean(), (ds > 0.5).mean(),
                fertility.mean(), fertility.std(), np.percentile(fertility, 90), fert_diff1.mean(),
                s_at_punct, ds_at_punct + pre_post_drop
            ]
        return np.nan_to_num(feats, 0.0)

# 6. Pillar 1: Non-Linear Kernel-FOSP via Exact RKHS Double-Centering
class NonLinearKernelFOSP:
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

# 7. Pillar 2: CPNL Logarithmic Opinion Pool (LogOP) for Tier-1 Provider Routing
class CPNLProviderLogOP:
    def __init__(self, n_providers=6, alpha_char=0.5, alpha_punct=0.25):
        self.n_providers = n_providers
        self.alpha_char = alpha_char
        self.alpha_punct = alpha_punct
        self.char_log_probs = {}
        self.char_default = {}
        self.punct_log_probs = {}
        self.punct_default = {}
        self.char_std = 1.0
        self.punct_std = 1.0
        self.eta_char = 0.25
        self.eta_punct = 0.35

    def _extract_punct_seq(self, text):
        seq = ["<BOS>"]
        for ch in str(text):
            if ch in PUNCT_SKELETON_SET:
                seq.append(ch)
        seq.append("<EOS>")
        return seq

    def fit_markov(self, texts, provider_labels):
        for g in range(self.n_providers):
            g_texts = [t for t, y in zip(texts, provider_labels) if y == g]
            c_ctx_counts = Counter()
            c_ngram_counts = Counter()
            p_ctx_counts = Counter()
            p_ngram_counts = Counter()

            for t in g_texts:
                padded = "^^^" + str(t) + "$"
                for i in range(len(padded) - 3):
                    ctx = padded[i:i+3]
                    ng = padded[i:i+4]
                    c_ctx_counts[ctx] += 1
                    c_ngram_counts[ng] += 1

                p_seq = self._extract_punct_seq(t)
                for i in range(len(p_seq) - 1):
                    ctx = p_seq[i]
                    bg = (p_seq[i], p_seq[i+1])
                    p_ctx_counts[ctx] += 1
                    p_ngram_counts[bg] += 1

            v_char = max(len(set(k[-1] for k in c_ngram_counts.keys())), 64)
            self.char_log_probs[g] = {
                ng: np.log((cnt + self.alpha_char) / (c_ctx_counts[ng[:3]] + self.alpha_char * v_char))
                for ng, cnt in c_ngram_counts.items()
            }
            self.char_default[g] = np.log(1.0 / v_char)

            v_punct = len(PUNCT_SKELETON_SET) + 2
            self.punct_log_probs[g] = {
                bg: np.log((cnt + self.alpha_punct) / (p_ctx_counts[bg[0]] + self.alpha_punct * v_punct))
                for bg, cnt in p_ngram_counts.items()
            }
            self.punct_default[g] = np.log(1.0 / v_punct)
        return self

    def compute_bayes_factors(self, texts):
        n = len(texts)
        raw_char = np.zeros((n, self.n_providers), dtype=np.float64)
        raw_punct = np.zeros((n, self.n_providers), dtype=np.float64)
        gates = np.zeros(n, dtype=np.float64)

        for i, t in enumerate(texts):
            padded = "^^^" + str(t) + "$"
            n_c = max(len(padded) - 3, 1)
            p_seq = self._extract_punct_seq(t)
            n_p = max(len(p_seq) - 1, 1)
            gates[i] = np.tanh(max(len(p_seq) - 2, 0) / 8.0)

            for g in range(self.n_providers):
                c_table = self.char_log_probs[g]
                c_def = self.char_default[g]
                c_sum = 0.0
                for j in range(n_c):
                    ng = padded[j:j+4]
                    c_sum += c_table.get(ng, c_def)
                raw_char[i, g] = c_sum / n_c

                p_table = self.punct_log_probs[g]
                p_def = self.punct_default[g]
                p_sum = 0.0
                for j in range(n_p):
                    bg = (p_seq[j], p_seq[j+1])
                    p_sum += p_table.get(bg, p_def)
                raw_punct[i, g] = p_sum / n_p

        b_char = raw_char - raw_char.mean(axis=1, keepdims=True)
        b_punct = raw_punct - raw_punct.mean(axis=1, keepdims=True)
        return b_char, b_punct, gates

    def fuse_proba(self, disc_probs, b_char, b_punct, gates):
        self.char_std = max(float(np.std(b_char)), 1e-6)
        self.punct_std = max(float(np.std(b_punct)), 1e-6)
        bc_norm = b_char / self.char_std
        bp_norm = (b_punct / self.punct_std) * gates[:, None]
        log_disc = np.log(np.clip(disc_probs, 1e-7, 1.0))
        logits = log_disc + self.eta_char * bc_norm + self.eta_punct * bp_norm
        return softmax(logits, axis=1)

# 8. Pillar 4: Scale-Free Laplacian Topic Centering (Top-17 Capped + Kish's Neff Shrinkage)
def scale_free_laplacian_center(X_test, texts_test, train_texts=None, k_max=17, sim_tau=0.18, beta=4.0, kappa=2.5, chunk_size=1024):
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

# 9. Group Centering Functions (Training & Validation)
def group_center_sparse(X, groups):
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
    X_out = X.copy()
    df_g = pd.Series(groups)
    for _, idx in df_g.groupby(df_g).groups.items():
        indices = np.asarray(list(idx))
        if len(indices) > 1:
            X_out[indices] -= X_out[indices].mean(axis=0, keepdims=True)
    return X_out

# 10. Cardinality-Compensated Temperature Scaling & Marginal Prior Normalization
def calibrated_tier2_proba(probs, cluster_size, beta=0.35):
    if cluster_size <= 1:
        return np.ones((probs.shape[0], 1), dtype=np.float64)
    temp = 1.0 / (1.0 + beta * np.log(cluster_size))
    p_exp = np.power(np.clip(probs, 1e-12, 1.0), 1.0 / temp)
    return p_exp / p_exp.sum(axis=1, keepdims=True)

def normalize_hierarchical_marginals(oof_hier, test_hier, tau=0.65, alpha_dirichlet=1e-3):
    n_classes = oof_hier.shape[1]
    target_prior = 1.0 / n_classes
    emp_marginal = oof_hier.mean(axis=0)
    scale_factors = ((target_prior + alpha_dirichlet) / (emp_marginal + alpha_dirichlet)) ** tau

    oof_cal = oof_hier * scale_factors[None, :]
    oof_cal /= oof_cal.sum(axis=1, keepdims=True)

    test_cal = test_hier * scale_factors[None, :]
    test_cal /= test_cal.sum(axis=1, keepdims=True)
    return oof_cal, test_cal

# 11. Metric-Direct Coordinate-Ascent Prior Optimizer
def optimize_class_offsets(oof_probs, y_true, n_rounds=4):
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

# 12. Main Pipeline Execution
print("\n--- Phase 1: Feature Extraction (Stylometrics + Analytic-ASTD + TF-IDF) ---", flush=True)
X_style_tr_raw = extract_stylometric_matrix(train_df["text"].tolist())
X_style_te_raw = extract_stylometric_matrix(test_df["text"].tolist())
print(f"Stylometrics extracted: train={X_style_tr_raw.shape}, test={X_style_te_raw.shape}", flush=True)

print("Extracting 32-dim Analytic Contrastive ASTD features...", flush=True)
astd_model = AnalyticContrastiveASTD()
astd_model.fit(train_df["text"].tolist())
X_astd_tr_raw = astd_model.transform(train_df["text"].tolist())
X_astd_te_raw = astd_model.transform(test_df["text"].tolist())
print(f"Analytic-ASTD extracted: train={X_astd_tr_raw.shape}, test={X_astd_te_raw.shape}", flush=True)

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

# Hellinger Transform on TF-IDF (Square-Root Preconditioning)
X_tfidf_tr = sparse.hstack([X_char_tr, X_word_tr], format="csr")
X_tfidf_te = sparse.hstack([X_char_te, X_word_te], format="csr")

X_tfidf_tr.data = np.sqrt(np.maximum(X_tfidf_tr.data, 0.0))
X_tfidf_te.data = np.sqrt(np.maximum(X_tfidf_te.data, 0.0))
print(f"Total Hellinger TF-IDF Space: {X_tfidf_tr.shape[1]:,} features", flush=True)

y_all = np.array([CLASS_TO_IDX[m] for m in train_df["model"]])
prov_all = np.array([PROV_TO_IDX[PROVIDER_MAP[m]] for m in train_df["model"]])
prompts_all = train_df["prompt_id"].values

gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(train_df, y_all, groups=prompts_all))

oof_global_lr = np.zeros((len(train_df), len(CLASSES)), dtype=np.float64)
oof_global_svc = np.zeros((len(train_df), len(CLASSES)), dtype=np.float64)
oof_nl_fosp = np.zeros((len(train_df), len(CLASSES)), dtype=np.float64)
oof_hier = np.zeros((len(train_df), len(CLASSES)), dtype=np.float64)

test_global_lr = np.zeros((len(test_df), len(CLASSES)), dtype=np.float64)
test_global_svc = np.zeros((len(test_df), len(CLASSES)), dtype=np.float64)
test_nl_fosp = np.zeros((len(test_df), len(CLASSES)), dtype=np.float64)
test_hier = np.zeros((len(test_df), len(CLASSES)), dtype=np.float64)

print("\n--- Commencing 5-Fold Grouped Cross-Validation (OPERA v3.0) ---", flush=True)

for fold, (tr_idx, va_idx) in enumerate(folds):
    t_f_start = time.time()
    tr_prompts = prompts_all[tr_idx]
    va_prompts = prompts_all[va_idx]
    assert len(set(tr_prompts) & set(va_prompts)) == 0, f"Prompt leakage in fold {fold}!"

    # 1. Feature Standardization & Prompt Centering
    scaler_s = StandardScaler()
    S_tr = scaler_s.fit_transform(X_style_tr_raw[tr_idx])
    S_va = scaler_s.transform(X_style_tr_raw[va_idx])
    S_te = scaler_s.transform(X_style_te_raw)

    scaler_a = StandardScaler()
    A_tr = scaler_a.fit_transform(X_astd_tr_raw[tr_idx])
    A_va = scaler_a.transform(X_astd_tr_raw[va_idx])
    A_te = scaler_a.transform(X_astd_te_raw)

    # Combined Dense Stylometrics + ASTD
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

    # 2. Global Logistic Regression Head (High-Dimensional L2)
    clf_lr = LogisticRegression(C=2.5, max_iter=1000, random_state=42 + fold)
    clf_lr.fit(X_tr_norm, y_all[tr_idx])
    oof_global_lr[va_idx] = clf_lr.predict_proba(X_va_norm)
    test_global_lr += clf_lr.predict_proba(X_te_norm) / len(folds)

    # 3. Global Fast LinearSVC Head
    base_svc = LinearSVC(C=0.45, dual=False, max_iter=3000, random_state=42 + fold)
    base_svc.fit(X_tr_norm, y_all[tr_idx])
    svc_va_dec = base_svc.decision_function(X_va_norm)
    svc_te_dec = base_svc.decision_function(X_te_norm)
    oof_global_svc[va_idx] = softmax(svc_va_dec / 1.5, axis=1)
    test_global_svc += softmax(svc_te_dec / 1.5, axis=1) / len(folds)

    # 4. Non-Linear Kernel-FOSP (Polynomial-Cosine + RBF Mixture in RKHS)
    nl_fosp = NonLinearKernelFOSP(n_components=32, lam_p_ratio=6.67, gamma_rel=0.06, rbf_gamma=2.0)
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

    # 5. CPNL LogOP Tier-1 Provider Router
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

    # 6. Tier-2 Specialized Intra-Family Resolvers with Cardinality Temperature Scaling
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

    # Fold Blend Metric
    fold_blend = (
        0.30 * oof_global_lr[va_idx]
        + 0.20 * oof_global_svc[va_idx]
        + 0.25 * oof_nl_fosp[va_idx]
        + 0.25 * oof_hier[va_idx]
    )
    fold_ba = balanced_accuracy_score(y_all[va_idx], np.argmax(fold_blend, axis=1))
    print(f"  Fold {fold + 1}/5 Balanced Accuracy: {fold_ba:.5f} (Elapsed: {time.time() - t_f_start:.1f}s)", flush=True)

# 13. Global Dirichlet-Smoothed Marginal Prior Normalization
oof_hier_norm, test_hier_norm = normalize_hierarchical_marginals(oof_hier, test_hier, tau=0.65)

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

# 14. Prior Calibration via Coordinate Ascent on Balanced Accuracy
best_offsets, cal_ba = optimize_class_offsets(oof_ensemble, y_all, n_rounds=4)
print(f">> Calibrated 5-Fold OOF Balanced Accuracy: {cal_ba:.5f} (Net Gain: {cal_ba - raw_ba:+.4f})", flush=True)

# 15. Per-Model Recall Breakdown
oof_final_preds = np.argmax(np.log(np.clip(oof_ensemble, 1e-7, 1.0)) + best_offsets[None, :], axis=1)
recalls = recall_score(y_all, oof_final_preds, average=None)
print("\nPer-Model Recall Breakdown:", flush=True)
rec_series = pd.Series(recalls, index=CLASSES).sort_values()
for m_name, r_val in rec_series.items():
    print(f"  {m_name:24s}: {r_val:.4f}", flush=True)

# 16. Write and Verify submission.parquet
test_final_idx = np.argmax(np.log(np.clip(test_ensemble, 1e-7, 1.0)) + best_offsets[None, :], axis=1)
test_pred_labels = [IDX_TO_CLASS[i] for i in test_final_idx]

sub = pd.DataFrame({"id": test_df["id"].astype(str), "model": test_pred_labels})
out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
sub_path = os.path.join(out_dir, "submission.parquet")
sub.to_parquet(sub_path, index=False)
print(f"\nSuccessfully wrote {len(sub)} predictions to {sub_path}", flush=True)

# Verification Assertions
assert os.path.exists(sub_path)
check_sub = pd.read_parquet(sub_path)
assert list(check_sub.columns) == ["id", "model"]
assert len(check_sub) == len(test_df)
assert check_sub["model"].isin(CLASSES).all()
print("Verification passed! Preview of submission:", flush=True)
print(check_sub.head(), flush=True)
print(f"\n=== Total OPERA v3.0 Pipeline Execution Time: {time.time() - t_start:.1f}s ===", flush=True)
