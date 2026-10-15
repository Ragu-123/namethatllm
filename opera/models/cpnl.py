import numpy as np
from collections import Counter
from scipy.special import softmax
from ..config import PUNCT_SKELETON_SET

class CPNLProviderLogOP:
    """Logarithmic Opinion Pool (LogOP) with length normalization and punctuation gating for Tier-1 routing."""
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
