import numpy as np
from collections import Counter
from ..config import TOKEN_RE, SYLLABLE_RE, MAJOR_PUNCT

class AnalyticContrastiveASTD:
    """Zero-GPU Analytic Contrastive Autoregressive Stylometric Trajectory Dynamics."""
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
