import re
import numpy as np
from ..config import FUNCTION_WORDS

def extract_stylometric_matrix(texts):
    """Extract fine-grained structural, typographic, lexical, and functional stylometrics."""
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
