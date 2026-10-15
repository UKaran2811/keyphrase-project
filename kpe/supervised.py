"""Improved approach: supervised candidate ranking with rich features.

Every n-gram candidate of an abstract becomes one row of a feature table
(frequency, position, graph, corpus-prior, linguistic and structural features).
A classifier estimates P(candidate is a keyphrase); candidates are ranked by that
probability, then a redundancy filter removes near-duplicates.
"""
import math
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .baselines import Background, textrank_word_scores
from .candidates import get_candidates

FEATURE_GROUPS = {
    "frequency": ["tf", "log_tf", "tf_norm", "idf", "tfidf", "w_idf_mean", "w_idf_max"],
    "position": ["first_pos", "last_pos", "spread", "in_title", "sent_frac"],
    "graph": ["tr_sum", "tr_mean", "tr_max"],
    "prior": ["kp_prior", "seen_bg", "w_prior_mean", "w_prior_max"],
    "linguistic": ["n_words", "n_chars", "has_hyphen", "last_plural", "first_adj", "doc_len"],
    "structure": ["n_super", "n_sub"],
}
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]


def doc_rows(doc, bg, tr_window=6):
    cs = get_candidates(doc, "ngram")
    if not cs:
        return []
    ws = textrank_word_scores(doc, tr_window)
    max_w = max(ws.values()) if ws else 1.0
    T = max(doc.n_tokens, 1)
    total_occ = sum(len(c.starts) for c in cs.values())
    tl = doc.title_len
    gold_u = doc.gold_union
    keys = list(cs)
    padded = {k: f" {k} " for k in keys}
    n_super, n_sub = Counter(), Counter()
    for a in keys:
        pa = padded[a]
        for b in keys:
            if a != b and pa in padded[b]:
                n_super[a] += 1      # b is a longer phrase containing a
                n_sub[b] += 1        # a is a shorter phrase inside b
    rows = []
    for st, c in cs.items():
        words = st.split()
        tf = len(c.starts)
        starts = np.array(c.starts)
        first, last = starts.min() / T, starts.max() / T
        sents = {doc.sent_idx[s] for s in c.starts}
        wvals = [ws.get(w, 0.0) / max_w for w in words]
        widf = [bg.word_idf(w) for w in words]
        wprior = [bg.w_gold.get(w, 0) / (bg.wdf.get(w, 0) + 2) for w in words]
        idf = bg.idf(st)
        rows.append({
            "doc_id": doc.doc_id, "stem": st,
            "tf": tf, "log_tf": math.log1p(tf), "tf_norm": tf / max(total_occ, 1),
            "idf": idf, "tfidf": math.log1p(tf) * idf,
            "w_idf_mean": float(np.mean(widf)), "w_idf_max": float(np.max(widf)),
            "first_pos": first, "last_pos": last, "spread": last - first,
            "in_title": int(any(s < tl for s in c.starts)),
            "sent_frac": len(sents) / max(doc.n_sents, 1),
            "tr_sum": sum(wvals), "tr_mean": float(np.mean(wvals)), "tr_max": float(np.max(wvals)),
            "kp_prior": bg.gold_df.get(st, 0) / (bg.df.get(st, 0) + 1.0),
            "seen_bg": int(bg.df.get(st, 0) > 0),
            "w_prior_mean": float(np.mean(wprior)), "w_prior_max": float(np.max(wprior)),
            "n_words": len(words), "n_chars": len(st),
            "has_hyphen": int("-" in st),
            "last_plural": int(c.last_pos.most_common(1)[0][0] in ("NNS", "NNPS")),
            "first_adj": int(c.first_pos.most_common(1)[0][0].startswith("JJ")),
            "doc_len": doc.n_tokens,
            "n_super": n_super[st], "n_sub": n_sub[st],
            "label": int(st in gold_u),
        })
    return rows


def build_table(docs, bg):
    rows = []
    for d in docs:
        rows.extend(doc_rows(d, bg))
    return pd.DataFrame(rows)


def build_training_table(train_docs, n_folds=5):
    """Out-of-fold construction: corpus statistics (idf, keyphrase priors) used for the rows of
    a document never include that document itself, which avoids label leakage."""
    parts = []
    for f in range(n_folds):
        held = [d for i, d in enumerate(train_docs) if i % n_folds == f]
        rest = [d for i, d in enumerate(train_docs) if i % n_folds != f]
        bg = Background("ngram").fit(rest)
        parts.append(build_table(held, bg))
    return pd.concat(parts, ignore_index=True)


def make_model(kind="gbm", **params):
    if kind == "lr":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=params.get("C", 1.0), max_iter=2000,
                                                class_weight="balanced"))
    return HistGradientBoostingClassifier(
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 4),
        max_iter=params.get("max_iter", 200),
        l2_regularization=params.get("l2", 1.0),
        random_state=0)


def fit_model(train_table, kind="gbm", features=None, **params):
    features = features or ALL_FEATURES
    m = make_model(kind, **params)
    m.fit(train_table[features].values, train_table["label"].values)
    return m, features


def rank_table(model, features, table):
    """Return dict doc_id -> list of stems sorted by predicted probability (descending)."""
    table = table.copy()
    table["score"] = model.predict_proba(table[features].values)[:, 1]
    table = table.sort_values(["doc_id", "score", "stem"], ascending=[True, False, True])
    out = {}
    for doc_id, g in table.groupby("doc_id", sort=False):
        out[doc_id] = g["stem"].tolist()
    return out, table
