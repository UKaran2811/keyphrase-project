"""Unsupervised baselines: TF-IDF, TextRank and YAKE, plus a redundancy filter."""
import math
from collections import Counter

import networkx as nx

from .candidates import eligibility, get_candidates
from .data import stem_word


# --------------------------------------------------------------------------- utils
def remove_redundant(stems, max_overlap=1.0):
    """Greedy filter: drop a phrase if its word-set Jaccard similarity with an already
    selected (higher ranked) phrase exceeds max_overlap. 1.0 disables the filter."""
    if max_overlap >= 1.0:
        return list(stems)
    kept, kept_sets = [], []
    for s in stems:
        ws = set(s.split())
        ok = True
        for ks in kept_sets:
            if len(ws & ks) / len(ws | ks) > max_overlap:
                ok = False
                break
        if ok:
            kept.append(s)
            kept_sets.append(ws)
    return kept


class Background:
    """Corpus statistics (document frequencies and keyphrase priors) from a labelled corpus."""

    def __init__(self, mode="ngram"):
        self.mode = mode

    def fit(self, docs):
        self.N = len(docs)
        self.df, self.gold_df, self.wdf, self.w_gold = Counter(), Counter(), Counter(), Counter()
        for d in docs:
            cs = get_candidates(d, self.mode)
            gu = d.gold_union
            for st in cs:
                self.df[st] += 1
                if st in gu:
                    self.gold_df[st] += 1
            words = {w for st in cs for w in st.split()}
            gold_words = {w for g in gu for w in g.split()}
            for w in words:
                self.wdf[w] += 1
                if w in gold_words:
                    self.w_gold[w] += 1
        return self

    def idf(self, stem):
        return math.log((self.N + 1) / (self.df.get(stem, 0) + 1))

    def word_idf(self, w):
        return math.log((self.N + 1) / (self.wdf.get(w, 0) + 1))


# --------------------------------------------------------------------------- TF-IDF
def tfidf_scores(doc, bg, mode):
    cs = get_candidates(doc, mode)
    return {st: math.log1p(len(c.starts)) * bg.idf(st) for st, c in cs.items()}


# --------------------------------------------------------------------------- TextRank
def textrank_word_scores(doc, window=6):
    key = ("tr", window)
    if key in doc.cache:
        return doc.cache[key]
    elig = eligibility(doc)
    toks = [stem_word(w) for w, e in zip(doc.words, elig) if e]
    G = nx.Graph()
    G.add_nodes_from(dict.fromkeys(toks))  # insertion order: deterministic
    for a in range(len(toks)):
        for b in range(a + 1, min(a + window, len(toks))):
            if toks[a] != toks[b]:
                if G.has_edge(toks[a], toks[b]):
                    G[toks[a]][toks[b]]["weight"] += 1
                else:
                    G.add_edge(toks[a], toks[b], weight=1)
    if G.number_of_nodes() == 0:
        scores = {}
    elif G.number_of_edges() == 0:
        scores = {n: 1.0 / G.number_of_nodes() for n in G.nodes}
    else:
        scores = nx.pagerank(G, alpha=0.85, weight="weight", max_iter=200)
    doc.cache[key] = scores
    return scores


def textrank_scores(doc, mode, window=6, agg="sum"):
    ws = textrank_word_scores(doc, window)
    out = {}
    for st in get_candidates(doc, mode):
        vals = [ws.get(w, 0.0) for w in st.split()]
        out[st] = sum(vals) if agg == "sum" else sum(vals) / len(vals)
    return out


# --------------------------------------------------------------------------- YAKE
def yake_ranked(doc, n=3, dedup=0.9, top=30):
    """Run the YAKE library (statistical, unsupervised) and map output to stemmed phrases."""
    import yake
    kw = yake.KeywordExtractor(lan="en", n=n, dedupLim=dedup, top=top, features=None)
    res = kw.extract_keywords(doc.text)
    res = sorted(res, key=lambda x: x[1])            # lower score = better
    out, seen = [], set()
    for phrase, _ in res:
        toks = [t for t in phrase.lower().split() if any(c.isalpha() for c in t)]
        st = " ".join(stem_word(t) for t in toks)
        if st and st not in seen:
            seen.add(st)
            out.append(st)
    return out


def rank_from_scores(scores):
    return [s for s, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]
