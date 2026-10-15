"""Experiments 2: topic-based organisation of the test collection using keyphrases.

Documents are represented in five ways and clustered (K-Means on LSA space, and NMF):
  fulltext      TF-IDF of the whole abstract (classic baseline)
  TextRank-10   bag of the top-10 TextRank keyphrases
  Proposed-10   bag of the top-10 keyphrases of the supervised GBM ranker
  Gold-10       bag of author/indexer gold keyphrases (oracle upper bound)
  Random-10     10 random candidates (sanity lower bound)
Evaluation
  * extrinsic : agreement (NMI, ARI, V-measure) with an independent reference grouping built from
                the Inspec controlled-thesaurus terms (never seen by any extractor)
  * intrinsic : silhouette (cosine), Davies-Bouldin, topic coherence (NPMI on full-text co-occurrence)
Outputs: results/topic_results.csv, results/topic_k_sweep.csv, results/topic_index.md
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score, homogeneity_completeness_v_measure,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.preprocessing import normalize

from kpe.candidates import get_candidates
from kpe.data import load_all

OUT = "results"
K_MAIN = 8
SEEDS = range(5)
TOPN = 10


def kp_tokens(stems):
    return [s.replace(" ", "_") for s in stems] + [w for s in stems for w in s.split()]


def doc_bags(test, preds, rng):
    bags = {}
    bags["TextRank-10"] = [kp_tokens(preds["TextRank"][d.doc_id][:TOPN]) for d in test]
    bags["Proposed-10"] = [kp_tokens(preds["Supervised-GBM (proposed)"][d.doc_id][:TOPN]) for d in test]
    bags["Gold-10"] = [kp_tokens([sorted(g)[0] for g in d.gold][:TOPN]) for d in test]
    rnd = []
    for d in test:
        cs = sorted(get_candidates(d, "ngram"))
        pick = list(rng.choice(cs, size=min(TOPN, len(cs)), replace=False))
        rnd.append(kp_tokens(pick))
    bags["Random-10"] = rnd
    return bags


def matrices(test, bags):
    mats = {}
    tv = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.5, sublinear_tf=True, ngram_range=(1, 2))
    mats["fulltext"] = tv.fit_transform([d.text.lower() for d in test])
    for name, b in bags.items():
        v = TfidfVectorizer(analyzer=lambda x: x, min_df=2, sublinear_tf=True)
        mats[name] = v.fit_transform(b)
    return mats


def reference_labels(test, k, seed=0):
    bag = [[" ".join(sorted(t)[0].split()).replace(" ", "_") for t in d.contr] +
           [w for t in d.contr for w in sorted(t)[0].split()] for d in test]
    X = TfidfVectorizer(analyzer=lambda x: x, sublinear_tf=True).fit_transform(bag)
    Z = normalize(TruncatedSVD(n_components=30, random_state=seed).fit_transform(X))
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(Z)


def lsa(X, seed, n=50):
    n = min(n, X.shape[1] - 1, X.shape[0] - 1)
    return normalize(TruncatedSVD(n_components=n, random_state=seed).fit_transform(X))


def cluster(X, method, k, seed):
    if method == "kmeans":
        Z = lsa(X, seed)
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
        return lab, Z
    W = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=600).fit_transform(X)
    W = normalize(W + 1e-9)
    return W.argmax(axis=1), W


class Coherence:
    """NPMI topic coherence using document co-occurrence in the full text of the collection."""

    def __init__(self, test):
        cv = CountVectorizer(stop_words="english", min_df=3, binary=True)
        self.B = cv.fit_transform([d.text.lower() for d in test]).tocsc()
        self.vocab = np.array(cv.get_feature_names_out())
        self.N = self.B.shape[0]
        self.tfidf = TfidfVectorizer(stop_words="english", min_df=3, sublinear_tf=True).fit_transform(
            [d.text.lower() for d in test])

    def score(self, labels, top=10):
        vals = []
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            if len(idx) < 3:
                continue
            centroid = np.asarray(self.tfidf[idx].mean(axis=0)).ravel()
            words = np.argsort(-centroid)[:top]
            npmis = []
            for i in range(len(words)):
                for j in range(i + 1, len(words)):
                    a = self.B[:, words[i]].toarray().ravel()
                    b = self.B[:, words[j]].toarray().ravel()
                    pa, pb, pab = a.mean(), b.mean(), (a * b).mean()
                    if pab == 0:
                        npmis.append(-1.0)
                    else:
                        npmis.append(np.log(pab / (pa * pb)) / (-np.log(pab)))
            vals.append(np.mean(npmis))
        return float(np.mean(vals))


def evaluate_clustering(Z, labels, ref, coh):
    nmi = normalized_mutual_info_score(ref, labels)
    ari = adjusted_rand_score(ref, labels)
    h, c, v = homogeneity_completeness_v_measure(ref, labels)
    sil = silhouette_score(Z, labels, metric="cosine") if len(set(labels)) > 1 else 0.0
    db = davies_bouldin_score(Z, labels) if len(set(labels)) > 1 else 0.0
    return {"NMI": nmi, "ARI": ari, "V": v, "Silhouette": sil, "DaviesBouldin": db, "NPMI": coh.score(labels)}


def surface_forms(test):
    sf = defaultdict(Counter)
    for d in test:
        for st, c in get_candidates(d, "ngram").items():
            sf[st].update(c.surfaces)
    return {st: cnt.most_common(1)[0][0] for st, cnt in sf.items()}


def write_topic_index(test, preds, labels, path, k):
    """Readable topic index. Topics are described by the stemmed WORDS of the extracted keyphrases
    (phrases almost never repeat across abstracts, words do), ranked by frequency x lift."""
    from kpe.data import stem_word
    sf = surface_forms(test)
    wsurf = defaultdict(Counter)
    for d in test:
        for w in d.words:
            wsurf[stem_word(w)][w.lower()] += 1
    wsurf = {st: c.most_common(1)[0][0] for st, c in wsurf.items()}
    kps = [preds["Supervised-GBM (proposed)"][d.doc_id][:TOPN] for d in test]
    wdf_all = Counter(w for ks in kps for w in {w for s_ in ks for w in s_.split()})
    N = len(test)
    lines = [f"# Topic index of the Inspec test collection (K-Means, k={k}, keyphrases from the proposed extractor)\n"]
    order = sorted(set(labels), key=lambda c: -(labels == c).sum())
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    for rank, c in enumerate(order, 1):
        idx = np.where(labels == c)[0]
        wdf_c = Counter(w for i in idx for w in {w for s_ in kps[i] for w in s_.split()})
        scored = []
        for w, n in wdf_c.items():
            if n < 3 or wsurf.get(w, w) in ENGLISH_STOP_WORDS or len(w) < 3:
                continue
            lift = (n / len(idx)) / (wdf_all[w] / N)
            scored.append((n * np.log2(1 + lift), w, n))
        scored.sort(reverse=True)
        label_words = [wsurf.get(w, w) for _, w, _ in scored[:6]]
        # representative phrases: phrases of the cluster containing >= 2 label words (else the most frequent)
        lab_stems = {w for _, w, _ in scored[:6]}
        pc = Counter(s_ for i in idx for s_ in set(kps[i]))
        rep = [s_ for s_, _ in pc.most_common() if len(set(s_.split()) & lab_stems) >= 1][:6]
        lines.append(f"## Topic {rank}: {', '.join(label_words[:4])}  ({len(idx)} documents)\n")
        lines.append("Label words (keyphrase-word frequency): " +
                     "; ".join(f"{wsurf.get(w, w)} ({n})" for _, w, n in scored[:10]) + "\n")
        lines.append("Example keyphrases: " + "; ".join(sf.get(s_, s_) for s_ in rep) + "\n")
        lines.append("Sample documents:")
        for i in idx[:4]:
            lines.append(f"- [{test[i].doc_id}] {test[i].title}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    D = load_all()
    test = D["test"]
    preds = json.load(open(os.path.join(OUT, "predictions_test.json")))
    rng = np.random.default_rng(0)
    bags = doc_bags(test, preds, rng)
    mats = matrices(test, bags)
    coh = Coherence(test)

    # ---------------------------------------------------------------- main table (k = K_MAIN)
    ref = reference_labels(test, K_MAIN)
    rows = []
    for rep, X in mats.items():
        for method in ("kmeans", "nmf"):
            res = []
            for seed in SEEDS:
                lab, Z = cluster(X, method, K_MAIN, seed)
                res.append(evaluate_clustering(Z, lab, ref, coh))
            m = pd.DataFrame(res).mean().to_dict()
            s = pd.DataFrame(res).std().to_dict()
            rows.append({"representation": rep, "algorithm": method, **m,
                         **{f"{k}_std": v for k, v in s.items()}})
    tab = pd.DataFrame(rows).round(4)
    tab.to_csv(os.path.join(OUT, "topic_results.csv"), index=False)
    print(tab[["representation", "algorithm", "NMI", "ARI", "V", "Silhouette", "DaviesBouldin", "NPMI"]].to_string(index=False))

    # ---------------------------------------------------------------- k sweep (k-means)
    sweep = []
    for k in range(4, 17, 2):
        refk = reference_labels(test, k)
        for rep, X in mats.items():
            res = []
            for seed in SEEDS:
                lab, Z = cluster(X, "kmeans", k, seed)
                res.append(evaluate_clustering(Z, lab, refk, coh))
            sweep.append({"k": k, "representation": rep, **pd.DataFrame(res).mean().to_dict()})
    sw = pd.DataFrame(sweep).round(4)
    sw.to_csv(os.path.join(OUT, "topic_k_sweep.csv"), index=False)
    print(sw.pivot(index="k", columns="representation", values="NMI").round(3).to_string())

    # ---------------------------------------------------------------- readable topic index
    lab, _ = cluster(mats["Proposed-10"], "kmeans", K_MAIN, 0)
    write_topic_index(test, preds, lab, os.path.join(OUT, "topic_index.md"), K_MAIN)
    pd.DataFrame({"doc_id": [d.doc_id for d in test], "topic": lab, "title": [d.title for d in test]}).to_csv(
        os.path.join(OUT, "document_topics.csv"), index=False)
    print("Wrote topic index.")


if __name__ == "__main__":
    main()
