"""Evaluation of keyphrase extraction: exact match on Porter-stemmed phrases.

A gold keyphrase is a SET of acceptable stemmed variants; it counts as found when
any variant is in the top-k predictions. Scores are macro-averaged over documents.
"""
import numpy as np

from .candidates import get_candidates, ngram_stems

KS = (5, 10, 15)


def prf_at_k(pred, gold, k):
    top = pred[:k]
    top_set = set(top)
    hits = sum(1 for g in gold if g & top_set)
    p = hits / len(top) if top else 0.0
    r = hits / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def reciprocal_rank(pred, gold):
    allowed = set().union(*gold) if gold else set()
    for i, s in enumerate(pred):
        if s in allowed:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(pred, gold, k=15):
    """AP@k where a prediction is relevant if it matches a not-yet-matched gold keyphrase."""
    matched, score, hits = set(), 0.0, 0
    for i, s in enumerate(pred[:k]):
        for gi, g in enumerate(gold):
            if gi not in matched and s in g:
                matched.add(gi)
                hits += 1
                score += hits / (i + 1)
                break
    return score / min(len(gold), k) if gold else 0.0


def evaluate(docs, ranked, ks=KS):
    """ranked: dict doc_id -> ordered list of stemmed phrases.
    Returns (summary dict, per-document dict of F1@k arrays)."""
    rows = {k: [] for k in ks}
    rr, ap = [], []
    for d in docs:
        pred = ranked.get(d.doc_id, [])
        for k in ks:
            rows[k].append(prf_at_k(pred, d.gold, k))
        rr.append(reciprocal_rank(pred, d.gold))
        ap.append(average_precision(pred, d.gold))
    summary, per_doc = {}, {}
    for k in ks:
        arr = np.array(rows[k])
        summary[f"P@{k}"], summary[f"R@{k}"], summary[f"F1@{k}"] = arr.mean(axis=0).tolist()
        per_doc[f"F1@{k}"] = arr[:, 2]
    summary["MRR"] = float(np.mean(rr))
    summary["MAP@15"] = float(np.mean(ap))
    return summary, per_doc


def candidate_recall(docs, mode):
    """Upper bound on recall of any ranker that only scores candidates of this type."""
    rec = []
    for d in docs:
        cs = set(get_candidates(d, mode))
        rec.append(sum(1 for g in d.gold if g & cs) / len(d.gold) if d.gold else 0.0)
    return float(np.mean(rec)), float(np.mean([len(get_candidates(d, mode)) for d in docs]))


def present_in_text(docs):
    """Fraction of gold keyphrases that literally occur (stemmed) in the abstract."""
    rec = []
    for d in docs:
        ng = ngram_stems(d)
        rec.append(sum(1 for g in d.gold if g & ng) / len(d.gold) if d.gold else 0.0)
    return float(np.mean(rec))


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    """Paired bootstrap on per-document scores. Returns mean diff, 95% CI, p-value (one-sided a<=b)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    diff = a - b
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = float((boots <= 0).mean())
    return float(diff.mean()), float(lo), float(hi), p
