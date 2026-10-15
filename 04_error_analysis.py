"""Experiments 3: error analysis of the proposed keyphrase extractor on the TEST split.

Produces results/error_analysis.json, results/error_by_factor.csv and results/error_examples.md
"""
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

from kpe.baselines import Background
from kpe.candidates import get_candidates, ngram_stems
from kpe.data import load_all
from kpe.evaluate import prf_at_k

OUT = "results"
K = 10
PROPOSED = "Supervised-GBM (proposed)"


def is_subseq(a, b):
    return f" {a} " in f" {b} "


def classify_fp(stem, gold_u, bg_df, n_docs):
    if any(is_subseq(stem, g) and stem != g for g in gold_u):
        return "sub-phrase of a gold keyphrase"
    if any(is_subseq(g, stem) and stem != g for g in gold_u):
        return "super-phrase of a gold keyphrase"
    words = set(stem.split())
    if any(words & set(g.split()) for g in gold_u):
        return "partial word overlap with gold"
    if len(words) == 1 and bg_df.get(stem, 0) / n_docs >= 0.02:
        return "generic frequent single word"
    return "other (no lexical relation to gold)"


def main():
    D = load_all()
    train, dev, test = D["train"], D["dev"], D["test"]
    preds = json.load(open(os.path.join(OUT, "predictions_test.json")))
    bg = Background("ngram").fit(train + dev)
    raw_gold = json.load(open("data/Inspec/references/test.uncontr.json"))

    # ------------------------------------------------------------ false negatives
    fn = Counter()
    gold_len_total, gold_len_found = Counter(), Counter()
    for d in test:
        ranked = preds[PROPOSED][d.doc_id]
        top = set(ranked[:K])
        cands = set(get_candidates(d, "ngram"))
        ng = ngram_stems(d)
        for g in d.gold:
            L = min(max(len(v.split()) for v in g), 4)
            gold_len_total[L] += 1
            if g & top:
                fn["found in top-10"] += 1
                gold_len_found[L] += 1
            elif not (g & ng):
                fn["absent from the abstract text"] += 1
            elif not (g & cands):
                fn["in text but not generated as candidate (POS/length/boundary)"] += 1
            else:
                fn["candidate generated but ranked below 10"] += 1
    total_gold = sum(fn.values())

    # ------------------------------------------------------------ false positives
    fp = Counter()
    n_pred = 0
    for d in test:
        gold_u = d.gold_union
        for s in preds[PROPOSED][d.doc_id][:K]:
            n_pred += 1
            if s not in gold_u:
                fp[classify_fp(s, gold_u, bg.df, bg.N)] += 1
    total_fp = sum(fp.values())

    # ------------------------------------------------------------ complementarity with TextRank
    only_prop = only_tr = both = neither = 0
    for d in test:
        a = set(preds[PROPOSED][d.doc_id][:K])
        b = set(preds["TextRank"][d.doc_id][:K])
        for g in d.gold:
            ia, ib = bool(g & a), bool(g & b)
            both += ia and ib
            only_prop += ia and not ib
            only_tr += ib and not ia
            neither += (not ia) and (not ib)

    # ------------------------------------------------------------ performance by factor
    rows = []
    for d in test:
        p, r, f = prf_at_k(preds[PROPOSED][d.doc_id], d.gold, K)
        ng = ngram_stems(d)
        rows.append({"doc_id": d.doc_id, "n_gold": len(d.gold), "n_tokens": d.n_tokens, "F1": f, "P": p, "R": r,
                     "present_frac": sum(1 for g in d.gold if g & ng) / len(d.gold)})
    df = pd.DataFrame(rows)
    df["gold_bin"] = pd.cut(df.n_gold, [0, 5, 9, 14, 100], labels=["1-5", "6-9", "10-14", "15+"])
    df["len_bin"] = pd.cut(df.n_tokens, [0, 100, 150, 200, 1000], labels=["<=100", "101-150", "151-200", ">200"])
    df["present_bin"] = pd.cut(df.present_frac, [-0.01, 0.5, 0.75, 0.99, 1.0], labels=["<=50%", "51-75%", "76-99%", "100%"])
    fac = []
    for col in ("gold_bin", "len_bin", "present_bin"):
        g = df.groupby(col, observed=True).agg(n_docs=("F1", "size"), F1=("F1", "mean"), P=("P", "mean"), R=("R", "mean"))
        for idx, r in g.iterrows():
            fac.append({"factor": col, "bin": str(idx), **r.round(4).to_dict()})
    pd.DataFrame(fac).to_csv(os.path.join(OUT, "error_by_factor.csv"), index=False)

    # ------------------------------------------------------------ qualitative examples
    surf = {}
    lines = ["# Qualitative error examples (proposed extractor, top-10, TEST split)\n",
             "`[x]` = exact (stemmed) match with a gold keyphrase. Gold keyphrases shown as annotated.\n"]
    worst = df[df.n_gold >= 5].sort_values("F1").head(5).doc_id.tolist()
    best = df[df.n_gold >= 5].sort_values("F1", ascending=False).head(2).doc_id.tolist()
    byid = {d.doc_id: d for d in test}
    for title, ids in (("Worst-scoring documents", worst), ("Best-scoring documents", best)):
        lines.append(f"## {title}\n")
        for i in ids:
            d = byid[i]
            cs = get_candidates(d, "ngram")
            gold_disp = [v[0] for v in raw_gold[i]]
            top = preds[PROPOSED][i][:K]
            pred_disp = []
            for s in top:
                sf = cs[s].surfaces.most_common(1)[0][0] if s in cs else s
                hit = any(s in g for g in d.gold)
                pred_disp.append(f"[{'x' if hit else ' '}] {sf}")
            ng = ngram_stems(d)
            absent = [raw_gold[i][j][0] for j, g in enumerate(d.gold) if not (g & ng)]
            lines.append(f"**Doc {i}: {d.title}**  (F1@10 = {df[df.doc_id == i].F1.iloc[0]:.2f})\n")
            lines.append("- Gold: " + "; ".join(gold_disp))
            lines.append("- Absent from text: " + ("; ".join(absent) if absent else "none"))
            lines.append("- Predicted: " + "; ".join(pred_disp) + "\n")
    open(os.path.join(OUT, "error_examples.md"), "w", encoding="utf-8").write("\n".join(lines))

    out = {
        "false_negative_breakdown": {k: {"count": v, "pct": round(100 * v / total_gold, 2)} for k, v in fn.items()},
        "false_positive_breakdown": {k: {"count": v, "pct": round(100 * v / total_fp, 2)} for k, v in fp.items()},
        "n_gold": total_gold, "n_predictions": n_pred, "n_false_positives": total_fp,
        "recall_by_gold_length": {str(L): {"gold": gold_len_total[L], "found": gold_len_found[L],
                                           "recall": round(gold_len_found[L] / gold_len_total[L], 4)}
                                  for L in sorted(gold_len_total)},
        "complementarity_vs_textrank": {"both": both, "only_proposed": only_prop, "only_textrank": only_tr,
                                        "neither": neither,
                                        "oracle_union_recall@10": round((both + only_prop + only_tr) / total_gold, 4)},
    }
    json.dump(out, open(os.path.join(OUT, "error_analysis.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(pd.DataFrame(fac).to_string(index=False))


if __name__ == "__main__":
    main()
