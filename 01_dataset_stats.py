"""Step 1: document the dataset (source, size, characteristics) and the candidate-generation upper bounds.

Outputs: results/dataset_stats.json, results/dataset_stats.csv
"""
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

from kpe.candidates import get_candidates
from kpe.data import load_all
from kpe.evaluate import candidate_recall, present_in_text

OUT = "results"
os.makedirs(OUT, exist_ok=True)


def main():
    D = load_all()
    rows = []
    for split, docs in D.items():
        ntok = np.array([d.n_tokens for d in docs])
        ngold = np.array([len(d.gold) for d in docs])
        glen = [len(sorted(g, key=len)[0].split()) for d in docs for g in d.gold]
        rec_np, n_np = candidate_recall(docs, "np")
        rec_ng, n_ng = candidate_recall(docs, "ngram")
        rows.append({
            "split": split, "documents": len(docs),
            "avg_tokens": round(ntok.mean(), 1), "median_tokens": float(np.median(ntok)),
            "min_tokens": int(ntok.min()), "max_tokens": int(ntok.max()),
            "avg_sentences": round(np.mean([d.n_sents for d in docs]), 2),
            "gold_keyphrases_total": int(ngold.sum()), "avg_gold_per_doc": round(ngold.mean(), 2),
            "avg_gold_words": round(np.mean(glen), 2),
            "pct_gold_1word": round(100 * np.mean(np.array(glen) == 1), 1),
            "pct_gold_2word": round(100 * np.mean(np.array(glen) == 2), 1),
            "pct_gold_3word": round(100 * np.mean(np.array(glen) == 3), 1),
            "pct_gold_4plus": round(100 * np.mean(np.array(glen) >= 4), 1),
            "pct_gold_present_in_text": round(100 * present_in_text(docs), 1),
            "np_candidate_recall": round(100 * rec_np, 1), "np_avg_candidates": round(n_np, 1),
            "ngram_candidate_recall": round(100 * rec_ng, 1), "ngram_avg_candidates": round(n_ng, 1),
            "avg_controlled_terms": round(np.mean([len(d.contr) for d in docs]), 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "dataset_stats.csv"), index=False)
    json.dump(rows, open(os.path.join(OUT, "dataset_stats.json"), "w"), indent=2)
    print(df.T.to_string(header=False))

    pos = Counter(p for d in D["train"] for p in d.pos)
    print("\nMost frequent POS tags (train):", pos.most_common(8))
    print("\nExample abstract (test):")
    d = D["test"][5]
    print(d.text[:400], "...")
    print("Gold:", [sorted(g)[0] for g in d.gold])


if __name__ == "__main__":
    main()
