"""Experiments 1: keyphrase extraction (baselines vs improved supervised ranker).

Protocol
  * Hyper-parameters are tuned on DEV with corpus statistics from TRAIN only.
  * Final systems use TRAIN+DEV as background / training data and are scored once on TEST.
Outputs (results/): keyphrase_results.csv, keyphrase_results.json, ablation.csv,
                    predictions_test.json, significance.json, tuning_log.json, pr_curve.csv
"""
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

from kpe.baselines import (Background, rank_from_scores, remove_redundant, textrank_scores,
                           tfidf_scores, yake_ranked)
from kpe.data import load_all
from kpe.evaluate import KS, evaluate, paired_bootstrap, prf_at_k
from kpe.supervised import (ALL_FEATURES, FEATURE_GROUPS, build_table, build_training_table,
                            fit_model, rank_table)

OUT = "results"
os.makedirs(OUT, exist_ok=True)
OVERLAPS = (1.0, 0.6, 0.4)
tuning_log = {}


def apply_filter(ranked, max_overlap, topn=30):
    return {k: remove_redundant(v[:topn], max_overlap) for k, v in ranked.items()}


def best_overlap(docs, ranked):
    """Pick the redundancy threshold maximising F1@10 on the given (dev) documents."""
    best = None
    for ov in OVERLAPS:
        s, _ = evaluate(docs, apply_filter(ranked, ov))
        if best is None or s["F1@10"] > best[1]:
            best = (ov, s["F1@10"])
    return best


def tune_unsupervised(dev, bg_by_mode):
    grids = {
        "TF-IDF": [dict(mode=m) for m in ("np", "ngram")],
        "TextRank": [dict(mode=m, window=w, agg=a)
                     for m, w, a in itertools.product(("np", "ngram"), (3, 6, 10), ("sum", "mean"))],
    }
    chosen = {}
    for name, grid in grids.items():
        best = None
        for params in grid:
            ranked = {}
            for d in dev:
                if name == "TF-IDF":
                    sc = tfidf_scores(d, bg_by_mode[params["mode"]], params["mode"])
                else:
                    sc = textrank_scores(d, **params)
                ranked[d.doc_id] = rank_from_scores(sc)
            ov, f = best_overlap(dev, ranked)
            tuning_log.setdefault(name, []).append({**params, "overlap": ov, "dev_F1@10": round(f, 4)})
            if best is None or f > best[0]:
                best = (f, {**params, "overlap": ov})
        chosen[name] = best[1]
        print(f"  tuned {name}: {best[1]}  dev F1@10={best[0]:.4f}")
    return chosen


def run_unsupervised(name, params, docs, bg_by_mode):
    ranked = {}
    for d in docs:
        if name == "TF-IDF":
            sc = tfidf_scores(d, bg_by_mode[params["mode"]], params["mode"])
        else:
            sc = textrank_scores(d, params["mode"], params["window"], params["agg"])
        ranked[d.doc_id] = rank_from_scores(sc)
    return apply_filter(ranked, params["overlap"])


def main():
    t0 = time.time()
    D = load_all()
    train, dev, test = D["train"], D["dev"], D["test"]

    # ------------------------------------------------------------------ DEV tuning
    print("[1/5] Tuning on DEV ...")
    bg_train = {m: Background(m).fit(train) for m in ("np", "ngram")}
    chosen = tune_unsupervised(dev, bg_train)

    # YAKE (library defaults n=3, dedup=0.9) + redundancy threshold tuned on dev
    yake_dev = {d.doc_id: yake_ranked(d) for d in dev}
    ov, f = best_overlap(dev, yake_dev)
    chosen["YAKE"] = {"n": 3, "dedup": 0.9, "overlap": ov}
    tuning_log["YAKE"] = [{"overlap": ov, "dev_F1@10": round(f, 4)}]
    print(f"  tuned YAKE: overlap={ov} dev F1@10={f:.4f}")

    # supervised ranker: tune on dev
    print("[2/5] Building feature tables and tuning supervised ranker ...")
    train_tab = build_training_table(train)
    dev_tab = build_table(dev, bg_train["ngram"])
    print(f"  train rows={len(train_tab)} (positives {train_tab.label.mean():.3%}), dev rows={len(dev_tab)}")
    sup_grid = [dict(kind="lr", C=c) for c in (0.1, 1.0)]
    sup_grid += [dict(kind="gbm", learning_rate=lr, max_depth=md, max_iter=it)
                 for lr, md, it in itertools.product((0.05, 0.1), (3, 5), (100, 300))]
    best = {"lr": None, "gbm": None}
    for params in sup_grid:
        model, feats = fit_model(train_tab, **params)
        ranked, _ = rank_table(model, feats, dev_tab)
        o, f = best_overlap(dev, ranked)
        tuning_log.setdefault("Supervised", []).append({**params, "overlap": o, "dev_F1@10": round(f, 4)})
        if best[params["kind"]] is None or f > best[params["kind"]][0]:
            best[params["kind"]] = (f, params, o)
    for k, v in best.items():
        print(f"  tuned supervised-{k}: {v[1]} overlap={v[2]} dev F1@10={v[0]:.4f}")

    # ------------------------------------------------------------------ TEST
    print("[3/5] Final evaluation on TEST (background = train+dev) ...")
    bg_all = {m: Background(m).fit(train + dev) for m in ("np", "ngram")}
    preds, results, per_doc = {}, {}, {}

    for name in ("TF-IDF", "TextRank"):
        preds[name] = run_unsupervised(name, chosen[name], test, bg_all)
    preds["YAKE"] = apply_filter({d.doc_id: yake_ranked(d) for d in test}, chosen["YAKE"]["overlap"])

    final_tab = build_training_table(train + dev)
    test_tab = build_table(test, bg_all["ngram"])
    names = {"lr": "Supervised-LR", "gbm": "Supervised-GBM (proposed)"}
    models = {}
    for kind in ("lr", "gbm"):
        f, params, o = best[kind]
        model, feats = fit_model(final_tab, **params)
        models[kind] = (model, feats)
        ranked, scored = rank_table(model, feats, test_tab)
        preds[names[kind]] = apply_filter(ranked, o)
        if kind == "gbm":
            scored.to_csv(os.path.join(OUT, "test_scored_candidates.csv.gz"), index=False)
    chosen["Supervised-LR"] = {**best["lr"][1], "overlap": best["lr"][2]}
    chosen["Supervised-GBM (proposed)"] = {**best["gbm"][1], "overlap": best["gbm"][2]}

    for name, ranked in preds.items():
        results[name], per_doc[name] = evaluate(test, ranked)

    df = pd.DataFrame(results).T
    cols = [f"{m}@{k}" for k in KS for m in ("P", "R", "F1")] + ["MRR", "MAP@15"]
    df = df[cols].round(4)
    df.to_csv(os.path.join(OUT, "keyphrase_results.csv"))
    print(df[["P@10", "R@10", "F1@5", "F1@10", "F1@15", "MRR", "MAP@15"]].to_string())

    # PR-style curve (k = 1..15)
    curve = []
    for name, ranked in preds.items():
        for k in range(1, 16):
            vals = np.array([prf_at_k(ranked.get(d.doc_id, []), d.gold, k) for d in test]).mean(axis=0)
            curve.append({"system": name, "k": k, "P": vals[0], "R": vals[1], "F1": vals[2]})
    pd.DataFrame(curve).to_csv(os.path.join(OUT, "pr_curve.csv"), index=False)

    # ------------------------------------------------------------------ significance
    print("[4/5] Significance tests (paired bootstrap on per-document F1@10) ...")
    prop = "Supervised-GBM (proposed)"
    sig = {}
    for other in [n for n in preds if n != prop]:
        for k in (5, 10, 15):
            d_, lo, hi, p = paired_bootstrap(per_doc[prop][f"F1@{k}"], per_doc[other][f"F1@{k}"])
            sig[f"{prop} vs {other} F1@{k}"] = {"mean_diff": d_, "ci95": [lo, hi], "p": p}
    json.dump(sig, open(os.path.join(OUT, "significance.json"), "w"), indent=2)
    for k, v in sig.items():
        if k.endswith("F1@10"):
            print(f"  {k}: diff={v['mean_diff']:+.4f} CI95=[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] p={v['p']:.4f}")

    # ------------------------------------------------------------------ ablation
    print("[5/5] Feature-group ablation (GBM, evaluated on TEST) ...")
    _, gparams, go = best["gbm"]
    rows = []
    full, _ = evaluate(test, preds[prop])
    rows.append({"setting": "all features", **{k: full[k] for k in ("P@10", "R@10", "F1@10", "MRR")}})
    for g, fs in FEATURE_GROUPS.items():
        keep = [f for f in ALL_FEATURES if f not in fs]
        model, feats = fit_model(final_tab, features=keep, **gparams)
        ranked, _ = rank_table(model, feats, test_tab)
        s, _ = evaluate(test, apply_filter(ranked, go))
        rows.append({"setting": f"- {g}", **{k: s[k] for k in ("P@10", "R@10", "F1@10", "MRR")}})
    for g, fs in FEATURE_GROUPS.items():
        model, feats = fit_model(final_tab, features=fs, **gparams)
        ranked, _ = rank_table(model, feats, test_tab)
        s, _ = evaluate(test, apply_filter(ranked, go))
        rows.append({"setting": f"only {g}", **{k: s[k] for k in ("P@10", "R@10", "F1@10", "MRR")}})
    # without redundancy filter
    ranked, _ = rank_table(*models["gbm"], test_tab)
    s, _ = evaluate(test, apply_filter(ranked, 1.0))
    rows.append({"setting": "no redundancy filter", **{k: s[k] for k in ("P@10", "R@10", "F1@10", "MRR")}})
    ab = pd.DataFrame(rows).round(4)
    ab.to_csv(os.path.join(OUT, "ablation.csv"), index=False)
    print(ab.to_string(index=False))

    # feature importance via permutation on a dev-sized sample of the test table is skipped (GBM has none);
    # use permutation importance on the test candidates
    from sklearn.inspection import permutation_importance
    model, feats = models["gbm"]
    sample = test_tab.sample(n=min(20000, len(test_tab)), random_state=0)
    pi = permutation_importance(model, sample[feats].values, sample["label"].values,
                                scoring="average_precision", n_repeats=3, random_state=0)
    imp = pd.DataFrame({"feature": feats, "importance": pi.importances_mean}).sort_values("importance", ascending=False)
    imp.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)

    json.dump({"settings": chosen, "results": results}, open(os.path.join(OUT, "keyphrase_results.json"), "w"), indent=2)
    json.dump(preds, open(os.path.join(OUT, "predictions_test.json"), "w"))
    json.dump(tuning_log, open(os.path.join(OUT, "tuning_log.json"), "w"), indent=2)
    print(f"Finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
