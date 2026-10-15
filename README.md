# Keyphrase Extraction and Topic-Based Organization of Research Documents

Mini project (NLP). The system extracts keyphrases from research abstracts, compares unsupervised baselines
(TF-IDF, TextRank, YAKE) with an improved supervised ranker, and then uses the extracted keyphrases to organize
a document collection into topics (K-Means / NMF clustering) with a readable topic index.

Everything in this folder is reproducible with one command. All numbers in `paper/paper.md` come from the files in `results/`.

## 1. Quick start

Requirements: Python 3.9 or newer, `git` (only used to download the dataset), an internet connection for the first run.

```bash
# 1. create an environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run everything (download data, experiments, figures). About 5 minutes on a CPU, no GPU needed.
bash run_all.sh                    # Windows: run_all.bat
```

Or run the steps one by one:

```bash
python download_data.py              # downloads the Inspec dataset into data/Inspec
python 01_dataset_stats.py           # dataset size/characteristics, candidate recall upper bounds
python 02_keyphrase_experiments.py   # tuning on dev, final test results, significance tests, ablation
python 03_topic_organization.py      # clustering experiments, k sweep, topic index
python 04_error_analysis.py          # error breakdowns and qualitative examples
python 05_make_figures.py            # figures into results/figures
```

No NLTK corpora and no pre-trained models need to be downloaded. The dataset already contains POS tags, and
stop words come from scikit-learn.

If `download_data.py` fails (for example git is missing), download or clone
<https://github.com/boudinfl/ake-datasets> manually and copy the folder `datasets/Inspec` to `data/Inspec`.
The final layout must be `data/Inspec/{train,dev,test,references}`.

## 2. Project structure

```
keyphrase-project/
  README.md                      this file
  requirements.txt
  run_all.sh / run_all.bat       one-command reproduction
  download_data.py               dataset download
  01_dataset_stats.py            dataset documentation
  02_keyphrase_experiments.py    baselines vs improved method, evaluation, significance, ablation
  03_topic_organization.py       topic-based organization (clustering) and evaluation
  04_error_analysis.py           error analysis
  05_make_figures.py             figures
  kpe/                           library code
    data.py                      Inspec loader (CoreNLP XML + JSON references), Porter stemming
    candidates.py                POS-pattern candidate generation ('np' and 'ngram')
    baselines.py                 TF-IDF, TextRank, YAKE wrapper, redundancy filter, corpus statistics
    supervised.py                feature extraction (28 features) and LR / gradient boosting rankers
    evaluate.py                  P/R/F1@k, MRR, MAP, candidate recall, paired bootstrap
  data/Inspec/                   (created by download_data.py)
  results/                       all outputs (tables, json, figures, topic index, error examples)
  paper/paper.md                 the research paper (also provided as a Word/PDF-ready document)
```

## 3. Dataset

| Item | Value |
|---|---|
| Name | Inspec (Hulth, 2003) |
| Source | Original: Hulth (2003), EMNLP. Used here in the CoreNLP-processed version of Boudin's `ake-datasets` repository: <https://github.com/boudinfl/ake-datasets> (folder `datasets/Inspec`) |
| Content | 2,000 abstracts of journal papers (Computer Science and Control, Information Technology), including the title as first sentence |
| Splits | train 1,000, dev 500, test 500 |
| Length | about 135 tokens (about 6 sentences) per abstract on average, range 15 to 557 |
| Gold keyphrases | free-term keyphrases assigned by professional indexers (about 9.8 per abstract, 2.3 words on average) |
| Gold in text | about 78% of gold keyphrases occur literally (after stemming) in the abstract, so about 22% are "absent" and can never be extracted |
| Preprocessing | tokenization, lemmas and POS tags are provided (Stanford CoreNLP); Porter stemming is applied to phrases |
| Extra labels | Inspec controlled-thesaurus terms (about 4.5 per abstract). They are NOT used by any extractor and serve only as an independent reference for the topic-organization evaluation |

`python 01_dataset_stats.py` prints and saves the full statistics (`results/dataset_stats.csv`).

## 4. Methodology (workflow)

Problem definition, then dataset, preprocessing, feature representation, baselines, improved method, evaluation,
comparison, error analysis, conclusion, future scope (the paper follows the same order).

1. **Preprocessing.** POS-tag based candidate generation. `np` mode keeps the longest noun phrase `(JJ|NN)* NN`;
   `ngram` mode keeps every sub-span of up to 4 words ending in a noun. Stop words, numbers and punctuation are removed.
   Phrases are lower-cased and Porter-stemmed for matching.
2. **Baselines** (unsupervised): TF-IDF (`log(1+tf) x idf`, idf from training abstracts), TextRank (PageRank on a word
   co-occurrence graph, phrase score is the sum of word scores), and YAKE (library, default settings).
3. **Improved approach.** Supervised candidate ranking. Each candidate is described by 28 features in six groups
   (frequency, position, graph, corpus prior, linguistic, structure). Logistic regression and gradient boosting
   (`HistGradientBoostingClassifier`) estimate the probability that a candidate is a keyphrase. The corpus-prior
   features are computed out-of-fold on the training data to avoid label leakage.
4. **Redundancy filter.** Optional greedy removal of near-duplicates (Jaccard overlap threshold, tuned on dev).
5. **Topic-based organization.** Documents are represented by TF-IDF of their top-10 keyphrases and clustered
   with K-Means on an LSA space and with NMF. Topics are described by their keyphrase words and a topic index is written to
   `results/topic_index.md`.
6. **Protocol.** Hyper-parameters are chosen on dev (corpus statistics from train only). Final systems are trained on
   train+dev and scored once on test.

### Evaluation metrics

* Keyphrase extraction: Precision, Recall, F1 at k = 5, 10, 15 (macro-averaged over documents, exact match on stemmed
  phrases, any listed variant of a gold keyphrase counts), MRR, MAP@15, paired bootstrap significance (10,000 resamples).
* Topic organization: NMI, ARI, V-measure against the thesaurus reference grouping, silhouette (cosine),
  Davies-Bouldin index and NPMI topic coherence. Averaged over 5 random seeds.
* Error analysis: fate of every gold keyphrase, type of every false positive, performance by abstract length,
  number of gold keyphrases and share of keyphrases present in the text.

## 5. Main results (test split, 500 abstracts)

| System | P@10 | R@10 | F1@5 | F1@10 | F1@15 | MRR |
|---|---|---|---|---|---|---|
| YAKE | 0.225 | 0.264 | 0.188 | 0.228 | 0.234 | 0.496 |
| TF-IDF | 0.348 | 0.404 | 0.282 | 0.353 | 0.366 | 0.641 |
| TextRank | 0.355 | 0.410 | 0.286 | 0.359 | 0.368 | 0.640 |
| Supervised-LR | 0.396 | 0.443 | 0.354 | 0.394 | 0.377 | 0.726 |
| **Supervised-GBM (proposed)** | **0.419** | **0.477** | **0.373** | **0.420** | **0.402** | **0.803** |

The proposed ranker improves F1@10 by +0.061 (95% CI 0.050 to 0.071) over the best baseline, TextRank (paired bootstrap, p < 0.001).
Topic organization from the proposed keyphrases reaches NMI 0.347 (K-Means, k = 8) against 0.300 for TextRank keyphrases and
0.345 for full-text TF-IDF, with an oracle (gold keyphrases) at 0.371 and a random-keyphrase lower bound at 0.169.
See `results/` for all tables and `paper/paper.md` for the full discussion. Small differences may appear on other
library versions (the runs here are seeded).

## 6. Output files

| File | Content |
|---|---|
| `results/dataset_stats.csv/json` | dataset characteristics |
| `results/keyphrase_results.csv/json` | main comparison table and the tuned settings of every system |
| `results/pr_curve.csv` | P, R, F1 for k = 1..15 |
| `results/significance.json` | paired bootstrap tests |
| `results/ablation.csv` | feature-group ablation |
| `results/feature_importance.csv` | permutation importance |
| `results/predictions_test.json` | ranked predictions of every system for every test document |
| `results/topic_results.csv`, `topic_k_sweep.csv` | clustering metrics |
| `results/topic_index.md`, `document_topics.csv` | the organized collection (topics, label words, sample documents) |
| `results/error_analysis.json`, `error_by_factor.csv`, `error_examples.md` | error analysis |
| `results/tuning_log.json` | every configuration tried on dev |
| `results/figures/*.png` | figures used in the paper |

## 7. Using the extractor on your own text

The ranker works on POS-tagged text. To apply it to new abstracts, create `kpe.data.Doc` objects (words, POS tags, sentence
index), build the feature table with `kpe.supervised.build_table`, and rank with `rank_table` (see `02_keyphrase_experiments.py`
for the exact calls). Any tagger that outputs Penn Treebank tags (for example NLTK or spaCy) can supply the tags.

## 8. Limitations (short)

Exact-match evaluation under-credits good near-miss phrases, about 22% of gold keyphrases are absent from the text and cannot be
extracted, the supervised model is trained on one domain (computing and control abstracts), and the topic evaluation uses
a thesaurus-derived reference that is itself an approximation of "true" topics. The paper discusses these in detail.

## 9. References

* Hulth, A. (2003). Improved automatic keyword extraction given more linguistic knowledge. EMNLP.
* Boudin, F. ake-datasets (benchmark collection). <https://github.com/boudinfl/ake-datasets>
* Mihalcea, R., Tarau, P. (2004). TextRank: Bringing order into texts. EMNLP.
* Campos, R., Mangaravite, V., Pasquali, A., Jorge, A., Nunes, C., Jatowt, A. (2020). YAKE! Keyword extraction from single documents using multiple local features. Information Sciences, 509.
* Witten, I. H., Paynter, G. W., Frank, E., Gutwin, C., Nevill-Manning, C. G. (1999). KEA: Practical automatic keyphrase extraction. ACM DL.
* Hasan, K. S., Ng, V. (2014). Automatic keyphrase extraction: A survey of the state of the art. ACL.
* Lee, D. D., Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. Nature, 401.
* Deerwester, S. et al. (1990). Indexing by latent semantic analysis. JASIS, 41(6).
* Roder, M., Both, A., Hinneburg, A. (2015). Exploring the space of topic coherence measures. WSDM.
* Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. JMLR, 12.
