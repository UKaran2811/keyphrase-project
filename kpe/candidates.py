"""Candidate keyphrase generation from POS-tagged tokens.

Two strategies are implemented:
  * 'np'    : the longest noun phrase  (JJ|NN)* NN  of every chunk (classic, high precision, low recall)
  * 'ngram' : every contiguous sub-span (1-4 words) of a chunk that ends with a noun
              (higher recall, used by the supervised ranker)
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .data import Doc, stem_words

NOUN = {"NN", "NNS", "NNP", "NNPS"}
ADJ = {"JJ", "JJR", "JJS"}
STOP = set(ENGLISH_STOP_WORDS)


@dataclass
class Cand:
    stem: str
    n_words: int
    starts: List[int] = field(default_factory=list)
    surfaces: Counter = field(default_factory=Counter)
    last_pos: Counter = field(default_factory=Counter)
    first_pos: Counter = field(default_factory=Counter)


def eligibility(doc: Doc) -> List[bool]:
    if "elig" not in doc.cache:
        out = []
        for w, p in zip(doc.words, doc.pos):
            wl = w.lower()
            out.append((p in NOUN or p in ADJ) and any(c.isalpha() for c in wl)
                       and len(wl) > 1 and wl not in STOP)
        doc.cache["elig"] = out
    return doc.cache["elig"]


def get_candidates(doc: Doc, mode: str = "ngram", max_len: int = 4) -> Dict[str, Cand]:
    key = ("cands", mode, max_len)
    if key in doc.cache:
        return doc.cache[key]
    elig = eligibility(doc)
    T = len(doc.words)
    low = [w.lower() for w in doc.words]
    cands: Dict[str, Cand] = {}

    def add(s, e):
        stem = stem_words(low[s:e])
        c = cands.get(stem)
        if c is None:
            c = cands[stem] = Cand(stem, e - s)
        c.starts.append(s)
        c.surfaces[" ".join(low[s:e])] += 1
        c.last_pos[doc.pos[e - 1]] += 1
        c.first_pos[doc.pos[s]] += 1

    i = 0
    while i < T:
        if not elig[i]:
            i += 1
            continue
        j = i
        while j < T and elig[j] and doc.sent_idx[j] == doc.sent_idx[i]:
            j += 1
        end = j
        while end > i and doc.pos[end - 1] not in NOUN:
            end -= 1
        if end > i:
            if mode == "np":
                add(max(i, end - 5), end)
            else:
                for e in range(i, end):
                    if doc.pos[e] in NOUN:
                        for s in range(max(i, e + 1 - max_len), e + 1):
                            add(s, e + 1)
        i = j
    doc.cache[key] = cands
    return cands


def ngram_stems(doc: Doc, max_n: int = 6) -> set:
    """All contiguous stemmed n-grams of the abstract (used to test whether a gold
    keyphrase is literally present in the text)."""
    if "ngrams" not in doc.cache:
        low = [w.lower() for w in doc.words]
        out = set()
        for n in range(1, max_n + 1):
            for s in range(0, len(low) - n + 1):
                if doc.sent_idx[s] == doc.sent_idx[s + n - 1]:
                    out.add(stem_words(low[s:s + n]))
        doc.cache["ngrams"] = out
    return doc.cache["ngrams"]
