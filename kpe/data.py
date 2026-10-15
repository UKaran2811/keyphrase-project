"""Dataset loading for the Inspec benchmark (CoreNLP XML + JSON references)."""
import glob
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List

from nltk.stem.porter import PorterStemmer

_ps = PorterStemmer()
_stem_cache = {}


def stem_word(w: str) -> str:
    s = _stem_cache.get(w)
    if s is None:
        s = _ps.stem(w.lower())
        _stem_cache[w] = s
    return s


def stem_words(words) -> str:
    return " ".join(stem_word(w) for w in words)


@dataclass
class Doc:
    doc_id: str
    split: str
    words: List[str]
    pos: List[str]
    sent_idx: List[int]
    gold: List[frozenset]          # free-indexer keyphrases (stemmed, variants per keyphrase)
    contr: List[frozenset]         # controlled-vocabulary (thesaurus) terms, stemmed variants
    cache: dict = field(default_factory=dict, repr=False)

    @property
    def n_tokens(self):
        return len(self.words)

    @property
    def n_sents(self):
        return (self.sent_idx[-1] + 1) if self.sent_idx else 0

    @property
    def title_len(self):
        """Inspec abstracts start with the title as the first 'sentence'."""
        return sum(1 for s in self.sent_idx if s == 0)

    @property
    def title(self):
        return " ".join(self.words[: self.title_len])

    @property
    def text(self):
        return " ".join(self.words)

    @property
    def gold_union(self):
        if "gold_union" not in self.cache:
            self.cache["gold_union"] = set().union(*self.gold) if self.gold else set()
        return self.cache["gold_union"]


def _read_refs(root, split, annot):
    path = os.path.join(root, "references", f"{split}.{annot}.stem.json")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: [frozenset(v) for v in vs] for k, vs in raw.items()}


def load_split(root: str, split: str) -> List[Doc]:
    gold = _read_refs(root, split, "uncontr")
    contr = _read_refs(root, split, "contr")
    docs = []
    paths = glob.glob(os.path.join(root, split, "*.xml"))
    for path in sorted(paths, key=lambda p: int(os.path.basename(p)[:-4])):
        doc_id = os.path.basename(path)[:-4]
        if doc_id not in gold:
            continue
        tree = ET.parse(path)
        words, pos, sidx = [], [], []
        for si, sent in enumerate(tree.iter("sentence")):
            for tok in sent.iter("token"):
                words.append(tok.findtext("word"))
                pos.append(tok.findtext("POS"))
                sidx.append(si)
        docs.append(Doc(doc_id, split, words, pos, sidx, gold[doc_id], contr.get(doc_id, [])))
    return docs


def load_all(root="data/Inspec"):
    return {s: load_split(root, s) for s in ("train", "dev", "test")}
