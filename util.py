import spacy
from spacy.lang.vi import Vietnamese
from spacy.lang.zh import Chinese

from model import Batch


def tokenize(language):
    nlp = {
        'en': spacy.load('en_core_web_sm'),
        'vi': Vietnamese(),
        'zh': Chinese()
    }

    return lambda sentence: [tok.text for tok in nlp[language](sentence)]


def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, tgt = batch.src.transpose(0, 1), batch.tgt.transpose(0, 1)
    return Batch(src, tgt, pad_idx)
