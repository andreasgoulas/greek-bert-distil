# Andreas Goulas <goulasand@gmail.com>

import unicodedata
import numpy as np
import spacy
from tqdm.auto import tqdm
import torch.nn.functional as F

nlp = spacy.load('el_core_news_sm')

def tokenize(text, w2i):
    """Split a sentence into a sequence of tokens."""

    ids = []
    for word in nlp.tokenizer(text):
        if word.text in w2i:
            ids.append(w2i[word.text])
        else:
            ids.append(w2i['<unk>'])
    return ids

def normalize_text(s):
    """Lower and remove accents from a sentence."""

    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn').lower()

def load_vocab(filename, embed_dim):
    """Load pre-trained word embeddings."""

    words = ['PAD']
    vectors = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='loading vectors'):
            items = line.strip().split(' ')
            words.append(items[0])
            vectors[items[0]] = [float(x) for x in items[1:]]

    w2i = {word: index for index, word in enumerate(words)}
    emb_matrix = np.zeros((len(words), embed_dim), np.float32)
    for word, v in vectors.items():
        emb_matrix[w2i[word], :] = v
    return w2i, emb_matrix

def soft_ce(pred, target):
    """Cross-entropy loss with soft labels."""

    pred_log = F.log_softmax(pred, dim=-1)
    target_prob = F.softmax(target, dim=-1)
    return (-target_prob * pred_log).mean()
