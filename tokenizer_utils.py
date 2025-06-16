# tokenizer_utils.py (plain text version)

import pickle
from collections import defaultdict
from tensorflow.keras.preprocessing.text import Tokenizer

def load_captions(fname):
    """Load captions from plain .txt: each line 'image_name[TAB]caption' """
    captions = defaultdict(list)
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                img, cap = parts
                cap = 'startseq ' + cap.strip() + ' endseq'
                captions[img].append(cap)
    return captions

def create_tokenizer(captions):
    lines = []
    for caps in captions.values():
        lines.extend(caps)
    tokenizer = Tokenizer(oov_token='unk')
    tokenizer.fit_on_texts(lines)
    return tokenizer

def save_tokenizer(tokenizer):
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        return pickle.load(f)

def max_length(captions):
    # captions is a list of lines, not a dict!
    return max(len(line.split()) for line in captions)

