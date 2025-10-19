import re, json, numpy as np
from collections import Counter

BREAKER = re.compile(r"[\d]|[\w']+|[<\w>]+|[^\s\w]", re.UNICODE)
SPECIALS = [
    # Longest first
    "operator:", "agent:", "user:", "bot:"
    "ization", "ically", "iness","'ight", "ening", "ingly", "lessly", "fully", "ation", "tion", "soon", "land",
    "ment", "ness", "able", "ible", "ance", "ence", "ship", "ward", "wise", "ical",
    "dom", "hood", "ism", "ist", "ity", "ure", "ant", "ary", "ful", "ish", "ern",
    "ive", "ous", "ate", "ify", "ise", "ize", "ily",
    "ing", "est", "ers", "ies", "ed", "es", "ly", "ie","le","y", "s", 
    # Contractions
    "'re", "'ve", "'ll", "n't", "'st", "'s", "'d", "'m"
]
SPECIAL_SPACES = [
    ".", ",", "!", "?", ":", ";", "/", "\\", "-", "+",
]

class SimpleTokenizer:
    def __init__(self, unk_token="<unk>", pad_token="<pad>", bos_token="<bos>", eos_token="<eos>"):
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.specials = [self.pad_token, self.unk_token, self.bos_token, self.eos_token, "bot", "agent", "user", "operator"]
        self.vocab = {}
        self.inv_vocab = {}
        self.fitted = False

    def tokenize(self, text):
        tokens = []
        for match in BREAKER.finditer(text.lower()):
            word = match.group(0)
            # check for suffixes
            m = re.match(rf"(\w+?)({'|'.join(SPECIALS)})$", word, flags=re.IGNORECASE)
            if m:
                tokens.extend([m.group(1).lower(), m.group(2).lower()])
            else:
                tokens.append(word.lower())
        return tokens

    def build_vocab(self, texts, min_freq=1, max_size=None):
        counter = Counter()
        for t in texts:
            counter.update(self.tokenize(t))
        vocab_list = list(self.specials)
        for token, freq in counter.most_common():
            if freq < min_freq:
                continue
            if token in vocab_list:
                continue
            vocab_list.append(token)
            if max_size and len(vocab_list) >= max_size:
                break
        self.vocab = {tok: i for i, tok in enumerate(vocab_list)}
        self.inv_vocab = {i: tok for tok, i in self.vocab.items()}
        self.fitted = True
        return self.vocab

    @property
    def pad_id(self):
        return self.vocab[self.pad_token]

    @property
    def unk_id(self):
        return self.vocab[self.unk_token]

    @property
    def bos_id(self):
        return self.vocab[self.bos_token]

    @property
    def eos_id(self):
        return self.vocab[self.eos_token]

    def encode(self, text, add_bos=False, add_eos=False, max_len=None):
        if not self.fitted:
            raise ValueError("Tokenizer not fitted. Call build_vocab first.")
        toks = self.tokenize(text)
        ids = [self.vocab.get(t, self.unk_id) for t in toks]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        if max_len is not None:
            return ids[:max_len]
        return ids

    def decode(self, ids, skip_specials=True):
        tokens = [self.inv_vocab.get(i, self.unk_token) for i in ids]
        if skip_specials:
            tokens = [t for t in tokens if t not in self.specials]
        return " ".join(tokens)

    def save_vocab(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.inv_vocab = {int(v): k for k, v in self.vocab.items()}
        self.inv_vocab = {int(k): v for v, k in self.vocab.items()}
        self.fitted = True

    def collate_batch(self, batch_ids, pad_id=None):
        if pad_id is None:
            pad_id = self.pad_id
        max_len = max(len(x) for x in batch_ids)
        batch = np.full((len(batch_ids), max_len), pad_id, dtype=np.int64)
        mask = np.zeros((len(batch_ids), max_len), dtype=np.float32)
        for i, seq in enumerate(batch_ids):
            batch[i, :len(seq)] = seq
            mask[i, :len(seq)] = 1.0
        return batch, mask
