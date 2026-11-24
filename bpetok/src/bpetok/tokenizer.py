from collections import Counter
from bpetok.patterns import GPT4_PATTERN
import regex as re

def get_pair_counts(ids):
    return Counter(zip(ids, ids[1:]))

def merge(ids, pair, new_id):
    newids = []
    i = 0

    while(i < len(ids)):
        if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(new_id)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

class Tokenizer:
    def __init__(self, merges, pattern=GPT4_PATTERN):
        self.merges = merges
        self.vocab = {i: bytes([i]) for i in range(256)}
        self._pat = re.compile(GPT4_PATTERN)
        for (a, b), new_id in merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

    def _encode_chunk(self, chunk_text):
        tokens = list(chunk_text.encode("utf-8"))
        while len(tokens) >= 2:
            counts = get_pair_counts(tokens)
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            new_id = self.merges[pair]
            tokens = merge(tokens, pair, new_id)
        return tokens

    def decoder(self, ids):
        tokens = b"".join(self.vocab[new_id] for new_id in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def encoder(self, text):
        ids = []
        for chunk in self._pat.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids