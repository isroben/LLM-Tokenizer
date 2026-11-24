from collections import Counter

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
    def __init__(self, merges):
        self.merges = merges
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

def decode(ids, vocab):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text

def encoder(text, merges):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        count = get_pair_counts(tokens)
        pair = min(count, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens