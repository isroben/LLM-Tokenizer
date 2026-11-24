from bpetok.tokenizer import get_pair_counts, merge

def train(text, vocab_size):
    tokens = text.encode("utf-8")
    ids = list(map(int, tokens))
    merges = {}

    for i in range(vocab_size - 256):
        counts = get_pair_counts(ids)
        if not counts:
            break
        
        pair = min(counts, key=lambda p: (-counts[p], p))
        new_id = 256 + i
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
    return merges