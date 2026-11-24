from bpetok.tokenizer import get_pair_counts, merge
from bpetok.patterns import GPT4_PATTERN
import regex as re
from collections import Counter

def train(text, vocab_size, pattern=GPT4_PATTERN):
    pat = re.compile(pattern)
    chunks = pat.findall(text)
    ids = [list(chunk.encode("utf-8")) for chunk in chunks]
    merges = {}

    for i in range(vocab_size - 256):
        counts = Counter()
        for chunk_ids in ids:
            counts.update(get_pair_counts(chunk_ids))
        if not counts:
            break
        
        pair = min(counts, key=lambda p: (-counts[p], p))
        new_id = 256 + i
        ids = [merge(chunk_ids, pair, new_id) for chunk_ids in ids]
        merges[pair] = new_id
    return merges


