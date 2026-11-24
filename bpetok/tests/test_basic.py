from bpetok.tokenizer import get_pair_counts, merge
from bpetok.trainer import train

def test_counts_adjacent_pairs():
    assert get_pair_counts([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}


def test_empty_list():
    assert get_pair_counts([]) == {}


def test_single_element():
    assert get_pair_counts([7]) == {}

def test_merge_basic():
    assert merge([1, 2, 3, 1, 2], (1, 2), 9) == [9, 3, 9]

def test_train_first_merges_example():
    merges = train("aaabdaaabac", 259)
    assert list(merges.items()) == [((97, 97), 256), ((97, 98), 257), ((256, 257), 258)]

def test_most_common_pair():
    ids = list("hello hello hello".encode("utf-8"))
    counts = get_pair_counts(ids)
    best = max(counts, key=counts.get)
    assert counts[best] == 3  # (e, l), (l, l), (l, o), (h, e) etc. all tie or lead