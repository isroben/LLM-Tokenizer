import pytest
import regex as re

from bpetok.patterns import GPT4_PATTERN
from bpetok.tokenizer import Tokenizer, get_pair_counts, merge
from bpetok.trainer import train

## test written by AI.

SAMPLE = "the quick brown fox jumps over the lazy dog. " * 20


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def tok():
    """A small tokenizer trained on English text."""
    return Tokenizer(train(SAMPLE, 280), GPT4_PATTERN)


# ------------------------------------------------------- get_pair_counts
def test_pair_counts_basic():
    assert get_pair_counts([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}


def test_pair_counts_empty():
    assert get_pair_counts([]) == {}


def test_pair_counts_single_element():
    assert get_pair_counts([7]) == {}


def test_pair_counts_top_count_with_tie():
    ids = list("hello hello hello".encode("utf-8"))
    counts = get_pair_counts(ids)
    assert max(counts.values()) == 3
    assert counts[(104, 101)] == 3  # "he" is one of four pairs tied at 3


# ----------------------------------------------------------------- merge
def test_merge_basic():
    assert merge([1, 2, 3, 1, 2], (1, 2), 9) == [9, 3, 9]


def test_merge_pair_not_present():
    assert merge([1, 2, 3], (5, 6), 9) == [1, 2, 3]


def test_merge_overlap_three():
    assert merge([7, 7, 7], (7, 7), 9) == [9, 7]


def test_merge_overlap_four():
    assert merge([7, 7, 7, 7], (7, 7), 9) == [9, 9]


def test_merge_does_not_modify_input():
    ids = [1, 2, 3, 1, 2]
    merge(ids, (1, 2), 9)
    assert ids == [1, 2, 3, 1, 2]


def test_merge_empty():
    assert merge([], (1, 2), 9) == []


# ----------------------------------------------------------------- train
def test_train_known_example():
    merges = train("aaabdaaabac", 259)
    assert list(merges.items()) == [((97, 97), 256), ((97, 98), 257), ((256, 257), 258)]


def test_train_is_deterministic():
    assert train(SAMPLE, 290) == train(SAMPLE, 290)


def test_train_no_merges_needed():
    assert train("aaabdaaabac", 256) == {}


def test_train_empty_text():
    assert train("", 300) == {}


def test_train_no_repeated_pairs_does_not_crash():
    # Without a min_frequency option, count-1 pairs still get merged.
    # "abcdef" has 6 bytes, so at most 5 merges are possible.
    merges = train("abcdef", 300)
    assert len(merges) <= 5


def test_train_explicit_default_pattern_matches_default():
    assert train(SAMPLE, 290, GPT4_PATTERN) == train(SAMPLE, 290)


def test_train_no_token_contains_inner_space():
    # The regex keeps a space only at the START of a chunk,
    # so no learned token may have a space after its first byte.
    merges = train("hello world hello world hello world", 270)
    vocab = Tokenizer(merges).vocab
    for token_id in range(256, 256 + len(merges)):
        assert b" " not in vocab[token_id][1:]


# ------------------------------------------------------------- Tokenizer
def test_vocab_bytes():
    vocab = Tokenizer(train("aaabdaaabac", 259)).vocab
    assert vocab[256] == b"aa"
    assert vocab[257] == b"ab"
    assert vocab[258] == b"aaab"


def test_encode_known_example():
    t = Tokenizer(train("aaabdaaabac", 259))
    assert t.encoder("aaabdaaabac") == [258, 100, 258, 97, 99]


ROUNDTRIP_TEXTS = [
    "",
    "hello world",
    "hello 🙂",
    "नमस्ते नेपाल",
    "line1\nline2\n\nline3",
    "I'm sure they'll say we've 12345678 reasons",
    "    indented",
]


@pytest.mark.parametrize("text", ROUNDTRIP_TEXTS)
def test_roundtrip(tok, text):
    assert tok.decoder(tok.encoder(text)) == text


def test_no_merges_gives_raw_bytes():
    assert Tokenizer({}).encoder("hi") == [104, 105]


def test_decode_invalid_bytes_does_not_raise():
    assert Tokenizer({}).decoder([128]) == "\ufffd"


def test_encode_unseen_script_falls_back_to_bytes(tok):
    # The English-trained tokenizer never saw Devanagari bytes,
    # so they must come out as raw byte ids.
    text = "नमस्ते"
    assert tok.encoder(text) == list(text.encode("utf-8"))


def test_encode_empty(tok):
    assert tok.encoder("") == []


def test_split_matches_whole(tok):
    # " world" starts a new regex chunk, so encoding the two pieces
    # separately must give the same ids as encoding the whole string.
    assert tok.encoder("hello world") == tok.encoder("hello") + tok.encoder(" world")


def test_encode_equals_per_chunk_encoding(tok):
    text = "I'm sure they'll say we've 12345678 reasons"
    chunks = re.findall(GPT4_PATTERN, text)
    assert tok.encoder(text) == [i for c in chunks for i in tok.encoder(c)]