from infolingo import Infolingo


def test_load_word_counts_from_file():
    il = Infolingo(language="english", custom_vocab_file="../infolingo/data/sample1.csv")
    assert il.corpus_word_counts == {
        'a': 1,
        'b': 2,
        'c': 3,
    }
    assert type(il.corpus_word_counts) == dict

    il = Infolingo(language="english", custom_vocab_file="../infolingo/data/sample2.csv")
    assert il.corpus_word_counts == {
        '1,5': 10,
        '"': 20,
        'sample': 30,
    }
    assert type(il.corpus_word_counts) == dict


def test_load_word_counts_from_language():
    il = Infolingo(language="english")
    assert il.corpus_word_counts
    assert il.corpus_probs
    assert type(il.corpus_word_counts) == dict
