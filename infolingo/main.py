# main.py
import csv
import io
from collections import Counter
from importlib.resources import files
from typing import Tuple

from nltk import word_tokenize
from nltk.corpus import brown


########################
# VOCAB FUNCTIONS
########################

def get_word_stats(words: list[str]) -> Tuple[dict, dict]:
    """
    Get word count and word probability mass function (PMF) from list.
    :param words: List of words.
    :return: Word counts and word PMF.
    """
    word_counts = dict(Counter(words))
    total_words = sum(word_counts.values())
    word_probs = {word: count / total_words for word, count in word_counts.items()}
    return word_counts, word_probs


def tokenize_text(text: str) -> list[str]:
    """
    Tokenize text into words and remove special characters.
    :param text: Input text.
    :return: List of words.
    """
    tokens = word_tokenize(text.lower())
    words = [word for word in tokens if word.isalnum()]
    words = [word for word in words if not word.isdigit()]
    return words


def get_word_probs(word_counts: dict, only_alnum: bool = True) -> dict:
    """
    Turn word-count mapping into word probability mass function.
    :param word_counts: Word to occurrences dictionary mapping.
    :param only_alnum: Only keep alphanumeric words.
    :return:
    """
    if only_alnum:
        word_counts = {word.lower(): count for word, count in word_counts.items() if word.isalnum()}

    total_words = sum(word_counts.values())
    return {word: count / total_words for word, count in word_counts.items()}


########################
# FILE I/O
########################

def load_word_counts_from_file(vocab_file: str) -> dict:
    """
    Load word counts from CSV file formatted as word,occurrences.
    :param vocab_file: Filepath to vocabulary file.
    :return: Word to occurrences dictionary mapping.
    """
    with open(vocab_file, 'r') as infile:
        reader = csv.reader(infile, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        word_counts = {row[0]: int(row[1]) for row in reader}
    return word_counts


def load_word_counts_from_datafile(data_file: str) -> dict:
    """
    Load word counts from a CSV file within the package.
    :param data_file: The name of the CSV file inside the 'data' directory.
    :return: Word to occurrences dictionary mapping.
    """
    data_text = files('infolingo.data').joinpath(data_file).read_text()
    csv_file = io.StringIO(data_text)
    reader = csv.reader(csv_file, quotechar='"', quoting=csv.QUOTE_MINIMAL)
    word_counts = {row[0]: int(row[1]) for row in reader}
    return word_counts


def load_word_counts_from_language(language: str) -> dict:
    """
    Load word counts for a specific language.
    :param language: Language of text.
    :return: Word to occurrences dictionary mapping.
    """
    if language == "english":
        words = brown.words()
        return dict(Counter(words))
    elif language == "french":
        return load_word_counts_from_datafile("fra_news_2023_300K-words.csv")
    elif language == "spanish":
        return load_word_counts_from_datafile("spa_news_2023_300K-words.csv")
    else:
        raise ValueError("Language not supported.")
