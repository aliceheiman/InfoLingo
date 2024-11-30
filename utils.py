import re
import random
from collections import Counter
import math
import numpy as np


import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import brown
from nltk.tokenize import word_tokenize

import argparse
import collections
import json
import numpy as np
import os
import string
import sys
from tqdm import tqdm
import random
from collections import Counter
from nltk.corpus import stopwords
import streamlit as st

stop_words = set(stopwords.words("english"))

nltk.download("punkt_tab")
nltk.download("brown")

###################################
# File Handling Code
###################################


def load_vocab(filename):
    with open(filename, "r") as f:
        contents = f.read()
    vocab = contents.split("\n")
    return vocab


def save_result(contents, output_name):
    with open(output_name, "w") as f:
        file_contents = "\n".join(contents)
        f.write(file_contents)
        print(f"[+] Experiment stats saved to {output_name}")


def save_json(data, output_name):
    with open(output_name, "w") as json_file:
        json.dump(data, json_file, indent=4)
    print(f"[+] Experiment preds saved to {output_name}")


def load_json(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except:
        return {}


###################################
# Corpus Code
###################################


def get_word_stats(words):
    word_counts = Counter(words)
    total_words = sum(word_counts.values())
    word_probs = {word: count / total_words for word, count in word_counts.items()}
    return word_counts, word_probs


@st.cache_data
def get_corpus_word_probs(language="English"):
    if language == "English":
        words = brown.words()
        words = [
            word.lower() for word in words if word.isalnum()
        ]  # Lowercase and remove non-alphanumeric

        # Count word occurrences
        word_counts = Counter(words)
    elif language == "French":
        with open("vocab/fra_news_2023_300K/fra_news_2023_300K-words.txt") as f:
            contents = f.read().split("\n")
        word_counts = {
            row.split("\t")[1]: int(row.split("\t")[2])
            for row in contents
            if row.strip()
        }
    elif language == "Spanish":
        with open("vocab/spa_news_2023_300K/spa_news_2023_300K-words.txt") as f:
            contents = f.read().split("\n")
        word_counts = {
            row.split("\t")[1]: int(row.split("\t")[2])
            for row in contents
            if row.strip()
        }

    total_words = sum(word_counts.values())

    # Compute probabilities
    word_probs = {word: count / total_words for word, count in word_counts.items()}
    return word_probs


def mask_text(text, vocab):
    """
    Masks words in a sentence not found in the provided vocabulary,
    preserving punctuation and numbers, and matching stems.

    Args:
        sentence (str): The input sentence.
        vocab (list): A list of words forming the vocabulary.
    Returns:
        str: The masked sentence.
    """
    # Tokenize the sentence into words, punctuation, and numbers
    # tokens = re.findall(r"\b\w+\b|[^\w\s]", text)
    tokens = word_tokenize(text)

    # Mask tokens not in the stemmed vocabulary or not numbers
    masked_tokens = [
        (
            token
            if re.match(r"[^\w\s]", token) or token.isdigit()
            # or stemmer.stem(token.lower()) in stemmed_vocab # use stemmer when choosing word
            or token.lower() in vocab
            else "_"
        )
        for token in tokens
    ]

    # Reassemble the sentence while preserving punctuation
    return "".join(
        token if re.match(r"[^\w\s]", token) else f" {token}" for token in masked_tokens
    ).strip()


def tokenize_text(text):
    tokens = word_tokenize(text.lower())
    words = [word for word in tokens if word.isalnum()]
    words = [word for word in words if not word.isdigit()]
    return words


###################################
# Vocab Code
###################################


def get_vocab(pick_next_func, text, target_pct, prior_vocab=[]):
    words = tokenize_text(text)
    words = [word for word in words if word not in prior_vocab]  # prior knowledge

    word_counts, word_probs = get_word_stats(words)
    num_unique = len(word_counts)
    num_to_pick = int(target_pct * num_unique)

    vocab = pick_next_func(words, word_counts, word_probs, n=num_to_pick)
    return vocab


def pick_baseline_random(words, word_counts, word_probs, n=1):
    """Picks random next word."""
    picks = random.sample(list(word_counts.keys()), n)
    return picks


def pick_baseline_frequent(words, word_counts, word_probs, n=1):
    """Picks most frequent next word."""
    picks = word_counts.most_common(n)
    picks = [p[0] for p in picks]
    return picks


# corpus_word_probs = get_corpus_word_probs()


def pick_entropy(words, word_counts, word_probs, corpus_word_probs, n=1):
    def _compute_entropy(w_counts, w_probs):
        entropy = 0
        for w in w_counts:
            entropy += math.log2(1 / corpus_word_probs.get(w, 0.00001)) * w_probs[w]
        return entropy

    # baseline entropy of text
    baseline_entropy = _compute_entropy(word_counts, word_probs)
    deltas = []

    # remove each word in turn
    for word in word_counts:
        candidate_words = [w for w in words if w != word]
        candidate_word_counts, candidate_probs = get_word_stats(candidate_words)
        candidate_entropy = _compute_entropy(candidate_word_counts, candidate_probs)
        # print(f"H(X) after removing {word} is {candidate_entropy:.2f}")

        delta = baseline_entropy - candidate_entropy
        deltas.append((word, delta))

    deltas = sorted(deltas, key=lambda x: x[1], reverse=True)
    picks = [w[0] for w in deltas[:n]]
    return picks


def pick_kl_divergence(words, word_counts, word_probs, corpus_word_probs, n=1):
    all_words = set(corpus_word_probs.keys()).union(set(words))

    smoothing = 0.00001
    P_smoothed = {w: corpus_word_probs.get(w, 0) + smoothing for w in all_words}
    Q_smoothed = {w: word_probs.get(w, 0) + smoothing for w in all_words}

    P_total = sum(P_smoothed.values())
    Q_total = sum(Q_smoothed.values())

    P_normalized = {w: p / P_total for w, p in P_smoothed.items()}
    Q_normalized = {w: q / Q_total for w, q in Q_smoothed.items()}

    kl_divergence = 0.0
    contributions = {}
    for w in all_words:
        q = Q_normalized[w]
        p = P_normalized[w]
        if q > 0:  # Skip if Q(w) is 0 (log(0) is undefined)
            contribution = q * np.log(q / p)
            kl_divergence += contribution
            contributions[w] = contribution

    sorted_words = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    picks = [w[0] for w in sorted_words[:n]]
    return picks


def create_qa_item(qa, vocab_func, target_pct, prior_vocab):
    item = {
        "question": qa["question"],
        "context": qa["context"],
    }
    vocab = get_vocab(vocab_func, item["context"], target_pct, prior_vocab)
    item["context"] = mask_text(item["context"], vocab=vocab + prior_vocab)
    return item, vocab


###################################
# Author Analysis
###################################


def pick_top_n_bigrams(text, n=10):
    words = tokenize_text(text)
    bigrams = list(nltk.bigrams(words))
    bigram_counts = Counter(bigrams)
    return bigram_counts.most_common(n)


def pick_top_n_trigrams(text, n=10):
    words = tokenize_text(text)
    trigrams = list(nltk.trigrams(words))
    trigram_count = Counter(trigrams)
    return trigram_count.most_common(n)


def pick_top_n_zipf_deviation(text, n, stop_words=[]):
    words = tokenize_text(text)
    word_counts = Counter(words)
    sorted_counts = word_counts.most_common()
    ranks = range(1, len(sorted_counts) + 1)
    frequencies = np.array([count for _, count in sorted_counts])

    # Step 2: Calculate expected frequencies based on Zipf's law
    max_frequency = frequencies[0]
    expected_frequencies = max_frequency / ranks

    # Step 3: Compute deviations
    deviations = np.abs(frequencies - expected_frequencies)

    # Step 4: Identify top deviating words
    deviation_data = [
        (word, freq, dev) for (word, freq), dev in zip(sorted_counts, deviations)
    ]
    deviation_data = sorted(deviation_data, key=lambda x: x[2], reverse=True)

    # Print top deviating words
    top_deviating = []
    i = 0
    while len(top_deviating) < n:
        word, freq, dev = deviation_data[i]
        if word not in stop_words:
            top_deviating.append(f"Word: {word}, Frequency: {freq}, Deviation: {dev}")
        i += 1
    return top_deviating


###################################
# Eval Code
###################################


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    # Additional normalization
    s = s.replace(" _ ", " ")
    s = white_space_fix(remove_articles(remove_punc(lower(s))))
    s = [word for word in word_tokenize(s) if word not in stop_words]
    s = " ".join(s)

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores.values()) / total),
                ("f1", 100.0 * sum(f1_scores.values()) / total),
                ("total", total),
            ]
        )
    else:
        total = len(qid_list)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                ("total", total),
            ]
        )


def get_scores(gold_answers_map, pred_answers_map):
    exact_scores = {}
    f1_scores = {}
    for qid in gold_answers_map:
        if qid not in pred_answers_map:
            continue

        gold_answers = gold_answers_map[qid]
        a_pred = pred_answers_map[qid]

        exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
    return exact_scores, f1_scores


def get_gold_answers(dataset, n=None, empty=False):
    """Creates an ID-to-gold answer map. Adapted from official eval code."""
    n = len(dataset) if n is None else min(len(dataset), n)
    gold_answers_map = {}
    i = -1
    while len(gold_answers_map) < n:
        i += 1
        qa = dataset[i]
        qid = qa["id"]
        gold_answers = [
            normalize_answer(a) for a in qa["answers"]["text"] if normalize_answer(a)
        ]
        if not gold_answers:
            if empty == False:
                continue
            # For unanswerable questions, only correct answer is empty string
            gold_answers = [""]
        gold_answers_map[qid] = gold_answers
    return gold_answers_map
