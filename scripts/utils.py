import collections
import json
import re
import string
from collections import Counter

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
nltk.download("punkt_tab")
nltk.download("brown")

stop_words = set(stopwords.words("english"))


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
